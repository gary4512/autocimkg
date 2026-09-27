"""
MCP server exposing AutoCimKG competency knowledge graphs (stored in PostgreSQL/Apache AGE) as read-only tools,
e.g. for Microsoft Copilot Studio agents.

Configuration via environment variables (see README.md in this folder):
    AUTOCIMKG_DB_HOST, AUTOCIMKG_DB_PORT, AUTOCIMKG_DB_NAME, AUTOCIMKG_DB_USER, AUTOCIMKG_DB_PASSWORD
    AUTOCIMKG_GRAPH                 default graph (defaults to the latest KG version in the metadata repository)
    AUTOCIMKG_EMBEDDINGS_BASE_URL   OpenAI-compatible endpoint of the embeddings model the KG was built with
    AUTOCIMKG_EMBEDDINGS_MODEL      e.g. 'text-embedding-3-large' or 'nomic-embed-text'
    AUTOCIMKG_EMBEDDINGS_API_KEY    API key of the embeddings endpoint (if required)
    AUTOCIMKG_MCP_API_KEY           API key clients may send in the X-API-Key header (e.g. Copilot Studio)
    AUTOCIMKG_ENTRA_TENANT_ID       Microsoft Entra ID tenant; together w/ AUTOCIMKG_ENTRA_CLIENT_ID, clients may send
    AUTOCIMKG_ENTRA_CLIENT_ID       Entra access tokens issued for this app registration (e.g. Microsoft 365 Copilot)
    AUTOCIMKG_ENTRA_APP_ID_URIS     comma-separated Application ID URIs of the app registration, e.g. incl. the
                                    api://auth-... URI of an Entra SSO registration (defaults to api://<client id>)
    AUTOCIMKG_ENTRA_SCOPE           delegated scope tokens must carry (defaults to 'access_as_user')
    AUTOCIMKG_ENTRA_ROLES           comma-separated app roles accepted for app-only tokens (optional)
    AUTOCIMKG_MCP_PUBLIC_URL        public base URL, e.g. 'https://mcp.example.com' (defaults to the request's host)
    AUTOCIMKG_MCP_HOST, AUTOCIMKG_MCP_PORT
    AUTOCIMKG_MCP_ALLOWED_HOSTS     comma-separated Host headers accepted w/o authentication configured
"""

import hmac
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import anyio
import httpx
import jwt
import numpy as np
import psycopg2
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse

# labels and relationship names AutoCimKG assigns (see AutoCimKGCore.protected_kg_resources)
EXPERT, DEPARTMENT, COMPANY, TOPIC, DOCUMENT = "Expert", "Department", "Company", "Topic", "Document"
PROTECTED_LABELS = {EXPERT, DEPARTMENT, COMPANY, TOPIC, DOCUMENT}
KNOWS, WORKS_IN, PART_OF, SUBSUMES, WRITTEN_BY = "knows", "works_in", "part_of", "subsumes", "written_by"

GRAPH_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# defence in depth for run_cypher_query: write clauses and schema-qualified function calls (e.g.
# ag_catalog.drop_graph()) are rejected upfront, every transaction is rolled back (see cursor()), and the server
# should connect w/ a read-only database role (see create_readonly_role.sql)
WRITE_CLAUSE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DROP|LOAD|CALL)\b|\b[A-Za-z_]\w*\s*\.\s*[A-Za-z_]\w*\s*\(",
                          re.IGNORECASE)
STRING_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")
MAX_QUERY_ROWS = 200
CACHE_SECONDS = 300

logger = logging.getLogger("autocimkg_mcp")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


DB = dict(host=env("AUTOCIMKG_DB_HOST", "localhost"), port=int(env("AUTOCIMKG_DB_PORT", "5432")),
          dbname=env("AUTOCIMKG_DB_NAME", "postgres"), user=env("AUTOCIMKG_DB_USER", "postgres"),
          password=env("AUTOCIMKG_DB_PASSWORD"))


# ---------------------------------------------------------------- database access

@contextmanager
def cursor():
    """Yields a cursor w/ Apache AGE loaded. Its transaction is read-only and always rolled back, so nothing a query
    does is ever persisted."""
    connection = psycopg2.connect(connect_timeout=10, **DB)
    try:
        connection.autocommit = True
        with connection.cursor() as cur:
            try:
                cur.execute("LOAD 'age';")
            except psycopg2.Error:
                pass  # not allowed for non-superusers, who get AGE via session_preload_libraries instead
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor() as cur:
            cur.execute("SET search_path = ag_catalog, \"$user\", public; SET statement_timeout = '30s';")
            yield cur
    finally:
        connection.rollback()
        connection.close()


def parse_agtype(value):
    """Converts an agtype result (e.g. a vertex, list or scalar) into plain Python values."""
    if value is None:
        return None
    text = re.sub(r"::(vertex|edge|path|numeric)\b", "", str(value))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def run_cypher(graph: str, query: str, max_rows: int = 0) -> list:
    """Runs a Cypher query returning exactly one column per row (all rows, if max_rows is 0)."""
    with cursor() as cur:
        # graph name is validated against GRAPH_NAME_PATTERN; queries never contain user-supplied values except
        # in run_cypher_query, which relies on the read-only transaction
        cur.execute(f"SELECT * FROM cypher('{graph}', $$ {query} $$) AS (result agtype);")
        rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
        return [parse_agtype(row[0]) for row in rows]


def list_graph_names() -> list[str]:
    with cursor() as cur:
        cur.execute("SELECT name FROM ag_catalog.ag_graph ORDER BY name;")
        return [row[0] for row in cur.fetchall()]


def list_kg_versions() -> dict:
    """KG versions from the AutoCimKG metadata repository (empty, if MetadataIntegrator.init_db() never ran)."""
    try:
        with cursor() as cur:
            cur.execute("SELECT kg_name, agent, start_proc_ts, end_proc_ts FROM public.kg_version ORDER BY end_proc_ts;")
            return {name: {"agent": agent, "started": str(start), "finished": str(end)}
                    for name, agent, start, end in cur.fetchall()}
    except psycopg2.Error:
        return {}


def resolve_graph(graph: str) -> str:
    """Returns the requested graph or the default one, making sure it exists."""
    names = list_graph_names()
    if not graph:
        graph = env("AUTOCIMKG_GRAPH")
    if not graph:
        versions = [name for name in list_kg_versions() if name in names]
        graph = versions[-1] if versions else (names[-1] if names else "")
    if not graph:
        raise ToolError("The database does not contain any knowledge graph yet.")
    if not GRAPH_NAME_PATTERN.match(graph) or graph not in names:
        raise ToolError(f"Unknown knowledge graph '{graph}'. Available graphs: {', '.join(names) or 'none'}")
    return graph


# ---------------------------------------------------------------- in-memory graph snapshot

@dataclass
class Node:
    id: int
    label: str
    name: str
    valid: bool
    origins: list
    embedding: np.ndarray | None = None


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)          # id -> Node
    edges: list = field(default_factory=list)          # (start id, relationship name, end id, valid)
    loaded_at: float = 0.0

    def out(self, node_id: int, relationship: str, include_invalid: bool = False) -> list[Node]:
        return [self.nodes[end] for start, name, end, valid in self.edges
                if start == node_id and name == relationship and (valid or include_invalid)]

    def into(self, node_id: int, relationship: str, include_invalid: bool = False) -> list[Node]:
        return [self.nodes[start] for start, name, end, valid in self.edges
                if end == node_id and name == relationship and (valid or include_invalid)]


_graphs: dict[str, Graph] = {}
_graphs_lock = threading.Lock()


def to_embedding(value) -> np.ndarray | None:
    if not value:
        return None
    try:
        return np.array(str(value).split(","), dtype=np.float64)
    except ValueError:
        return None


def load_graph(graph: str) -> Graph:
    """Loads all nodes and relationships of a graph, cached for CACHE_SECONDS."""
    with _graphs_lock:
        cached = _graphs.get(graph)
        if cached and time.time() - cached.loaded_at < CACHE_SECONDS:
            return cached
        snapshot = Graph(loaded_at=time.time())
        nodes = run_cypher(graph, "MATCH (n) RETURN [id(n), label(n), n.name, n.invalidated_at_time, "
                                      "n.origins, n.embeddings]")
        for node_id, label, name, invalidated, origins, embedding in nodes:
            snapshot.nodes[node_id] = Node(id=node_id, label=label, name=name or "", valid=not invalidated,
                                           origins=origins or [], embedding=to_embedding(embedding))
        edges = run_cypher(graph, "MATCH (a)-[r]->(b) RETURN [id(a), type(r), id(b), r.invalidated_at_time]")
        snapshot.edges = [(start, name, end, not invalidated) for start, name, end, invalidated in edges]
        _graphs[graph] = snapshot
        return snapshot


# ---------------------------------------------------------------- semantic search

def embed(text: str) -> np.ndarray | None:
    """Embeds a text w/ the configured OpenAI-compatible embeddings endpoint (None, if not configured)."""
    base_url, model = env("AUTOCIMKG_EMBEDDINGS_BASE_URL"), env("AUTOCIMKG_EMBEDDINGS_MODEL")
    if not base_url or not model:
        return None
    api_key = env("AUTOCIMKG_EMBEDDINGS_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}", "api-key": api_key} if api_key else {}
    response = httpx.post(base_url.rstrip("/") + "/embeddings", headers=headers, timeout=30,
                          json={"model": model, "input": text.lower()})
    response.raise_for_status()
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float64)


def rank(nodes: list[Node], query: str, limit: int) -> list[tuple[Node, float]]:
    """Ranks nodes by semantic similarity to the query (or by word overlap w/o embeddings model)."""
    try:
        query_embedding = embed(query)
    except (httpx.HTTPError, KeyError, IndexError) as e:
        raise ToolError(f"The embeddings endpoint failed ({type(e).__name__}), semantic search is unavailable.")
    scored = []
    if query_embedding is not None:
        for node in nodes:
            if node.embedding is None:
                continue
            if node.embedding.shape != query_embedding.shape:
                raise ToolError("The configured embeddings model differs from the one the knowledge graph was built "
                                 f"with ({query_embedding.shape[0]} vs. {node.embedding.shape[0]} dimensions).")
            similarity = float(np.dot(node.embedding, query_embedding)
                               / (np.linalg.norm(node.embedding) * np.linalg.norm(query_embedding)))
            scored.append((node, similarity))
    else:
        words = set(re.findall(r"\w+", query.lower()))
        for node in nodes:
            overlap = len(words & set(re.findall(r"\w+", node.name.lower())))
            if overlap:
                scored.append((node, overlap / len(words)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def find_person(g: Graph, name: str, include_inactive: bool) -> list[Node]:
    """Finds experts whose name contains all words of the given name (in any order)."""
    words = re.findall(r"\w+", name.lower())
    return [node for node in g.nodes.values()
            if node.label == EXPERT and (node.valid or include_inactive)
            and all(word in node.name.lower() for word in words)]


def org_units(g: Graph, expert: Node) -> tuple[str, str]:
    departments = g.out(expert.id, WORKS_IN)
    department = departments[0] if departments else None
    companies = g.out(department.id, PART_OF) if department else []
    return (department.name if department else ""), (companies[0].name if companies else "")


# ---------------------------------------------------------------- MCP tools

mcp = MCPServer(
    name="AutoCimKG",
    description="Experts, competencies, documents and organisational units from AutoCimKG knowledge graphs.",
    instructions="Use these tools to answer questions about who knows what in the organisation: find experts for a "
                 "topic or skill, look up an expert's competencies and publications, and explore subject areas. "
                 "Names of experts and competencies are stored in lower case. Prefer find_experts and "
                 "get_expert_profile; use run_cypher_query only for questions the other tools cannot answer.",
)
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def as_text(result) -> str:
    return json.dumps(result, ensure_ascii=False, indent=1)


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def list_knowledge_graphs() -> str:
    """Lists the available competency knowledge graphs and which one is used by default."""
    versions = list_kg_versions()
    graphs = [{"graph": name, **versions.get(name, {})} for name in list_graph_names()]
    default = ""
    try:
        default = resolve_graph("")
    except ToolError:
        pass
    return as_text({"default_graph": default, "graphs": graphs})


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def get_graph_overview(graph: str = "") -> str:
    """Summarises a knowledge graph: numbers of experts, competencies, documents and relationships, plus all
    subject areas (topics), departments and companies.

    Args:
        graph: Knowledge graph name; leave empty for the default graph.
    """
    graph = resolve_graph(graph)
    g = load_graph(graph)
    valid = [node for node in g.nodes.values() if node.valid]
    labels: dict[str, int] = {}
    for node in valid:
        labels[node.label] = labels.get(node.label, 0) + 1
    relationships: dict[str, int] = {}
    for _, name, _, is_valid in g.edges:
        if is_valid:
            relationships[name] = relationships.get(name, 0) + 1
    names = lambda label: sorted(node.name for node in valid if node.label == label)
    return as_text({"graph": graph,
                    "active_experts": labels.get(EXPERT, 0),
                    "competencies": sum(count for label, count in labels.items() if label not in PROTECTED_LABELS),
                    "documents": labels.get(DOCUMENT, 0),
                    "entities_per_label": dict(sorted(labels.items(), key=lambda item: -item[1])),
                    "relationships_per_type": relationships,
                    "topics": names(TOPIC), "departments": names(DEPARTMENT), "companies": names(COMPANY)})


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def find_experts(topic_or_skill: str, graph: str = "", limit: int = 10, include_inactive: bool = False) -> str:
    """Finds experts who know a topic or skill, ranked by how closely their competencies match it (semantic search).
    Also considers subject areas (topics) that group competencies.

    Args:
        topic_or_skill: What the expert should know, e.g. 'stress testing of banks' or 'python'.
        graph: Knowledge graph name; leave empty for the default graph.
        limit: Maximum number of experts to return.
        include_inactive: Also return experts who left the organisation.
    """
    graph = resolve_graph(graph)
    g = load_graph(graph)
    candidates = [node for node in g.nodes.values()
                  if node.valid and (node.label == TOPIC or node.label not in PROTECTED_LABELS)]
    matches = rank(candidates, topic_or_skill, limit=25)

    experts: dict[int, dict] = {}
    for node, similarity in matches:
        competencies = g.out(node.id, SUBSUMES) if node.label == TOPIC else [node]
        for competency in competencies:
            for expert in g.into(competency.id, KNOWS, include_invalid=include_inactive):
                if not (expert.valid or include_inactive):
                    continue
                entry = experts.setdefault(expert.id, {"expert": expert.name, "score": 0.0, "active": expert.valid,
                                                       "matching_competencies": []})
                entry["score"] = max(entry["score"], round(similarity, 3))
                reason = competency.name if node.label != TOPIC else f"{competency.name} (topic: {node.name})"
                if reason not in entry["matching_competencies"]:
                    entry["matching_competencies"].append(reason)

    ranked = sorted(experts.items(), key=lambda item: (-item[1]["score"], -len(item[1]["matching_competencies"])))
    ranked = ranked[:limit]
    for expert_id, entry in ranked:
        entry["department"], entry["company"] = org_units(g, g.nodes[expert_id])
        entry["matching_competencies"] = entry["matching_competencies"][:8]
    return as_text({"graph": graph, "query": topic_or_skill,
                    "search": "semantic" if embed_configured() else "keyword (no embeddings model configured)",
                    "closest_competencies": [{"name": node.name, "label": node.label, "similarity": round(score, 3)}
                                             for node, score in matches[:8]],
                    "experts": [entry for _, entry in ranked]})


def embed_configured() -> bool:
    return bool(env("AUTOCIMKG_EMBEDDINGS_BASE_URL") and env("AUTOCIMKG_EMBEDDINGS_MODEL"))


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def get_expert_profile(name: str, graph: str = "", include_inactive: bool = False) -> str:
    """Returns an expert's profile: department, company, competencies (grouped by subject area) and documents
    written. The name may be partial or in any word order, e.g. 'smith' or 'john smith'.

    Args:
        name: (Part of) the expert's name.
        graph: Knowledge graph name; leave empty for the default graph.
        include_inactive: Also consider experts who left the organisation.
    """
    graph = resolve_graph(graph)
    g = load_graph(graph)
    people = find_person(g, name, include_inactive)
    if not people:
        return as_text({"graph": graph, "error": f"No expert matching '{name}' found."})
    if len(people) > 1:
        return as_text({"graph": graph, "error": f"Several experts match '{name}', please be more specific.",
                        "candidates": sorted(person.name for person in people)[:20]})

    expert = people[0]
    department, company = org_units(g, expert)
    by_topic: dict[str, list[str]] = {}
    for competency in g.out(expert.id, KNOWS):
        topics = [topic.name for topic in g.into(competency.id, SUBSUMES)] or ["(no subject area)"]
        for topic in topics:
            by_topic.setdefault(topic, []).append(competency.name)
    return as_text({"graph": graph, "expert": expert.name, "active": expert.valid,
                    "department": department, "company": company,
                    "competencies_by_topic": {topic: sorted(names) for topic, names in sorted(by_topic.items())},
                    "documents": sorted(doc.name for doc in g.into(expert.id, WRITTEN_BY))})


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def search_competencies(query: str, graph: str = "", limit: int = 15) -> str:
    """Searches competencies and subject areas (topics) semantically similar to a query, with the number of experts
    who know each of them. Useful to explore what skills exist in the organisation.

    Args:
        query: Topic or skill to search for.
        graph: Knowledge graph name; leave empty for the default graph.
        limit: Maximum number of results.
    """
    graph = resolve_graph(graph)
    g = load_graph(graph)
    candidates = [node for node in g.nodes.values()
                  if node.valid and (node.label == TOPIC or node.label not in PROTECTED_LABELS)]
    results = []
    for node, similarity in rank(candidates, query, limit):
        competencies = g.out(node.id, SUBSUMES) if node.label == TOPIC else [node]
        experts = {expert.id for competency in competencies for expert in g.into(competency.id, KNOWS)
                   if expert.valid}
        entry = {"name": node.name, "label": node.label, "similarity": round(similarity, 3), "experts": len(experts)}
        if node.label == TOPIC:
            entry["competencies"] = sorted(competency.name for competency in competencies)[:15]
        else:
            entry["topics"] = sorted(topic.name for topic in g.into(node.id, SUBSUMES))
        results.append(entry)
    return as_text({"graph": graph, "query": query, "results": results})


@mcp.tool(annotations=READ_ONLY, structured_output=False)
def run_cypher_query(cypher: str, graph: str = "") -> str:
    """Runs a read-only openCypher query (Apache AGE dialect) against a knowledge graph, for questions the other
    tools cannot answer. The query must RETURN exactly one expression per row; return a map to get several values,
    e.g. MATCH (e:Expert)-[:knows]->(c) RETURN {expert: e.name, competency: c.name} LIMIT 50.
    Schema: (Expert)-[:knows]->(competency, any other label), (Topic)-[:subsumes]->(competency),
    (Document)-[:written_by]->(Expert), (Expert)-[:works_in]->(Department)-[:part_of]->(Company), plus relationships
    between competencies. Names are lower case. Facts whose invalidated_at_time is not empty ("") are no longer valid.
    Results are limited to 200 rows.

    Args:
        cypher: The openCypher query.
        graph: Knowledge graph name; leave empty for the default graph.
    """
    graph = resolve_graph(graph)
    if "$$" in cypher:
        raise ToolError("The query must not contain '$$'.")
    if WRITE_CLAUSE.search(STRING_LITERAL.sub("''", cypher)):
        raise ToolError("Only read-only queries (MATCH ... RETURN ...) are allowed.")
    try:
        rows = run_cypher(graph, cypher, max_rows=MAX_QUERY_ROWS + 1)
    except psycopg2.errors.ReadOnlySqlTransaction:
        raise ToolError("Only read-only queries (MATCH ... RETURN ...) are allowed.")
    except psycopg2.Error as e:
        raise ToolError(f"Query failed: {str(e).strip()}")
    truncated = len(rows) > MAX_QUERY_ROWS
    # embeddings are long lists of numbers and of no use to the agent
    strip = lambda value: ({k: strip(v) for k, v in value.items() if k != "embeddings"} if isinstance(value, dict)
                           else [strip(v) for v in value] if isinstance(value, list) else value)
    return as_text({"graph": graph, "rows": [strip(row) for row in rows[:MAX_QUERY_ROWS]], "truncated": truncated})




@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------- authentication (Entra ID and/or API key)

class EntraTokenVerifier:
    """Validates access tokens Microsoft Entra ID issued for this server's app registration (v1 and v2 tokens)."""

    def __init__(self, tenant_id: str, client_id: str, app_id_uris: list[str], scope: str, roles: list[str]):
        self.tenant_id, self.scope, self.roles = tenant_id, scope, set(roles)
        self.audiences = [client_id, *app_id_uris]
        self.issuers = [f"https://login.microsoftonline.com/{tenant_id}/v2.0", f"https://sts.windows.net/{tenant_id}/"]
        self.jwks = jwt.PyJWKClient(f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
                                    cache_keys=True, lifespan=3600)

    def verify(self, token: str) -> tuple[dict | None, str]:
        """Returns the token's claims (or None) and, if rejected, the OAuth error code."""
        try:
            key = self.jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=self.audiences, issuer=self.issuers,
                                leeway=60, options={"require": ["exp", "iat", "iss", "aud", "tid"]})
        except jwt.PyJWTError as e:
            logger.info("Rejected bearer token: %s", e)
            return None, "invalid_token"
        if claims.get("tid") != self.tenant_id:
            return None, "invalid_token"
        # delegated tokens (users, e.g. via Microsoft 365 Copilot) carry scopes, app-only tokens carry app roles
        if self.scope in claims.get("scp", "").split() or self.roles & set(claims.get("roles", [])):
            return claims, ""
        return None, "insufficient_scope"


def entra_verifier() -> EntraTokenVerifier | None:
    tenant_id, client_id = env("AUTOCIMKG_ENTRA_TENANT_ID"), env("AUTOCIMKG_ENTRA_CLIENT_ID")
    if not tenant_id or not client_id:
        return None
    roles = [role.strip() for role in env("AUTOCIMKG_ENTRA_ROLES").split(",") if role.strip()]
    return EntraTokenVerifier(tenant_id, client_id, app_id_uris(),
                              env("AUTOCIMKG_ENTRA_SCOPE", "access_as_user"), roles)


def app_id_uris() -> list[str]:
    """Application ID URIs tokens may be issued for; the first one is advertised to clients."""
    uris = [uri.strip() for uri in env("AUTOCIMKG_ENTRA_APP_ID_URIS").split(",") if uri.strip()]
    return uris or [f"api://{env('AUTOCIMKG_ENTRA_CLIENT_ID')}"]


def public_url(request: Request) -> str:
    """Public base URL of the server (configured, or derived from the request, e.g. behind a dev tunnel)."""
    configured = env("AUTOCIMKG_MCP_PUBLIC_URL")
    if configured:
        return configured.rstrip("/")
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    return f"{scheme}://{host}"


async def protected_resource_metadata(request: Request) -> JSONResponse:
    """OAuth 2.0 Protected Resource Metadata (RFC 9728), telling MCP clients where to get tokens for this server."""
    if not env("AUTOCIMKG_ENTRA_TENANT_ID") or not env("AUTOCIMKG_ENTRA_CLIENT_ID"):
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse({
        "resource": f"{public_url(request)}/mcp",
        "authorization_servers": [f"https://login.microsoftonline.com/{env('AUTOCIMKG_ENTRA_TENANT_ID')}/v2.0"],
        "scopes_supported": [f"{app_id_uris()[0]}/{env('AUTOCIMKG_ENTRA_SCOPE', 'access_as_user')}"],
        "bearer_methods_supported": ["header"],
        "resource_name": "AutoCimKG",
    })


for metadata_path in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
    mcp.custom_route(metadata_path, methods=["GET"], include_in_schema=False)(protected_resource_metadata)


class AuthMiddleware:
    """Accepts requests w/ a valid Entra ID bearer token or the configured API key (X-API-Key header).
    Health check and protected resource metadata are public."""

    def __init__(self, app, api_key: str, verifier: EntraTokenVerifier | None):
        self.app, self.api_key, self.verifier = app, api_key.encode(), verifier

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] == "/health" \
                or scope["path"].startswith("/.well-known/oauth-protected-resource"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        error = "invalid_request"
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        if self.verifier and authorization[:7].lower() == "bearer ":
            claims, error = await anyio.to_thread.run_sync(self.verifier.verify, authorization[7:].strip())
            if claims:
                scope.setdefault("state", {})["user"] = claims.get("preferred_username") or claims.get("upn") \
                    or claims.get("oid") or claims.get("azp") or claims.get("appid")
                logger.info("Request by %s (Entra ID)", scope["state"]["user"])
                await self.app(scope, receive, send)
                return
        provided = headers.get(b"x-api-key", b"")
        if self.api_key and provided and hmac.compare_digest(provided, self.api_key):
            await self.app(scope, receive, send)
            return
        await self.reject(scope, receive, send, error)

    async def reject(self, scope, receive, send, error: str):
        status = 403 if error == "insufficient_scope" else 401
        headers = {}
        if self.verifier:
            metadata = f"{public_url(Request(scope))}/.well-known/oauth-protected-resource/mcp"
            headers["WWW-Authenticate"] = f'Bearer error="{error}", resource_metadata="{metadata}"'
        message = {"invalid_request": "authentication required", "invalid_token": "invalid or expired token",
                   "insufficient_scope": "token lacks the required scope or role"}[error]
        await JSONResponse({"error": error, "error_description": message}, status_code=status,
                           headers=headers)(scope, receive, send)


def create_app():
    """Builds the ASGI app: stateless Streamable HTTP w/ JSON responses (as required by Copilot Studio)."""
    api_key, verifier = env("AUTOCIMKG_MCP_API_KEY"), entra_verifier()
    allowed_hosts = [host.strip() for host in env("AUTOCIMKG_MCP_ALLOWED_HOSTS").split(",") if host.strip()]
    if allowed_hosts:
        security = TransportSecuritySettings(allowed_hosts=allowed_hosts + ["127.0.0.1:*", "localhost:*"],
                                             allowed_origins=["https://*", "http://127.0.0.1:*", "http://localhost:*"])
    elif api_key or verifier:
        # authentication already prevents DNS rebinding attacks, so any Host (tunnel, Azure, ...) is fine
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    else:
        security = None  # SDK default: localhost only
    app = mcp.streamable_http_app(stateless_http=True, json_response=True, transport_security=security,
                                  host=env("AUTOCIMKG_MCP_HOST", "127.0.0.1"))
    return AuthMiddleware(app, api_key, verifier) if api_key or verifier else app


if __name__ == "__main__":
    host, port = env("AUTOCIMKG_MCP_HOST", "127.0.0.1"), int(env("AUTOCIMKG_MCP_PORT", "8000"))
    if not env("AUTOCIMKG_MCP_API_KEY") and not entra_verifier():
        print("WARNING: neither AUTOCIMKG_MCP_API_KEY nor AUTOCIMKG_ENTRA_* is set, "
              "the server accepts unauthenticated requests.")
    uvicorn.run(create_app(), host=host, port=port)
