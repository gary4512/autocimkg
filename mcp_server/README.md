# AutoCimKG MCP Server

Exposes AutoCimKG competency knowledge graphs, stored in PostgreSQL/Apache AGE, as read-only
[Model Context Protocol](https://modelcontextprotocol.io/) tools. Agents such as Microsoft Copilot Studio agents can then
answer questions like *"Who in banking supervision knows about stress testing?"*. Building and maintaining the graphs stays
with the AutoCimKG pipeline (see the tutorial); the server only reads them.

The server is standalone. It needs neither the `autocimkg` package nor LangChain, and it runs in its own environment with
current dependencies: Python 3.10 or later (tested with 3.12, which the Docker image uses) and the packages in
`requirements.txt`. The current MCP SDK requires a newer pydantic than AutoCimKG's pinned LangChain works with, so the two
can't share one environment.

## Tools

| Tool | Purpose |
|------|---------|
| `find_experts` | Experts for a topic or skill, ranked by semantic similarity of their competencies, incl. subject areas, department and company |
| `get_expert_profile` | An expert's department, company, competencies by subject area, and documents; partial names work |
| `search_competencies` | Competencies and subject areas similar to a query, with the number of experts knowing each |
| `get_graph_overview` | Counts plus all subject areas, departments and companies of a graph |
| `list_knowledge_graphs` | Available graphs and KG versions, incl. the default |
| `run_cypher_query` | Read-only openCypher for anything else, max. 200 rows |

By default, experts who left the organisation and other invalidated facts are hidden (`include_inactive` shows experts).
Every tool accepts an optional `graph`. Without it, the server uses `AUTOCIMKG_GRAPH` or, if that is unset, the latest KG version.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOCIMKG_DB_HOST` / `_PORT` / `_NAME` | `localhost` / `5432` / `postgres` | PostgreSQL/Apache AGE database AutoCimKG writes to |
| `AUTOCIMKG_DB_USER` / `_PASSWORD` | `postgres` / – | Use the read-only role, see *Security* |
| `AUTOCIMKG_GRAPH` | latest KG version | Default graph |
| `AUTOCIMKG_EMBEDDINGS_BASE_URL` | – | OpenAI-compatible endpoint, e.g. `https://api.openai.com/v1`, `https://<resource>.openai.azure.com/openai/v1`, `http://localhost:11434/v1` (Ollama) |
| `AUTOCIMKG_EMBEDDINGS_MODEL` | – | **Must be the model the graph was built with**, e.g. `text-embedding-3-large` |
| `AUTOCIMKG_EMBEDDINGS_API_KEY` | – | Key of the embeddings endpoint |
| `AUTOCIMKG_MCP_API_KEY` | – | Key clients may send in the `X-API-Key` header (Copilot Studio) |
| `AUTOCIMKG_ENTRA_TENANT_ID` | – | Entra ID tenant. Set it with the client ID to accept Entra access tokens (Microsoft 365 Copilot, Copilot Studio) |
| `AUTOCIMKG_ENTRA_CLIENT_ID` | – | Application (client) ID of the server's app registration |
| `AUTOCIMKG_ENTRA_APP_ID_URIS` | `api://<client id>` | Comma-separated Application ID URIs accepted as token audience; add the `api://auth-…` URI of an Entra SSO registration here |
| `AUTOCIMKG_ENTRA_SCOPE` | `access_as_user` | Delegated scope that user tokens must carry |
| `AUTOCIMKG_ENTRA_ROLES` | – | Optional app roles that app-only tokens may carry instead |
| `AUTOCIMKG_MCP_PUBLIC_URL` | request's host | Public base URL, e.g. `https://mcp.example.com`, used in the OAuth metadata |
| `AUTOCIMKG_MCP_HOST` / `_PORT` | `127.0.0.1` / `8000` | Listen address (`0.0.0.0` in the Docker image) |
| `AUTOCIMKG_MCP_ALLOWED_HOSTS` | – | Only without any authentication: public host names to accept, e.g. a dev tunnel's |

A request is accepted with **either** a valid Entra token (`Authorization: Bearer …`) **or** the API key, so both kinds of
client can share one deployment.

Without an embeddings endpoint, search falls back to keyword matching.

## Run locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
export AUTOCIMKG_DB_NAME=<database> AUTOCIMKG_DB_USER=autocimkg_reader AUTOCIMKG_DB_PASSWORD=<password>
export AUTOCIMKG_MCP_API_KEY=<secret>
export AUTOCIMKG_EMBEDDINGS_BASE_URL=https://api.openai.com/v1 AUTOCIMKG_EMBEDDINGS_MODEL=text-embedding-3-large AUTOCIMKG_EMBEDDINGS_API_KEY=<key>
.venv/bin/python autocimkg_mcp.py
```

The MCP endpoint is `http://127.0.0.1:8000/mcp` (Streamable HTTP, stateless, JSON responses), and `/health` answers without a key.
For Docker, use `docker build -t autocimkg-mcp .` and pass the variables above with `-e`. From inside a container, the
database on the host is reached as `host.docker.internal`.

## Connect to Microsoft Copilot Studio

Copilot Studio calls the server from the Microsoft cloud, so it needs a public HTTPS URL.

1. **Expose the server**
   - *For testing*, use a [dev tunnel](https://learn.microsoft.com/en-us/azure/developer/dev-tunnels/):
     ```bash
     devtunnel user login
     devtunnel host -p 8000 --allow-anonymous
     ```
     The tunnel is anonymous, so the API key is what protects the server. Dev tunnels cut requests off after 60 seconds.
   - *For production*, deploy the Docker image, e.g. to Azure Container Apps with HTTPS ingress on port 8000. Store the
     secrets as Container Apps secrets, and place the database where the app can reach it: the AGE container on Azure, or
     Azure Database for PostgreSQL flexible server with the Apache AGE extension enabled.
2. **Add the tool:** in your agent, go to **Tools → Add a tool → New tool → Model Context Protocol** and fill in:
   - **Server name:** `AutoCimKG`
   - **Server description:** *Finds experts, their competencies, documents and organisational units in the organisation's competency knowledge graph.* The orchestrator uses this text to decide when to call the server.
   - **Server URL:** `https://<your host>/mcp`
   - **Authentication:** **API key**, type **Header**, header name `X-API-Key`
3. **Connect:** select **Create → Create a new connection**, enter the API key, then **Add to agent**. The agent must use
   *generative orchestration*.
4. **Test** in the agent's test pane, e.g. *"Who knows about solvency II?"* or *"What does Anna Huber work on?"*.

Instead of an API key, Copilot Studio can sign users in with Entra ID. Choose **OAuth 2.0 → Manual** in step 2 and
follow *Copilot Studio with Entra ID* below.

## Entra ID (Microsoft 365 Copilot and Copilot Studio)

Microsoft 365 Copilot declarative agents don't support API keys for MCP servers; they need Entra ID. The server validates
the users' Entra access tokens:
- signature via the tenant's published keys, RS256 only,
- issuer (v1 or v2) and `tid` of your tenant,
- audience equal to the client ID or one of the configured App ID URIs,
- the `access_as_user` scope (or a configured app role).

Unauthenticated requests get a `401` with `WWW-Authenticate: Bearer resource_metadata=…`, and the
[RFC 9728](https://www.rfc-editor.org/rfc/rfc9728) metadata is served at `/.well-known/oauth-protected-resource`.
The server only checks tokens. Users sign in with Microsoft, and the server never passes tokens on.

### 1. App registration for the server

In the Entra admin center, go to **App registrations → New registration**: *AutoCimKG MCP*, single tenant. Then:

- **Expose an API → Add a scope**:
  - accept the Application ID URI `api://<client id>`,
  - scope `access_as_user`, *Admins and users* can consent.
- **Manifest:** set `"requestedAccessTokenVersion": 2`. Optional; the server accepts v1 and v2 tokens.
- Configure the server:
  ```
  AUTOCIMKG_ENTRA_TENANT_ID=<tenant id>
  AUTOCIMKG_ENTRA_CLIENT_ID=<client id>
  AUTOCIMKG_MCP_PUBLIC_URL=https://<your host>
  ```

Who may use the agent is controlled in Entra. On the app's enterprise application, set *Assignment required* and assign
users or groups.

### 2a. Microsoft 365 Copilot with Entra SSO (recommended)

Silent sign-in, no client secret.

1. In the [Teams Developer Portal](https://dev.teams.microsoft.com), go to **Tools → Microsoft Entra SSO client ID
   registration** and register:
   - the client ID from step 1,
   - base URL `https://<your host>/mcp`, which must equal the agent's MCP server URL,
   - *Restrict usage by app*: **Any Teams app**. Binding it to a single app makes every tool call return 404.
2. The portal returns a **registration ID** and an **Application ID URI** (`api://auth-…`). In the app registration from
   step 1:
   - add that URI to `identifierUris` in the manifest (the portal UI shows only one URI),
   - add the web redirect URI `https://teams.microsoft.com/api/platform/v1.0/oAuthConsentRedirect`,
   - under **Expose an API**, pre-authorize the client `ab3be6b7-f5df-413d-ac2d-abf1e3fd9c0b` (Microsoft's enterprise
     token store) for `access_as_user`.
3. Let the server accept the new URI:
   `AUTOCIMKG_ENTRA_APP_ID_URIS=api://<client id>,api://auth-…`
4. In VS Code with the **Microsoft 365 Agents Toolkit** (6.12 or later), go to **Create a New Agent/App → Declarative Agent → Add an
   Action → Start with an MCP Server**. Enter `https://<your host>/mcp`, choose **Entra SSO** and paste the registration ID.
   The generated `ai-plugin.json` contains the runtime `RemoteMCPServer` with
   `"auth": {"type": "OAuthPluginVault", "reference_id": "<registration ID>"}`.
5. Provision, then test in Microsoft 365 Copilot, e.g. *"Who in our organisation knows about stress testing?"*.

### 2b. Microsoft 365 Copilot with OAuth (static registration)

An alternative to SSO. Users confirm a sign-in once.

1. In the app registration from step 1:
   - add a client secret,
   - add the web redirect URI `https://teams.microsoft.com/api/platform/v1.0/oAuthRedirect`.
2. In the Teams Developer Portal, go to **Tools → OAuth client registration** and fill in:
   - client ID and secret,
   - authorization endpoint `https://login.microsoftonline.com/<tenant id>/oauth2/v2.0/authorize`,
   - token and refresh endpoint `https://login.microsoftonline.com/<tenant id>/oauth2/v2.0/token`,
   - scope `api://<client id>/access_as_user offline_access` (without `offline_access`, tokens aren't refreshed),
   - base URL `https://<your host>/mcp`,
   - *Any Teams app*.
3. In the Agents Toolkit, choose **OAuth (static registration)** with the resulting registration ID. Continue as in 2a.

Entra ID doesn't support dynamic client registration, so the *dynamic registration* options don't work with it.

### Copilot Studio with Entra ID

1. In the app registration from step 1, add a client secret.
2. In the MCP wizard, choose **OAuth 2.0 → Manual** and fill in:
   - client ID and secret,
   - authorization URL `https://login.microsoftonline.com/<tenant id>/oauth2/v2.0/authorize`,
   - token URL and refresh URL `https://login.microsoftonline.com/<tenant id>/oauth2/v2.0/token`,
   - scopes `api://<client id>/access_as_user offline_access`.
3. After **Create**, the wizard shows a **callback URL**. Add it to the app registration as a web redirect URI.

## Security

- **Use a read-only database role.** [`create_readonly_role.sql`](create_readonly_role.sql) creates `autocimkg_reader`, which
  can read all current and future graphs:
  ```bash
  psql -U <owner> -d <database> -v reader_password="'<password>'" -f create_readonly_role.sql
  ```
- **Use Apache AGE 1.7.0 or later.** Older versions don't enforce PostgreSQL privileges or read-only transactions for
  Cypher `SET` and `DELETE`, so even a read-only role can modify graphs there. This was fixed in
  [apache/age#2309](https://github.com/apache/age/pull/2309); the Docker image `apache/age:release_PG18_1.8.0` works.
  AutoCimKG itself runs unchanged on AGE 1.8.0 / PostgreSQL 18.
- **Defence in depth:** the server doesn't rely on the database alone, which also protects installations on older AGE versions. It
  - rejects write clauses and schema-qualified function calls in `run_cypher_query`,
  - runs every query in a read-only transaction that is always rolled back.
- Configure Entra ID and/or `AUTOCIMKG_MCP_API_KEY` whenever the server is reachable from outside the machine. Without
  either and without `AUTOCIMKG_MCP_ALLOWED_HOSTS`, the server only accepts requests addressed to `localhost`.
- Prefer Entra ID over the API key where the client supports it. Access is then tied to individual users, can be restricted
  to assigned users or groups, and the server logs each request's user (`preferred_username`/`upn`).
- Every user who may use the agent can read the whole graph; there is no per-user filtering of results.
- Tool results contain names of people and organisational data. Check that your Copilot Studio environment's
  data policies (DLP) and the agent's audience fit that.
