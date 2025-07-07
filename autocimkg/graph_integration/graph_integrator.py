from datetime import datetime

import age
import numpy as np
from ..models import KnowledgeGraph, Relationship, Entity
import logging
import sys
from importlib import reload
from typing import Union

class GraphIntegrator:
    """
    Designed to integrate and manage graph data in a PostgreSQL (and Apache AGE) database.
    """

    def __init__(self, host: str, port: int, dbname: str, username: str, password: str):
        """
        Initializes the GraphIntegrator with database connection parameters.
        
        :param host: URI for the database
        :param port: Port for the database
        :param dbname: Database name
        :param username: Username for database access
        :param password: Password for database access
        """

        self.host = host
        self.port = port
        self.dbname = dbname
        self.username = username
        self.password = password

        # stdout logger
        reload(logging)
        self.logger = logging.getLogger("autocimkg")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s: %(message)s',
                                      datefmt="%Y-%m-%d %H:%M:%S")
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)

    def create_graph(self, graph_name: str):
        """
        Creates a named graph in the database.

        :param graph_name: Graph name in the PGSQL/AAGE database
        """
        connection = None
        try:
            # implicitly creates kg (if not already present) ...
            connection = age.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.username,
                                     password=self.password, graph=graph_name)
            connection.commit()
        except Exception:
            self.logger.exception("PostgreSQL communication failed")
            if connection: connection.rollback()
        finally:
            if connection: connection.close()

    def delete_graph(self, graph_name: str):
        """
        Deletes a named graph in the database.

        :param graph_name: Graph name in the PGSQL/AAGE database
        """
        connection = None
        try:
            connection = age.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.username,
                                     password=self.password, graph=graph_name)
            age.deleteGraph(connection.connection, graph_name)
            connection.commit()
        except Exception as e:
            self.logger.exception("PostgreSQL communication failed")
            if connection: connection.rollback()
        finally:
            if connection: connection.close()

    def read_graph(self, graph_name: str) -> KnowledgeGraph:
        """
        Runs the necessary queries to read a graph structure from the PGSQL/AAGE database.

        :param graph_name: Graph name in the PGSQL/AAGE database
        :returns: KnowledgeGraph containing the graph structure
        """

        connection = None
        cursor = None
        try:
            connection = age.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.username,
                                     password=self.password, graph=graph_name)
            cursor = connection.connection.cursor()

            # basic assumption: reading all (!) nodes and all (!) relations leads to consistent result!

            # NODES
            cursor.execute("SELECT * from cypher(%s, $$ MATCH (n) RETURN n $$) as (v agtype);", (graph_name,))
            # execCypher() and cypher() do not work due to invalid escaping of special characters (e.g. umlauts) ...
            entities = []
            for row in cursor:
                v = row[0]

                entity = Entity(label=v.label, name=v["name"])
                entity.properties.embeddings = GraphIntegrator.transform_str_list_to_embeddings(v["embeddings"])
                entity.properties.generated_at_time = GraphIntegrator.transform_str_to_datetime(v["generated_at_time"])
                entity.properties.invalidated_at_time = GraphIntegrator.transform_str_to_datetime(v["invalidated_at_time"])
                entity.properties.agents = v["agents"]
                entity.properties.origins = v["origins"]
                # test
                entities.append(entity)

            # EDGES
            cursor.execute("SELECT * from cypher(%s, $$ MATCH p=()-[]->() RETURN p $$) as (p agtype);", (graph_name,))
            # execCypher() and cypher() do not work due to invalid escaping of special characters (e.g. umlauts) ...
            relationships = []
            for row in cursor:
                path = row[0]

                v1 = path[0]
                entity1 = Entity(label=v1.label, name=v1["name"])
                entity1.properties.embeddings = GraphIntegrator.transform_str_list_to_embeddings(v1["embeddings"])
                entity1.properties.generated_at_time = GraphIntegrator.transform_str_to_datetime(v1["generated_at_time"])
                entity1.properties.invalidated_at_time = GraphIntegrator.transform_str_to_datetime(v1["invalidated_at_time"])
                entity1.properties.agents = v1["agents"]
                entity1.properties.origins = v1["origins"]
                v2 = path[2]
                entity2 = Entity(label=v2.label, name=v2["name"])
                entity2.properties.embeddings = GraphIntegrator.transform_str_list_to_embeddings(v2["embeddings"])
                entity2.properties.generated_at_time = GraphIntegrator.transform_str_to_datetime(v2["generated_at_time"])
                entity2.properties.invalidated_at_time = GraphIntegrator.transform_str_to_datetime(v2["invalidated_at_time"])
                entity2.properties.agents = v2["agents"]
                entity2.properties.origins = v2["origins"]

                e = path[1]
                relationship = Relationship(startEntity=entity1,
                                            endEntity=entity2, name=e.label)
                relationship.properties.embeddings = GraphIntegrator.transform_str_list_to_embeddings(e["embeddings"])
                relationship.properties.generated_at_time = GraphIntegrator.transform_str_to_datetime(e["generated_at_time"])
                relationship.properties.invalidated_at_time = GraphIntegrator.transform_str_to_datetime(e["invalidated_at_time"])
                relationship.properties.agents = e["agents"]
                relationship.properties.origins = e["origins"]
                relationships.append(relationship)

            # = KG
            knowledge_graph = KnowledgeGraph(relationships=relationships, entities=entities)

            return knowledge_graph

        except Exception as e:
            self.logger.exception("PostgreSQL communication failed")
            if connection: connection.rollback()  # even w/o change!
        finally:
            if cursor: cursor.close()
            if connection: connection.close()

    def write_graph(self, graph_name: str, knowledge_graph: KnowledgeGraph) -> None:
        """
        Runs the necessary queries to write a graph structure into the PGSQL/AAGE database.

        :param graph_name: Graph name in the PGSQL/AAGE database
        :param knowledge_graph: KnowledgeGraph containing the graph structure
        """

        node_write_queries, relationship_write_queries = (
            self.create_node_write_queries(knowledge_graph=knowledge_graph),
            self.create_relationship_write_queries(knowledge_graph=knowledge_graph),
        )

        connection = None
        cursor = None
        try:
            connection = age.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.username,
                                     password=self.password, graph=graph_name)
            cursor = connection.connection.cursor()

            for node_write_query in node_write_queries:
                cursor.execute(f"SELECT * from cypher(%s, $$ {node_write_query} $$) as (v agtype);", (graph_name,))
                # execCypher() and cypher() do not work due to invalid escaping of special characters (e.g. umlauts) ...
            for relationship_write_query in relationship_write_queries:
                cursor.execute(f"SELECT * from cypher(%s, $$ {relationship_write_query} $$) as (v agtype);", (graph_name,))
                # execCypher() and cypher() do not work due to invalid escaping of special characters (e.g. umlauts) ...

            connection.commit()
        except Exception as e:
            self.logger.exception("PostgreSQL communication failed")
            if connection: connection.rollback()
        finally:
            if cursor: cursor.close()
            if connection: connection.close()

    @staticmethod
    def create_node_write_queries(knowledge_graph: KnowledgeGraph) -> list[str]:
        """
        Constructs Cypher queries for creating nodes in the graph database from a KnowledgeGraph object.

        :param knowledge_graph: KnowledgeGraph containing entities
        :returns: List of Cypher queries for node creation
        """

        queries = []
        for node in knowledge_graph.entities:
            properties = []
            for prop, value in node.properties.model_dump().items():
                if prop == "embeddings":
                    value = GraphIntegrator.transform_embeddings_to_str_list(value)
                elif prop == "generated_at_time":
                    value = GraphIntegrator.transform_datetime_to_str(value)
                elif prop == "invalidated_at_time":
                    value = GraphIntegrator.transform_datetime_to_str(value)

                if isinstance(value, list): properties.append(f'SET n.{prop.replace(" ", "_")} = {value}')
                else: properties.append(f'SET n.{prop.replace(" ", "_")} = "{value}"')

            query = f'CREATE (n:{node.label} {{name: "{node.name}"}}) ' + ' '.join(properties)
            queries.append(query)
        return queries

    @staticmethod
    def create_relationship_write_queries(knowledge_graph: KnowledgeGraph) -> list:
        """
        Constructs Cypher queries for creating relationships in the graph database from a KnowledgeGraph object.

        :param knowledge_graph: KnowledgeGraph containing relationships
        :returns: List of Cypher queries for relationship creation
        """

        rels = []
        for rel in knowledge_graph.relationships:
            properties = []
            for key, value in rel.properties.model_dump().items():
                if key == "embeddings":
                    value = GraphIntegrator.transform_embeddings_to_str_list(value)
                elif key == "generated_at_time":
                    value = GraphIntegrator.transform_datetime_to_str(value)
                elif key == "invalidated_at_time":
                    value = GraphIntegrator.transform_datetime_to_str(value)

                if isinstance(value, list): properties.append(f'SET r.{key.replace(" ", "_")} = {value}')
                else: properties.append(f'SET r.{key.replace(" ", "_")} = "{value}"')

            query = (
                f'MATCH (n:{rel.startEntity.label} {{name: "{rel.startEntity.name}"}}), '
                f'(m:{rel.endEntity.label} {{name: "{rel.endEntity.name}"}}) '
                f'CREATE (n)-[r:{rel.name}]->(m) ' + ' '.join(properties)
            )
            rels.append(query)

        return rels

    @staticmethod
    def transform_datetime_to_str(ts: Union[datetime, None]) -> str:
        """
        Transforms a datetime object to a string representation.
        Format: %Y:%M:%D %h%m%s.%f (e.g. 2025-02-28 14:55:02.775121)

        :param ts: Datetime object (or None)
        :return: String representation of datetime object
        """

        if ts is None: return ""
        return ts.__str__()

    @staticmethod
    def transform_str_to_datetime(ts_str: str) -> Union[datetime, None]:
        """
        Transforms a string representation to a datetime object.
        Format: '%Y:%M:%D %h%m%s.%f' (e.g. 2025-02-28 14:55:02.775121)

        :param ts_str: String representation of datetime object
        :return: Datetime object (or None)
        """

        if ts_str is None or ts_str == "": return None
        return datetime.fromisoformat(ts_str)

    @staticmethod
    def transform_embeddings_to_str_list(embeddings: np.array):
        """
        Transforms a NumPy array of embeddings into a comma-separated string.

        :param embeddings: Array of embeddings
        :returns: Comma-separated string of embeddings
        """

        if embeddings is None:
            return ""
        return ",".join(list(embeddings.astype("str")))

    @staticmethod
    def transform_str_list_to_embeddings(embeddings: str):
        """
        Transforms a comma-separated string of embeddings back into a NumPy array.

        :param embeddings: Comma-separated string of embeddings
        :returns: NumPy array of embeddings
        """

        if embeddings is None:
            return ""
        return np.array(embeddings.split(",")).astype(np.float64)