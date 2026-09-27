-- Read-only database role for the AutoCimKG MCP server.
-- It can read all graphs and the metadata repository. The MCP server does not rely on it alone, but also rejects
-- write clauses and rolls back every transaction (defence in depth).
-- NOTE: Apache AGE enforces this role for Cypher SET and DELETE only from version 1.7.0 on (apache/age#2309).
-- Run as the owner of the AutoCimKG database (the user AutoCimKG writes with):
--   psql -U <owner> -d <database> -v reader_password="'<password>'" -f create_readonly_role.sql

CREATE ROLE autocimkg_reader LOGIN PASSWORD :reader_password;
DO $$ BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO autocimkg_reader', current_database());
END $$;

-- AGE catalog and AutoCimKG metadata repository
GRANT USAGE ON SCHEMA ag_catalog, public TO autocimkg_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA ag_catalog, public TO autocimkg_reader;

-- existing graphs (each AGE graph is a schema of the same name)
DO $$
DECLARE graph_name name;
BEGIN
    FOR graph_name IN SELECT name FROM ag_catalog.ag_graph LOOP
        EXECUTE format('GRANT USAGE ON SCHEMA %I TO autocimkg_reader', graph_name);
        EXECUTE format('GRANT SELECT ON ALL TABLES IN SCHEMA %I TO autocimkg_reader', graph_name);
    END LOOP;
END $$;

-- graphs and tables created later by the database owner
ALTER DEFAULT PRIVILEGES GRANT USAGE ON SCHEMAS TO autocimkg_reader;
ALTER DEFAULT PRIVILEGES GRANT SELECT ON TABLES TO autocimkg_reader;

-- non-superusers may not LOAD 'age' themselves, so preload it for this role
ALTER ROLE autocimkg_reader SET session_preload_libraries = 'age';
