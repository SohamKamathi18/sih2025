from __future__ import annotations

from typing import Any, Dict

import psycopg2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ExecuteSQL")


@mcp.tool()
def execute_sql(
    sql_query: str,
    dbname: str = "testdb",
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: int = 5432,
) -> Dict[str, Any]:
    """Execute a SQL statement and return rows or a status message."""

    try:
        with psycopg2.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql_query)

                if sql_query.strip().lower().startswith("select"):
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    return {"columns": columns, "rows": rows}

                conn.commit()
                return {"message": "Query executed successfully."}
    except Exception as exc:  # pragma: no cover - surfaced through MCP
        return {"error": str(exc)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
