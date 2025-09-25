mcp = FastMCP("ExecuteSQL")

from mcp.server.fastmcp import FastMCP

import psycopg2

import plotly.graph_objects as go
import io
import base64

mcp = FastMCP("ExecuteSQL")

@mcp.tool()
def execute_sql(
    sql_query: str,
    dbname: str = "testdb",
    user: str = "postgres",
    password: str = "postgres",
    host: str = "localhost",
    port: int = 5432,
    plot: bool = False
) -> str:
    """
    Executes an SQL query on the specified PostgreSQL database and returns the result or error.
    If plot=True and the query is SELECT, returns a base64-encoded PNG plot of the data using Plotly.
    Args:
        sql_query: The SQL query to execute.
        dbname, user, password, host, port: PostgreSQL connection parameters.
        plot: Whether to plot the result of a SELECT query.
    Returns:
        Query result as a string, or base64 PNG image if plot=True.
    """
    try:
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        if sql_query.strip().lower().startswith("select"):
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            if plot and result:
                x = [row[0] for row in result]
                y = [row[1] for row in result] if len(result[0]) > 1 else None
                fig = go.Figure()
                if y:
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Data'))
                    fig.update_layout(xaxis_title=columns[0], yaxis_title=columns[1])
                else:
                    fig.add_trace(go.Scatter(x=list(range(len(x))), y=x, mode='lines+markers', name=columns[0]))
                    fig.update_layout(xaxis_title='Index', yaxis_title=columns[0])
                fig.update_layout(title='Query Result Plot')
                buf = io.BytesIO()
                fig.write_image(buf, format='png')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                return img_base64
            return str(result)
        else:
            conn.commit()
            conn.close()
            return "Query executed successfully."
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
