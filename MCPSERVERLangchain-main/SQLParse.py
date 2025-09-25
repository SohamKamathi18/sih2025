from mcp.server.fastmcp import FastMCP

mcp=FastMCP("SQLParse")

@mcp.tool()
def add(a:int,b:int)->int:
    """_summary_
    Add to numbers
    """
    return a+b

@mcp.tool()
def multiple(a:int,b:int)-> int:
    """Multiply two numbers"""
    return a*b

#The transport="stdio" argument tells the server to:

#Use standard input/output (stdin and stdout) to receive and respond to tool function calls.

if __name__=="__main__":
    mcp.run(transport="stdio")


# MCP tool: Convert natural language to SQL query
try:
    import openai
except ImportError:
    openai = None

@mcp.tool()
def natural_to_sql(nl_query: str, table_schema: str = "") -> str:
    """
    Converts a natural language query to an SQL query using Groq API.
    Args:
        nl_query: The natural language question/request.
        table_schema: (Optional) The schema of the table(s) involved.
    Returns:
        SQL query as a string.
    """
    if openai is None:
        return "Error: openai package not installed. Please install openai."
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY environment variable not set."
    openai.api_key = api_key
    openai.base_url = "https://api.groq.com/openai/v1"
    prompt = f"Convert this natural language query to SQL.\nSchema: {table_schema}\nQuery: {nl_query}\nSQL: "
    try:
        response = openai.Completion.create(
            model="mixtral-8x7b-32768",
            prompt=prompt,
            max_tokens=150,
            temperature=0
        )
        sql = response.choices[0].text.strip()
        return sql
    except Exception as e:
        return f"Error: {str(e)}"