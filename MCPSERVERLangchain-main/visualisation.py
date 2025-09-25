from mcp.server.fastmcp import FastMCP

mcp=FastMCP("Visualisation")

@mcp.tool()
async def get_weather(location:str)->str:
    """Generate Visualisation """
    return "It's always raining in California"

if __name__=="__main__":
    mcp.run(transport="streamable-http")
