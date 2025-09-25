from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client = MultiServerMCPClient(
        {
            "sqlparse": {
                "command": "python",
                "args": ["SQLParse.py"],
                "transport": "stdio",
            },
            "visualisation": {
                "command": "python",
                "args": ["visualisation.py"],
                "transport": "streamable-http",
            }
        }
    )

    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools = await client.get_tools()
    model = ChatGroq(model="mixtral-8x7b-32768")
    agent = create_react_agent(model, tools)

    # Example user input
    user_input = input("Enter your request: ")

    # Simple intent detection
    def detect_intent(text):
        text = text.lower()
        if any(q in text for q in ["sql", "query", "database"]):
            return "sqlparse"
        if any(v in text for v in ["visualise", "visualization", "plot", "graph"]):
            return "visualisation"
        return None

    intent = detect_intent(user_input)
    if intent == "sqlparse":
        response = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]})
        print("SQLParse response:", response['messages'][-1].content)
    elif intent == "visualisation":
        response = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]})
        print("Visualisation response:", response['messages'][-1].content)
    else:
        print("Intent not recognized. Please ask for SQL query or visualization.")

asyncio.run(main())
