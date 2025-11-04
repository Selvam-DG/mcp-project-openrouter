import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient
import os
import json
import logging

#  Disable noisy logs
logging.getLogger("mcp_use").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)



async def run_memory_chat():
    load_dotenv()

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        print("Error: OPENROUTER_API_KEY not found")
        return

    # Load MCP Servers from JSON (duckduckgo, airbnb, etc.)
    client = MCPClient.from_config_file("mcp_server.json")

    # Use OpenRouter instead of Groq
    llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),  
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7,
        default_headers={
            "HTTP-Referer": "http://localhost",
            "X-Title": "MCP OpenRouter Agent"
        }
    )

    # MCP Agent with memory + automatic tool usage
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=10,
        memory_enabled=True,
        system_prompt= """
        You are a real tool-using assistant.
        - Decide when to use an MCP tool (like duckduckgo_search or airbnb_search).
        - DO NOT show tool names or internal commands to the user.
        - If a tool is needed, call it using MCP, wait for the result,
          and summarize it clearly.
        - If no tool is needed, answer from your knowledge.
        - Final output must be natural text only (no <tool> or JSON unless user asks).
        """
    )

    print("\n=====  Interactive MCP Chat (OpenRouter + MCP) =====\n")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        print("\nAssistant: ", end="", flush=True)

        try:
            response = await agent.run(user_input)

            # If the LLM requested a tool → call it automatically
            if isinstance(response, dict) and "tool" in response:
                tool_name = response["tool"]
                tool_args = response.get("arguments", {})

                tool_result = await client.call_tool(tool_name, tool_args)

                # Ask LLM to format final answer from tool result
                final_answer = await llm.ainvoke([
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": user_input},
                    {"role": "tool", "content": json.dumps(tool_result)},
                    {"role": "system", "content": "Summarize this result for the user cleanly."}
                ])
                print(final_answer.content)

            else:
                print(response)

        except Exception as e:
            print("Error:", e)

    await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
