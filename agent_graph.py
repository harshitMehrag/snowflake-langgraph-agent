import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatSnowflakeCortex
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# Import our custom tools
from tools import query_sales_database, search_policy_handbook

load_dotenv()

# 1. Initialize the LLM
# We use 'mistral-large' or 'llama3-70b' as they follow instructions best
llm = ChatSnowflakeCortex(
    model="mistral-large",
    snowflake_account=os.getenv("SNOWFLAKE_ACCOUNT"),
    snowflake_username=os.getenv("SNOWFLAKE_USER"),
    snowflake_password=os.getenv("SNOWFLAKE_PASSWORD"),
    snowflake_role=os.getenv("SNOWFLAKE_ROLE"),
    snowflake_warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    snowflake_database=os.getenv("SNOWFLAKE_DATABASE"),
    snowflake_schema=os.getenv("SNOWFLAKE_SCHEMA"),
    temperature=0
)

# 2. Define Tools List
tools = [query_sales_database, search_policy_handbook]

# 3. Create the ReAct Agent
# This prebuilt function automatically adds a system prompt that teaches the model:
# "You have access to these tools. To use one, output a specific format..."
graph = create_react_agent(llm, tools)

# --- TEST FUNCTION ---
if __name__ == "__main__":
    print("🤖 Waking up the ReAct Agent...")
    
    # Test 1: Ask a Policy Question
    query = "What is the severance policy? Be specific based on the handbook."
    print(f"\nUser: {query}")
    print("-" * 40)
    
    inputs = {"messages": [HumanMessage(content=query)]}
    
    # We stream the output so we can see the "Thinking" process
    for event in graph.stream(inputs, stream_mode="values"):
        message = event["messages"][-1]
        if hasattr(message, "content") and message.content:
            print(f"[{message.type.upper()}]: {message.content}\n")