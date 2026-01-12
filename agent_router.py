import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatSnowflakeCortex
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator

# Import your tools
from tools import query_sales_database, search_policy_handbook

load_dotenv()

# 1. Setup LLM
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

# 2. Define State
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage], operator.add]
    context: str # We add a field to store the tool output

# 3. NODE 1: The Router (The Decision Maker)
def router_node(state: AgentState):
    user_query = state["messages"][-1].content
    
    # We force the model to categorize the query
    router_prompt = f"""
    You are a classifier. Determine which tool is needed to answer the user's question.
    
    - If the user asks about Policies, Rules, HR, or the Handbook -> reply "SEARCH"
    - If the user asks about Sales, Revenue, Data, or Numbers -> reply "SQL"
    - Otherwise -> reply "CHAT"
    
    User Question: {user_query}
    
    Reply ONLY with the keyword.
    """
    
    decision = llm.invoke(router_prompt).content.strip().upper()
    print(f"🚦 Router Decision: {decision}")
    
    # We pass the decision to the next step via a special "routing" message (not shown to user)
    return {"messages": [HumanMessage(content=decision)]}

# 4. NODE 2: The Tool Executors (UPDATED)
def run_tool_node(state: AgentState):
    decision = state["messages"][-1].content
    user_query = state["messages"][-2].content # Get the original question
    
    tool_output = ""
    
    if "SEARCH" in decision:
        print("Running Vector Search...")
        tool_output = search_policy_handbook.invoke(user_query)
        
    elif "SQL" in decision:
        print("Running SQL Generator...")
        
        # --- THE FIX: Provide the Schema Context ---
        schema_context = """
        You have access to a Snowflake table named: PORTFOLIO_DB.ANALYTICS.STG_DAILY_REVENUE
        Columns:
        - DATE (Date)
        - REGION (String) - e.g., 'North', 'South'
        - TOTAL_REVENUE (Number)
        - TRANSACTION_COUNT (Number)
        
        IMPORTANT: Always use the full table name 'PORTFOLIO_DB.ANALYTICS.STG_DAILY_REVENUE' in your FROM clause.
        """
        
        # We instruct the LLM to use the specific table and columns
        sql_prompt = f"""
        {schema_context}
        
        Generate a valid Snowflake SQL query for: {user_query}. 
        Return ONLY the SQL string. Do not add markdown or explanations.
        """
        
        generated_sql = llm.invoke(sql_prompt).content.replace("```sql", "").replace("```", "").strip()
        print(f"Executing SQL: {generated_sql}")
        
        # Execute the generated SQL
        tool_output = query_sales_database.invoke(generated_sql)
        print(f"Tool Output: {tool_output}") 
        
    else:
        tool_output = "No tool needed."

    return {"context": tool_output}

# 5. NODE 3: The Final Answer
def answer_node(state: AgentState):
    user_query = state["messages"][-2].content
    context = state["context"]
    
    final_prompt = f"""
    Answer the user question using the Context provided below.
    If the context contains the answer, cite it. 
    If the context is empty or irrelevant, say "I don't know."
    
    Context:
    {context}
    
    User Question:
    {user_query}
    """
    
    response = llm.invoke(final_prompt)
    return {"messages": [response]}

# 6. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("tools", run_tool_node)
workflow.add_node("final", answer_node)

workflow.set_entry_point("router")
workflow.add_edge("router", "tools")
workflow.add_edge("tools", "final")
workflow.add_edge("final", END)

app = workflow.compile()

# --- TEST ---
if __name__ == "__main__":
    print("🤖 Waking up the Router Agent...")
    
    # Test: Severity Policy
    query = "What is the total revenue by region?"
    print(f"\nUser: {query}")
    inputs = {"messages": [HumanMessage(content=query)]}
    
    result = app.invoke(inputs)
    print(f"\n[FINAL ANSWER]: {result['messages'][-1].content}")