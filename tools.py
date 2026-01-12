import os
from dotenv import load_dotenv
from snowflake.snowpark import Session
from langchain.tools import tool

load_dotenv()

# 1. Shared Session Factory
def get_session():
    connection_parameters = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "role": os.getenv("SNOWFLAKE_ROLE"),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        "database": os.getenv("SNOWFLAKE_DATABASE"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA")
    }
    return Session.builder.configs(connection_parameters).create()


# --- TOOL 1: The Data Analyst (SQL) ---
@tool
def query_sales_database(query: str):
    """
    Useful for answering questions about sales, revenue, numbers, or data metrics.
    Input should be a fully formed SQL query executable in Snowflake.
    """
    try:
        session = get_session()
        # Safety: We assume the LLM generates valid SQL. 
        # In prod, you'd use a Read-Only user.
        result_df = session.sql(query).to_pandas()
        return result_df.to_markdown()
    except Exception as e:
        return f"Error executing SQL: {e}"

# --- TOOL 2: The HR Assistant (RAG) ---
@tool
def search_policy_handbook(question: str):
    """
    Useful for answering questions about HR policies, employee rules, or the handbook.
    Input should be a plain text question.
    """
    try:
        session = get_session()
        # Reusing the Cortex Search logic from Project 6
        search_sql = """
        SELECT chunk_text 
        FROM DOC_VECTORS 
        ORDER BY VECTOR_COSINE_SIMILARITY(
            CHUNK_VECTOR, 
            SNOWFLAKE.CORTEX.EMBED_TEXT_768('snowflake-arctic-embed-m', ?)
        ) DESC
        LIMIT 3;
        """
        df = session.sql(search_sql, params=[question]).to_pandas()
        
        if df.empty:
            return "No policy found."
            
        return "\n\n".join(df["CHUNK_TEXT"].tolist())
    except Exception as e:
        return f"Error searching handbook: {e}"