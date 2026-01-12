import pytest
from tools import query_sales_database, search_policy_handbook
from langchain_community.chat_models import ChatSnowflakeCortex
import os
from dotenv import load_dotenv

load_dotenv()

# Timeout after 20 seconds to prevent hanging
@pytest.mark.timeout(20) 
def test_snowflake_connection():
    """Check if we can connect to Snowflake"""
    try:
        print("Attempting to connect to Snowflake...")
        # Try a simple "Hello World" query
        query_sales_database.invoke("SELECT 1")
        print("Connection successful!")
        assert True
    except Exception as e:
        pytest.fail(f"Connection failed: {e}")

@pytest.mark.timeout(20)
def test_policy_search_tool():
    """Check if the Vector Search tool runs"""
    try:
        print("Attempting Vector Search...")
        result = search_policy_handbook.invoke("severance")
        print(f"Search Result: {result[:50]}...") 
        assert isinstance(result, str)
        assert len(result) > 0
    except Exception as e:
        pytest.fail(f"Vector search failed: {e}")