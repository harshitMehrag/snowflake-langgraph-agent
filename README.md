# Snowflake Data Agent (LangGraph + Cortex)


### Overview
This project implements the **ReAct (Reason + Act)** pattern using **LangGraph**. It acts as an intelligent router:
* **Analytical Questions** ("What is Q3 Revenue?") $\rightarrow$ **Text-to-SQL Agent** (Queries Snowflake Tables).
* **Policy Questions** ("What is the severance rule?") $\rightarrow$ **RAG Agent** (Searches Employee Handbook).



### Tech Stack
* **Orchestration:** LangGraph (State Machines)
* **LLM:** Snowflake Cortex (`mistral-large`)
* **Database:** Snowflake (Tables + Vector Types)
* **Frontend:** Streamlit

 Run the Agent
    ```bash
    streamlit run agent_app.py
    ```
