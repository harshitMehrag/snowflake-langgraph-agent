import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent_router import app  # Import the brain we just built!

st.set_page_config(page_title="Snowflake Data Agent", layout="wide")

st.title("🤖 Snowflake Data Agent")
st.caption("I can answer questions about your **Data** (SQL) or your **Documents** (RAG).")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Input Field
if query := st.chat_input("Ask: 'What is Q3 Revenue?' or 'What is the severance policy?'"):
    # 1. Display User Message
    st.chat_message("user").markdown(query)
    st.session_state.messages.append(HumanMessage(content=query))

    # 2. Process with Agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking (Routing to SQL or Vector Search)..."):
            try:
                # Prepare input for the graph
                inputs = {"messages": st.session_state.messages}
                
                # Run the Agent!
                result = app.invoke(inputs)
                
                # Extract the final response
                final_response = result['messages'][-1].content
                
                st.markdown(final_response)
                
                # Add to history
                st.session_state.messages.append(AIMessage(content=final_response))
                
            except Exception as e:
                st.error(f"An error occurred: {e}")