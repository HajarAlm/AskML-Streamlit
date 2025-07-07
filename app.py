from head_agent import Head_Agent
import streamlit as st
import os

st.title("AskML: The Smart Chatbot for Machine Learning")

# Load API keys 
openai_key = st.secrets["OPENAI_API_KEY"]
pinecone_key = st.secrets["PINECONE_API_KEY"]

with open(openai_key_file, "r") as f:
    openai_key = f.readline().strip()

with open(pinecone_key_file, "r") as f:
    pinecone_key = f.readline().strip()

pinecone_index_name = "mini-proj-2"

# Initialize chatbot
head_agent = Head_Agent(openai_key=openai_key, pinecone_key=pinecone_key, pinecone_index_name=pinecone_index_name)
head_agent.setup_sub_agents()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat messages
for message in st.session_state["messages"]:
    st.chat_message(message["role"]).markdown(message["content"])

# Capture user input
user_query = st.chat_input("Ask me anything...")
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state["messages"].append({"role": "user", "content": user_query})

    response = head_agent.get_response(user_query)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
