import openai
import streamlit as st
from pinecone import Pinecone
from obnoxious_agent import Obnoxious_Agent
from query_agent import Query_Agent
from relevant_documents_agent import Relevant_Documents_Agent
from answering_agent import Answering_Agent

class Relevant_Documents_Agent:
    def __init__(self, openai_client):
        self.openai_client = openai_client

    def get_relevance(self, retrieved_docs, user_query):
        # Your logic here to assess document relevance
        # Example return value:
        return "Yes"

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name):
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name


        self.openai_client = openai.OpenAI(api_key=openai_key)

        pc = Pinecone(api_key=pinecone_key)
        self.pinecone_index = pc.Index(pinecone_index_name)

        self.sub_agents = {}

    def get_embeddings(self, text):
        """Generates embeddings using OpenAI API"""
        response = self.openai_client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def setup_sub_agents(self):
        """Initializes the chatbot's sub-agents."""
        self.sub_agents = {
            "obnoxious_agent": Obnoxious_Agent(self.openai_client),
            "relevant_documents_agent": Relevant_Documents_Agent(self.openai_client),
            "query_agent": Query_Agent(self.pinecone_index, self.openai_client, self.get_embeddings),
            "answering_agent": Answering_Agent(self.openai_client, mode="concise")
        }

    def is_greeting(self, query):
        """Check if the query is a general greeting."""
        greetings = ["hello", "hi", "hey", "good morning", "good evening", "what's up"]
        return query.lower() in greetings

    def get_response(self, user_query):
        """Handles a single user query and returns the chatbot response."""

        # Maintain conversation history across turns
        if "conversation_history" not in st.session_state:
            st.session_state["conversation_history"] = []

        # Handle greetings before checking for relevant documents
        if self.is_greeting(user_query):
            return "Hello! How can I assist you today?"

        # Check if the query is obnoxious
        if self.sub_agents["obnoxious_agent"].check_query(user_query):
            return "Please keep the conversation respectful."

        # Retrieve relevant documents
        retrieved_docs = self.sub_agents["query_agent"].query_vector_store(user_query)

        #  Debugging: Print retrieved documents
        print(f"📂 Retrieved Documents: {retrieved_docs}")

        # Check if documents were retrieved
        if not retrieved_docs:
            return "I'm sorry, but I don't have relevant information on that topic."

        # Verify if retrieved documents are relevant
        relevance = self.sub_agents["relevant_documents_agent"].get_relevance(retrieved_docs, user_query)

        if relevance == "No":
            return "I couldn't find relevant information. Could you clarify your question?"

        # Generate a response with context
        response = self.sub_agents["answering_agent"].generate_response(
            user_query, retrieved_docs, st.session_state["conversation_history"]
        )

        # Store conversation history
        st.session_state["conversation_history"].append(f"You: {user_query}")
        st.session_state["conversation_history"].append(f"Bot: {response}")

        return response
