import openai

class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        self.pinecone_index = pinecone_index
        self.client = openai_client
        self.prompt = "Determine if the user's query is relevant to the specific topic and retrieve relevant documents."

    def query_vector_store(self, query, k=5):
        """Queries the Pinecone vector store for relevant documents."""
        query_embedding = self.client.embeddings.create(
            input=[query],
            model="text-embedding-ada-002"
        ).data[0].embedding

        results = self.pinecone_index.query(vector=query_embedding, top_k=k, include_metadata=True)
        return results["matches"]

    def set_prompt(self, prompt):
        """Sets a custom prompt for the agent."""
        self.prompt = prompt

    def extract_action(self, response, query=None):
        """Extracts the action from the response."""
        return response.strip().lower() == "yes"
