import openai

class Answering_Agent:
    def __init__(self, openai_client, mode="concise") -> None:
        self.client = openai_client
        self.mode = mode
        self.prompt = "Generate a response to the user's query using the relevant documents provided."

    def set_mode(self, mode):
        self.mode = mode
        if mode == "concise":
            self.prompt += " Ensure the response is precise and informative."
        else:
            self.prompt += " Provide a more detailed and engaging response."

    def generate_response(self, query, docs, conv_history, k=5):
        context = "\n".join([doc["metadata"]["text"] for doc in docs[:k]])
        conversation = "\n".join(conv_history)
        full_prompt = f"{self.prompt}\n\nContext:\n{context}\n\nConversation History:\n{conversation}\n\nUser Query: {query}\n\nResponse:"

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": full_prompt}]
        )
        return response.choices[0].message.content.strip()
