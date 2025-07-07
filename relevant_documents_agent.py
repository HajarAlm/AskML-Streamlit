import openai

class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = "Determine if the retrieved documents are relevant to the user's query. Respond with 'Yes' or 'No'."

    def get_relevance(self, query) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": self.prompt},
                      {"role": "user", "content": f"Query: {query}"}]
        )
        return response.choices[0].message.content.strip()
