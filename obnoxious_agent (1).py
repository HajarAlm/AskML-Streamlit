class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.prompt = "Determine if the following query is obnoxious. Respond with 'Yes' or 'No'."

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        return response.strip().lower() == "yes"

    def check_query(self, query):
        """Checks if the query is obnoxious."""
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"{self.prompt}\nQuery: {query}"}]
        )
        return self.extract_action(response.choices[0].message.content)
