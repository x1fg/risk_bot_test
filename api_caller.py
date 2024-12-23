from openai import OpenAI

class APICaller:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key, 
            base_url="https://api.proxyapi.ru/openai/v1"
        )

    def call_gpt35_turbo(self, system_prompt, user_prompt, max_tokens=200, temperature=0.5):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens, 
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Ошибка вызова API: {e}")
            return None

class OpenAIEmbeddings:

    def __init__(self, api_caller):
        self.api_caller = api_caller

    def embed_query(self, text):
        return self.api_caller.embed_text(text)

    def embed_documents(self, texts):
        return [self.api_caller.embed_text(text) for text in texts]
