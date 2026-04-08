from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

class OpenRouterEmbeddings:

    def embed_documents(self, texts):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

class ORWrapper(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        return self.client.embed_documents(texts)

    def embed_query(self, text):
        return self.client.embed_query(text)
    

def get_embeddings():
    return OpenRouterEmbeddings()