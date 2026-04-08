from src.ingestion import load_documents
from src.embeddings import get_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

documents = load_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(documents)


class ORWrapper(Embeddings):
    def __init__(self, client):
        self.client = client

    def embed_documents(self, texts):
        return self.client.embed_documents(texts)

    def embed_query(self, text):
        return self.client.embed_query(text)


embedding = ORWrapper(get_embeddings())

vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("faiss_index")   

print("FAISS index saved")