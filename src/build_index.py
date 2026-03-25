from src.ingestion import load_documents
from src.embeddings import get_embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

documents = load_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "]
)

chunks = text_splitter.split_documents(documents)

embedding = get_embeddings()


vectorstore = FAISS.from_documents(chunks, embedding)
vectorstore.save_local("faiss_index")   

print("FAISS index saved")