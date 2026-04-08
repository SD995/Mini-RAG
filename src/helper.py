from http import client
import re
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from src.embeddings import get_embeddings, ORWrapper
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import psutil
from src.build_index import build_index
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")# Load environment variables from .env file
print("API key:",os.getenv("OPENROUTER_API_KEY"))  # Debug print to verify API key is loaded
def clean_text(text):
    # 1. Normalize whitespace
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # 🔥 2. REMOVE Q&A PATTERNS (CRITICAL FIX)
    text = re.sub(r"Q:.*?A:", "", text, flags=re.IGNORECASE)

    # 🔥 3. Remove leftover Q: or A:
    text = re.sub(r"\bQ:\s*", "", text)
    text = re.sub(r"\bA:\s*", "", text)

    # 4. Fix OCR mistakes
    replacements = {
        "saft": "sqft",
        "saqft": "sqft",
        "sq ft": "sqft",
        "rs.": "Rs",
        "₹": "Rs ",
        "=": "",
        "%": "",
        "—": "-",
        "–": "-",
    }

    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    # 🔥 5. Remove weird currency/symbol noise like ¥, §
    text = re.sub(r"[¥§]", "", text)

    # 🔥 6. Keep useful characters, remove junk
    text = re.sub(r"[^\w\s\.,:/\-\(\)]", "", text)

    # 🔥 7. Fix spacing around punctuation
    text = re.sub(r"\s+([.,])", r"\1", text)

    # 🔥 8. Remove repeated dashes or dots
    text = re.sub(r"-{2,}", "-", text)
    text = re.sub(r"\.{2,}", ".", text)

    return text.strip()


def log_memory(stage):
    process =psutil.Process(os.getpid())
    process_mem = process.memory_info().rss/(1024**2)  # Memory in MB
    print(f"{stage} | App: {process_mem:.2f} MB",flush=True)

def embed_text(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # or OpenRouter-supported model
        input=texts
    )
    return [item.embedding for item in response.data]


def load_retriever():
    try:
        print(" Loading retriever...")

        # Initialize embeddings
        embedding = ORWrapper(get_embeddings())

        # Build FAISS if not exists
        if not os.path.exists("faiss_index"):
            print("FAISS index not found. Building...")
            build_index()

        # Load FAISS index
        db = FAISS.load_local(
            "faiss_index",
            embedding,
            allow_dangerous_deserialization=True
        )

        print(" FAISS loaded successfully")

        return db.as_retriever(search_kwargs={"k": 7})

    except Exception as e:
        print(" Error loading FAISS:", e)
        raise
    



def get_llm():
    return ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    model="meta-llama/llama-3.1-8b-instruct",
    temperature=0
)

reranker_model = None

def get_reranker():
    global reranker_model
    if reranker_model is None:
        log_memory("before loading reranker")
        reranker_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        log_memory("after loading reranker")
    return reranker_model



def rerank_docs(docs, query):
    model = get_reranker()

    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)

    ranked_docs = [
        doc for _, doc in sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )
    ]

    return ranked_docs