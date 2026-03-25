from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama

from sentence_transformers import CrossEncoder

# 🔹 Load embedding
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔹 Load FAISS index
vectorstore = FAISS.load_local(
    "../faiss_index",
    embedding,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 🔹 Cross encoder reranker
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_docs(docs, query):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker_model.predict(pairs)

    ranked_docs = [doc for _, doc in sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )]

    return ranked_docs

# 🔹 LLM (Ollama)
llm = Ollama(model="phi3")

# 🔹 MAIN FUNCTION
def run_rag(query):

    docs = retriever.get_relevant_documents(query)

    docs = rerank_docs(docs, query)

    docs = docs[:3]

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are a strict QA assistant.

    Answer ONLY the given question using the context below.

    Rules:
    - Do NOT include other questions or answers from the context
    - Do NOT repeat multiple Q&A pairs
    - Give a short, direct answer
    - If answer is not found, say: "Not found in context"

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    response = llm.invoke(prompt)

    return {
        "answer": response,
        "sources": docs   # return full docs (important for RAGAS)
    }