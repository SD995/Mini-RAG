from src.helper import rerank_docs

def run_rag(query,retriever,llm):
    
    # 🔹 Step 1: Retrieve (k=7)
    docs = retriever.get_relevant_documents(query)
    
    # 🔹 Step 2: Rerank (🔥 cross-encoder)
    docs = rerank_docs(docs, query)
    
    # 🔹 Step 3: Take top 3
    docs = docs[:3]
    
    # 🔹 Step 4: Build context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 🔹 Step 5: Prompt
    prompt = f"""
    You are a strict QA assistant.

    Answer ONLY the given question using the context below.

    Rules:
    - Do NOT add any information not present in context
    - If information is not present in context, say: "Not found in context"
    - Do NOT guess or assume
    - Keep answer factual and short
    -Focus on pricing-related information when question is about pricing
    -Do NOT give partial answers
    
    Context:
    {context}

    Question:
    {query}


    """

    response = llm.invoke(prompt)
    answer = response.content
    
    return answer
