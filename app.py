from flask import Flask, request, jsonify, render_template
from src.helper import load_retriever, get_llm
from src.pipeline import run_rag
from src.helper import log_memory

app = Flask(__name__)
log_memory("Startup")
# Load once (GOOD)
log_memory("Before loading retriever")
retriever = load_retriever()
log_memory("After loading retriever")

log_memory("Before loading LLM")
llm = get_llm()
log_memory("After loading LLM")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    log_memory("Before handling chat request")
    query = request.json["question"]

    answer = run_rag(query, retriever, llm)

    log_memory("After handling chat request")
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)