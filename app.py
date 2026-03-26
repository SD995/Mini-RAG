from flask import Flask, request, jsonify, render_template
from src.helper import load_retriever, get_llm
from src.pipeline import run_rag

app = Flask(__name__)

# Load once (GOOD)
retriever = load_retriever()
llm = get_llm()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json["question"]

    answer = run_rag(query, retriever, llm)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run()