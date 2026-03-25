# Mini-RAG — Construction Marketplace Assistant

A retrieval-augmented generation (RAG) system for an internal **construction marketplace** assistant. It answers user questions from **uploaded policy, FAQ, and specification documents** instead of relying on the model’s general knowledge alone. The pipeline combines **local embeddings**, **FAISS** semantic search, **cross-encoder reranking**, and **LLM answer generation** with strict grounding instructions.

---

## What this repository delivers

| Requirement | Implementation |
|-------------|----------------|
| Chunk & embed documents | `RecursiveCharacterTextSplitter` (800 chars, 100 overlap); `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | **FAISS** index (`faiss_index/`), top-`k` retrieval (`k=7`) |
| LLM generation | **OpenRouter** — `meta-llama/llama-3.1-8b-instruct` (temperature `0`) |
| Transparency | Strict prompt + research notebooks show **retrieved chunks** and **answers**; see [Transparency](#transparency-and-explainability) |
| Chat UI | **Flask** backend + **Bootstrap** frontend (`templates/index.html`, `static/style.css`) |

Optional research work (see `research/`) compares **Ollama** (`phi3`) with OpenRouter and runs **RAGAS** metrics on a fixed question set.

---

## Why these models?

### Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

- Runs **locally** (no embedding API cost), fast to index, and widely used for semantic similarity.
- Produces dense vectors compatible with **FAISS** inner-product search after LangChain’s normalization.
- Balanced quality vs. speed for short internal documents and FAQ-style chunks.

### Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

- **Bi-encoder** retrieval (FAISS) can return plausible but suboptimal chunks; a **cross-encoder** scores each *(query, chunk)* pair jointly and reorders results.
- Documented in `research/trials.ipynb`: retrieve `k=7`, rerank, then keep the **top 3** for the prompt.

### LLM: `meta-llama/llama-3.1-8b-instruct` (OpenRouter)

- Strong instruction-following for **short, factual** answers at low temperature.
- Accessible via OpenRouter with a single API compatible with the OpenAI client (`langchain_openai.ChatOpenAI`).

### Optional local LLM (bonus / research)

- `research/trials.ipynb` compares **Ollama** `phi3` against the same retrieval stack. Local inference adds latency but avoids cloud LLM calls; OpenRouter was **faster** in recorded runs while answers differ in verbosity and formatting.

---

## How document processing works

Internal PDFs are treated as **image-based** exports (standard text loaders returned empty text in experiments). The ingestion path therefore uses:

1. **pdf2image** + **Poppler** — render each PDF page to an image.  
2. **Tesseract OCR** — `pytesseract.image_to_string` per page.  
3. **`clean_text()`** (`src/helper.py`) — normalize whitespace, strip spurious Q/A patterns from OCR, fix common OCR artifacts (e.g. `saft` → `sqft`), and remove noisy symbols.  
4. **Chunking** — `RecursiveCharacterTextSplitter` with `chunk_size=800`, `chunk_overlap=100`, separators `["\n\n", "\n", ".", " "]`.  
5. **Embeddings** — one vector per chunk; index built with **FAISS** and saved under `faiss_index/`.

This flow is exercised end-to-end in `research/trials.ipynb` and implemented for production indexing in `src/ingestion.py` + `src/build_index.py`.

---

## How retrieval is implemented

1. **Retriever**: FAISS loaded with the same embedding model; `search_kwargs={"k": 7}`.  
2. **Reranking**: `rerank_docs()` in `src/helper.py` scores the 7 candidates with the cross-encoder and sorts by score.  
3. **Context window**: the **top 3** reranked chunks are concatenated and passed to the LLM (`src/pipeline.py`).

So retrieval is **semantic (FAISS)** plus **refinement (cross-encoder)**, not raw cosine similarity in application code — though FAISS with normalized embeddings is equivalent to cosine similarity for ranking purposes.

---

## Grounding: how answers stay tied to retrieved text

`src/pipeline.py` builds a **strict QA prompt**:

- Answer **only** from the provided context.  
- **Do not** add information not in the context.  
- If the answer is not supported, respond with **`Not found in context`**.  
- No guessing; keep answers factual and concise; extra focus on pricing when the question is pricing-related.

The LLM never sees the full corpus—only the three reranked chunks—so unsupported claims are discouraged. Remaining failure modes (e.g. wrong chunk retrieved, OCR errors) are discussed in the evaluation notebooks.

---

## Transparency and explainability

| Surface | What you see |
|---------|----------------|
| **`research/trials.ipynb`** | Full **debug** trace: FAISS hits, reranked order, and final answers; optional **side-by-side** local vs. OpenRouter. |
| **`research/evaluation.ipynb`** | Loads `research/results.json`, manual rubric (relevance, hallucination, completeness, clarity), and optional **RAGAS** metrics (faithfulness, answer relevancy, context precision/recall). |
| **Web app** | The UI currently shows the **final answer** returned by `/chat`. The pipeline always **computes** retrieved chunks before generation; to show them in the browser, extend `run_rag` to return `sources` and render them in `templates/index.html`. |

---

## Research & quality analysis (bonus)

### Test questions

`research/trials.ipynb` defines **10** grounded questions (pricing, payments, protections, services, tracking). Results are saved to `research/results.json` for reuse in `evaluation.ipynb`.

### Manual observations (example rubric in `evaluation.ipynb`)

Example aggregate scores from the notebook’s labeled rubric: **relevance** high across items; **hallucination** and **completeness** vary on questions that need arithmetic or very specific phrasing (e.g. price *difference* between tiers).

### RAGAS (OpenRouter judge LLM + same embeddings)

Aggregate scores from `research/evaluation.ipynb` (`results_local` / `results_open` after `ragas.evaluate`). **Latency** is the mean of per-question **LLM generation time** (seconds) for the same 10 runs, taken from `local_latency` / `openrouter_latency` in `research/results.json` (recorded in `research/trials.ipynb` when answers were produced—not a RAGAS metric).

| Setup | Faithfulness | Answer relevancy | Context precision | Context recall | Avg. latency (s) |
|-------|--------------|------------------|-------------------|----------------|------------------|
| Local answers (Ollama `phi3`) | 0.7233 | 0.6729 | 0.9833 | 0.9226 | **9.23** |
| OpenRouter answers (`meta-llama/llama-3.1-8b-instruct`) | 0.7750 | 0.6928 | 1.0000 | 0.8750 | **3.11** |

Interpretation: **retrieval** is strong (high precision/recall); **faithfulness** and **relevancy** benefit from careful prompts and, where needed, chunk quality improvements. **OpenRouter** is much lower-latency on average in this setup; local **phi3** is slower but avoids cloud LLM calls. Per-question faithfulness, relevancy, precision, recall, and both latencies are also compared side-by-side in the notebook’s `df_compare` table.

---

## Project layout

```
Mini-RAG/
├── app.py                 # Flask app: `/` UI, POST `/chat`
├── src/
│   ├── build_index.py     # Build FAISS index from data/
│   ├── ingestion.py       # PDF → OCR → Documents
│   ├── embeddings.py      # HuggingFace embedding wrapper
│   ├── helper.py          # clean_text, FAISS load, reranker, LLM client
│   └── pipeline.py        # run_rag()
├── templates/index.html   # Chat frontend
├── static/style.css
├── data/                  # Place your PDFs here (not always committed)
├── faiss_index/           # Generated vector store (after indexing)
├── research/
│   ├── trials.ipynb       # Experiments, OCR, reranking, eval_data, results export
│   ├── evaluation.ipynb   # RAGAS + comparison tables
│   └── results.json       # Saved eval runs
├── requirements_rag_pipeline.txt
└── requirements_evaluation.txt
```

---

## Prerequisites

- **Python 3.10+** (recommended)  
- **Tesseract OCR** — [installation](https://github.com/tesseract-ocr/tesseract)  
- **Poppler** — required by `pdf2image` to read PDFs ([Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) or your OS package manager)  
- **OpenRouter API key** — for the chat app and RAGAS judge in `evaluation.ipynb`  
- Optional: **Ollama** + `phi3` for local LLM experiments in `trials.ipynb`

On **Windows**, if Tesseract is not on `PATH`, set `pytesseract.pytesseract.tesseract_cmd` (as in `trials.ipynb`). For **Poppler**, either add its `bin` folder to `PATH` or pass `poppler_path` to `convert_from_path` during development (`ingestion.py` uses the default for typical Linux/macOS deployments).

---

## Run locally

### 1. Clone and create a virtual environment

```bash
git clone <your-repo-url>
cd Mini-RAG
python -m venv .venv
```

**Windows (PowerShell):** `.\.venv\Scripts\activate`  
**macOS/Linux:** `source .venv/bin/activate`

### 2. Install dependencies

Install the RAG/runtime stack:

```bash
pip install -r requirements_rag_pipeline.txt
```

Add packages required by `app.py` and indexing (not all are listed in that file):

```bash
pip install flask langchain-openai pdf2image pytesseract
```

For notebooks and RAGAS evaluation:

```bash
pip install -r requirements_evaluation.txt
pip install datasets ragas
```

### 3. Configure environment

Create `.env` in the project root (same folder as `app.py`):

```env
OPENROUTER_API_KEY=your_key_here
```

Never commit `.env` (it is gitignored).

### 4. Add documents and build the index

Place your PDFs under `data/`, then:

```bash
python src/build_index.py
```

This creates/updates `faiss_index/`. Re-run whenever `data/` changes.

### 5. Start the web app

From the project root:

```bash
python app.py
```

Open **http://127.0.0.1:5000** in a browser. Submit a question such as:

*“what is the pricing package of Indecimal?”*

(The answer will only reflect content present in your indexed PDFs.)

### 6. Run the research notebooks

```bash
jupyter notebook
```

Open `research/trials.ipynb` or `research/evaluation.ipynb`. Ensure paths and `results.json` locations match your machine (`evaluation.ipynb` expects `../research/results.json` when run from `research/`).

---

## Deployment notes

- Host any **WSGI-capable** Python platform (e.g. **Render**, **Railway**, **Fly.io**) or a small **VPS**.  
- Set **`OPENROUTER_API_KEY`** in the provider’s secret/environment settings.  
- Install **Tesseract** and **Poppler** on the server or switch ingestion to a PDF text layer if your documents are text-native (would simplify deployment).  
- Commit or rebuild **`faiss_index/`** in CI, or run `build_index.py` at deploy time if `data/` is available in the image.  
- For production, remove debug logging that prints secrets (e.g. avoid printing API keys in `helper.py`).

---

## Example query flow

1. User submits a question in the chat UI.  
2. Flask calls `run_rag()` → FAISS returns 7 chunks → cross-encoder reranks → top 3 form the context.  
3. **Llama 3.1 8B** generates a short answer obeying the strict prompt.  
4. JSON `{"answer": "..."}` is returned to the frontend.

---

## License

See `LICENSE` in the repository root.

---

## Acknowledgements

Document content and product names in sample PDFs refer to internal **Indecimal**-style construction materials used for this assignment. Embedding, reranker, and LLM providers are subject to their respective licenses and terms of use.
