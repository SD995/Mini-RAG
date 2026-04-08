import os
from langchain.schema import Document

DATA_PATH = "processed_text"

def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            path = os.path.join(DATA_PATH, file)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file}
                )
            )

    print("Loaded docs:", len(documents))
    return documents