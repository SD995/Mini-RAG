import os
from pdf2image import convert_from_path
import pytesseract
from langchain.schema import Document
from src.helper import clean_text

DATA_PATH = "data"

def load_documents():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):

            pdf_path = os.path.join(DATA_PATH, file)
            print(f"Processing: {file}")

            # ⚠️ REMOVE poppler_path for deployment (Linux handles it)
            images = convert_from_path(pdf_path)

            for i, image in enumerate(images):

                text = pytesseract.image_to_string(image)

                # ✅ USE YOUR CLEAN FUNCTION
                text = clean_text(text)

                if text:
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file,
                                "page": i
                            }
                        )
                    )

    print("Total documents:", len(documents))
    return documents