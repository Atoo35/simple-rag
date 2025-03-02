BASE_URL="https://2ab7-34-32-130-149.ngrok-free.app"
CHROMA_DB_PATH="./chroma_langchain_db"
CHROMA_PDF_DB_PATH="./chroma_pdf_langchain_db"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""