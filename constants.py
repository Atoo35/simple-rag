BASE_URL="https://ce94-35-187-156-153.ngrok-free.app"
CHROMA_DB_PATH="./chroma_langchain_db"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""