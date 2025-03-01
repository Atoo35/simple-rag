import argparse
import constants
# from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large",base_url=constants.BASE_URL)
    db = Chroma(
         collection_name="example_collection",
         persist_directory=constants.CHROMA_DB_PATH, 
         embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f"Unable to find matching results.")
    #     return

    print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(constants.PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    mistral = OpenAIModel(model_name='mistral', base_url=f"{constants.BASE_URL}/v1")
    mistral_agent = Agent(mistral,retries=2)

    response = mistral_agent.run_sync(prompt)
    print()
    print(response.data)


if __name__ == "__main__":
    main()

# if __name__ == '__main__':
    # db = Chromadb("test")
    # input_text = "Hello, world!"
    # db.save_to_chroma(["hello","world",input_text,"test"])


    # embeddings = db.query("wrld")
    # print(embeddings)