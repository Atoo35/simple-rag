import chromadb
from generate_embeddings import generate_embed

class Chromadb(object):
    def __init__(self, collection_name):
        self.client = chromadb.Client()
        self.collection=self.client.get_or_create_collection(name=collection_name)

    def save_to_chroma(self,chunks: list[str]):
        # collection has a .add method which takes a list of documents, a list of embeddings and list of ids
        for doc in chunks:
            input_text = doc  # assuming doc is a string containing the text to be saved in ChromaDB
            embed = generate_embed(input_text)
            # print(embed[0]["embedding"])
            count = self.collection.count()
            self.collection.add(embeddings=embed[0]["embedding"],metadatas=[{"text":input_text}],ids=[f"{count}"])

    def get_by_id(self, id):
        return self.collection.get(ids=[id],include=["metadatas","embeddings"])
    
    def query(self, queryStr):
        embed = generate_embed(queryStr)
        return self.collection.query(query_embeddings=embed[0]["embedding"],
                                     include=["metadatas","distances"],
                                     n_results=3)

