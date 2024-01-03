import chromadb
from tqdm import tqdm
import logging
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class ChromaInstance:

    def __init__(self) -> None:
        self.chroma_client = chromadb.Client()


    def create_index(self, collection_name:str) -> None:
        self.collection = self.chroma_client.create_collection(name=collection_name,
                                                          metadata={"hnsw:space": "cosine"})
        

    def load_index(self, collection_name:str) -> None:
        self.collection = self.chroma_client(name=collection_name)

    
    def add_to_collection(self, vectors:np.array, texts:list[str]) -> None:
        logging.info('Encoding corpus')

        uuid_list = [str(uuid.uuid4()) for _ in range(len(texts))]

        self.collection.add(
            embeddings=vectors,
            documents=texts,
            ids=uuid_list
        )


    def search(self, query_vector:np.array) -> list[str]:

        results =  self.collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=1
        )

        return results.get('documents')
    