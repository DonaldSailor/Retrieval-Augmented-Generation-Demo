from sentence_transformers import SentenceTransformer
import numpy as np

class Encoder:
    def __init__(self, model) -> None:
        self.model = SentenceTransformer(model)

    def encode(self, corpus:list[str], bsize=64) -> np.array:        
        encoded_data = self.model.encode(corpus, batch_size=bsize, show_progress_bar=True)
        return encoded_data