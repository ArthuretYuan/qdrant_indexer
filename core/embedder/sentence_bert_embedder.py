import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class PreTrainedSentenceBertEmbedder:

    def __init__(self, flavour):
        print(f"Loading pre-trained BERT model '{flavour}'...")
        self.model = SentenceTransformer(flavour, device='cpu')
        self.model.eval()
        self.display_name = flavour

    def get_vector(self, text):
        with torch.no_grad():
            embeddings = self.model.encode(text)
        return np.array(embeddings)
