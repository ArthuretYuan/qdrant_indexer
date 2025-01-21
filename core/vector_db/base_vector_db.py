import numpy as np

class BaseVectorDB:

    def __init__(self, definitions, embedder):
        self.definitions = definitions
        self.embedder = embedder
        self.num_cls = len(self.definitions)
        self.embeddings_positive = [self.embedder.get_vector(definition['definition']) for definition in self.definitions]
        self.embeddings_negative = [self.embedder.get_vector(definition['negative_refine']) for definition in self.definitions]
        
    def cosine_similarity(self, vect_1, vect_2, normalized=False):
        if normalized:
            return max(0, min(1, np.dot(vect_1, vect_2)))
        else:
            return max(0, min(1, np.dot(vect_1, vect_2) / (np.linalg.norm(vect_1) * np.linalg.norm(vect_2))))