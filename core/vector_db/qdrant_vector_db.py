from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from core.vector_db.base_vector_db import BaseVectorDB
import numpy as np


class QdrantVectorDB(BaseVectorDB):

    def __init__(self, definitions, embedder):
        print(f"Creating qdrant vector DB with embedder '{embedder.display_name}'...")
        super(QdrantVectorDB, self).__init__(definitions, embedder)
        
        # Initialize the client
        self.vector_db = QdrantClient(":memory:")  # or QdrantClient(path="path/to/db")        
        # Create a new collection
        self.vector_db.recreate_collection(
            collection_name="my_collection_pos",
            vectors_config=VectorParams(size=self.embeddings_positive[0].shape[0], distance=Distance.COSINE),
            )
        # Insert vectors into a collection
        self.vector_db.upsert(
            collection_name="my_collection_pos",
            points=[
                PointStruct(
                        id = idx,
                        vector = embed.tolist(),
                        payload = {"negative_refine": self.embeddings_negative[idx].tolist(), 
                                   "label": self.definitions[idx]["label"],
                                   "code": self.definitions[idx]["code"]}
                    )
                for idx, embed in enumerate(self.embeddings_positive)
                ]
            )
    
    def index_search(self, text, k):

        results_list = []
        vector = self.embedder.get_vector(text).tolist()
        hits_pos = self.vector_db.search(
            collection_name="my_collection_pos",
            query_vector=vector,
            limit=k 
        )
        
        for hit in hits_pos:
            idx = hit.id
            cosine_sim_pos = hit.score
    
            result_dict = {
                'Code': hit.payload['code'],
                'Label': hit.payload['label'],
                'Definition_postive': self.definitions[idx]["definition"],
                'Similarity_postive': cosine_sim_pos,}


            if self.definitions[idx]["negative"]:
                result_dict['Definition_negative'] = self.definitions[idx]["negative"]
                neg_vec = hit.payload['negative_refine']
                consine_sim_neg = self.cosine_similarity(vect_1=np.array(vector), vect_2=np.array(neg_vec))
                result_dict['Similarity_negative'] = f'neg_sim is {consine_sim_neg}'
                if cosine_sim_pos - consine_sim_neg <= 0.1:
                    result_dict['Warning'] = "Pay attention! Check the exception in the definition."
                        
            results_list.append(result_dict)
        sorted_results_list = sorted(results_list, key=lambda x: x['Similarity_postive'], reverse=True)
        
        results_str = ''
        for _, item in enumerate(sorted_results_list):
            for key, val in item.items():
                results_str += f'{key}: {val}\n'
            results_str += '\n'
        return results_str, sorted_results_list
