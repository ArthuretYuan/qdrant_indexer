from core.vector_db.base_vector_db import BaseVectorDB

class ExhaustiveSearchVectorDB(BaseVectorDB):

    def __init__(self, definitions, embedder):
        print(f"Creating exhaustive-search vector DB with embedder '{embedder.display_name}'...")
        super(ExhaustiveSearchVectorDB, self).__init__(definitions, embedder)


    def knn_search(self, text, k):

        query_vector = self.embedder.get_vector(text)
        similarities = []
        for i, vect in enumerate(self.embeddings_positive):
            cosine_sim_pos = self.cosine_similarity(query_vector, vect)
            similarities.append((i, cosine_sim_pos))
        similarities.sort(key=lambda x: x[1], reverse=True)

        results_list = []
        for (idx, cosine_sim_pos) in similarities[:k]:
            result_dict = {
                'Code': self.definitions[idx]["code"],
                'Label': self.definitions[idx]["label"],
                'Definition_positive': self.definitions[idx]["definition"],
                'Similarity_positive':cosine_sim_pos,}  
            if self.definitions[idx]["negative"]:
                result_dict['Definition_negative'] = self.definitions[idx]["negative"]
                neg_vec = self.embeddings_negative[idx]
                consine_sim_neg = self.cosine_similarity(vect_1=query_vector, vect_2=neg_vec)
                result_dict['Similarity_negative'] = f'neg_sim is {consine_sim_neg}'
                if cosine_sim_pos - consine_sim_neg <= 0.1:
                    result_dict['Warning'] = "Pay attention! Check the exception in the definition."
            results_list.append(result_dict)
            
        results_str = ''
        for _, item in enumerate(results_list):
            for key, val in item.items():
                results_str += f'{key}: {val}\n'
            results_str += '\n'
        return results_str, results_list
