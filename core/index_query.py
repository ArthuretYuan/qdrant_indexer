import sys
import os
# Add the current working directory to PYTHONPATH
sys.path.append(os.getcwd())
print(sys.path)


from core.vector_db.qdrant_vector_db import QdrantVectorDB
from core.vector_db.exhaustive_search_vector_db import ExhaustiveSearchVectorDB
from core.embedder.sentence_bert_embedder import PreTrainedSentenceBertEmbedder
import json

DICTIONARY_PATH = "./config_data/dictionary_gics_leaves_separate_negative.json"

sentbert_embedder = PreTrainedSentenceBertEmbedder(flavour='paraphrase-MiniLM-L6-v2') # 'all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'
    
try:
    print('Loading dictionary...')
    with open(DICTIONARY_PATH, 'rt', encoding='utf-8') as f:
        definitions = json.load(f)
    assert isinstance(definitions, list) and all(isinstance(df, dict) and 'code' in df and 'label' in df and 'definition' in df and 'negative' in df and 'negative_refine' in df for df in definitions)
except Exception:
    print('Will default to dummy dictionary')
    definitions = [
            {
                'code': '00000000',
                'label': 'Dummy category',
                'definition': 'Dictionary could not be loaded.'
            }
        ]

vector_db_sentbert_exhaustive = ExhaustiveSearchVectorDB(definitions, sentbert_embedder)
vector_db_sentbert_qdrant = QdrantVectorDB(definitions, sentbert_embedder)


text = "Azure Health Data Services is a suite of purpose-built technologies for protected health information (PHI) in the cloud."
k=5

results_str, sorted_results_list = vector_db_sentbert_qdrant.index_search(text, k)
print(results_str)
print(sorted_results_list)

