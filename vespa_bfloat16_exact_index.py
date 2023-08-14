import json
import math
import random
from sentence_transformers import SentenceTransformer
from vespa.application import Vespa
from vespa.deployment import VespaDocker


def normalize_vector(vec):
    # normalize vector
    norm = 0
    for i in range(len(vec)):
        norm += vec[i] * vec[i]
    norm = math.sqrt(norm)
    for i in range(len(vec)):
        vec[i] /= norm
    return vec


vespa_docker = VespaDocker()
app = Vespa(url="http://localhost", port=8080)

model = SentenceTransformer('msmarco-distilbert-base-v4')

with open('words_dictionary.json', 'r') as json_file:
    words = list(json.load(json_file).keys())
random.seed(7)
random.shuffle(words)

words_to_index = 10_000
batch_size = 1_000
count = 0

for batch in [words[i:i + batch_size] for i in range(0, words_to_index, batch_size)]:
    documents = []
    for word in batch:
        embedding = normalize_vector(model.encode(word).tolist())
        documents.append(
            {
                'id': str(count),
                'fields':
                    {
                        'filter_range': random.randint(0, 10000),
                        'doc_embedding': {
                            'values': normalize_vector(list(embedding))
                        },
                        'doc_embeddingb16': {
                            'values': normalize_vector(list(embedding))
                        },
                    }
            }
        )
        count += 1
    response = app.feed_batch(schema="dictionary_doc", batch=documents)
    print(f"Indexed {count} words")
print('Done')
