import json
import math
import random
from sentence_transformers import SentenceTransformer
from vespa.application import Vespa
from vespa.deployment import VespaDocker


def normalize_vector(vec):
    norm = 0
    for i in range(len(vec)):
        norm += vec[i] * vec[i]
    norm = math.sqrt(norm)
    for i in range(len(vec)):
        vec[i] /= norm
    return vec


def query(field, embedding, hits, approximate, filter_range):
    body = {
        'yql': 'select * from sources dictionary_doc where ({targetHits:' + str(
            hits) + ',approximate:' + approximate + '}nearestNeighbor(' + field + ', query_embedding))',
        'ranking.features.query(query_embedding)': embedding,
        'ranking.profile': field + "_ranking",
        'ranking.matching.approximateThreshold': 0.05,
        "timeout": "10000ms",
        'hits': hits,
    }
    if filter_range is not None:  # filter_range is a random number between 0 and 10,000
        body['yql'] += f' AND filter_range < {filter_range}'

    result = app.query(body=body)
    # print(json.dumps(result.json, indent=4))
    return [child['id'] for child in result.json['root']['children']]


def recall(exact_result, ann_result):
    return len(set(exact_result).intersection(set(ann_result))) / len(exact_result)


vespa_docker = VespaDocker()
app = Vespa(url="http://localhost", port=8080)

model = SentenceTransformer('msmarco-distilbert-base-v4')

with open('words_dictionary.json', 'r') as json_file:
    words = list(json.load(json_file).keys())
random.seed(11)  # different seed from indexer
random.shuffle(words)

for word in words:
    embedding = normalize_vector(model.encode(word).tolist())
    normal_result = query('doc_embedding', embedding, 100, 'true', None)
    exact_result = query('doc_embedding', embedding, 100, 'false', None)
    float16_ann_result = query('doc_embeddingb16', embedding, 100, 'true', None)
    float16_exact_result = query('doc_embeddingb16', embedding, 100, 'false', None)
    print(
        f"Normal recall: {recall(exact_result, normal_result)}, "
        f"Float16 ANN recall: {recall(exact_result, float16_ann_result)}, " +
        f"Float16 exact recall: {recall(exact_result, float16_exact_result)}")
