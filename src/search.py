from embed import Embed
from quantize import quantize
from retreive import FaissRetreiver, CosineRetreiver
import numpy as np
import json
import warnings
warnings.filterwarnings('ignore')

with open(r'data\wikipedia-embeddings-binary.json', 'r') as file:
    binary_embd = json.load(file)

binary_embd = np.array(binary_embd).astype('uint8')

with open(r'data\wikipedia-embeddings-int8.json', 'r') as file:
    int8_embd = json.load(file)

int8_embd = np.array(int8_embd)

with open(r'data\wikipedia-dataset-summary.json', 'r') as file:
    summary = json.load(file)

with open(r'data\wikipedia-dataset-titles.json', 'r') as file:
    titles = json.load(file)

with open(r'data\wikipedia-dataset-description.json', 'r') as file:
    description = json.load(file)

with open(r'data\wikipedia-dataset-full_text.json', 'r') as file:
    fulltext = json.load(file)

#binary_embd = quantize(X, precision = 'binary', calibration=None)
#int8_embd = quantize(X, precision='int8', calibration='auto')

faiss_retreiver = FaissRetreiver(topk=50, doc_embeddings = binary_embd, precision='binary', params='auto')

emb = Embed(model='gemini', out_dims=256, api_key='AIzaSyBpDFpC9EsOi2vijSnuY9IzM2k-ofV-mCQ')

result = emb.generate_embeddings("Who is the PM of India?")

binary_query = quantize(result, precision = 'binary', calibration=None)

faiss_results = faiss_retreiver.retreive(query_embedding=binary_query, documents=[titles, summary, description, fulltext], doc_embeddings=binary_embd)

cosine = CosineRetreiver(topk=5)

int8_embd_retreived = np.array([int8_embd[keys] for keys, _ in faiss_results.items()])
titles_retrieved, summary_retrieved, description_retrieved, fulltext_retrieved = zip(
    *(values for values in faiss_results.values())
)

titles_retrieved = list(titles_retrieved)
summary_retrieved = list(summary_retrieved)
description_retrieved = list(description_retrieved)
fulltext_retrieved = list(fulltext_retrieved)


cosine_results = cosine.retreive(query_embedding=result, doc_embeddings=int8_embd_retreived, documents=[titles_retrieved, description_retrieved, summary_retrieved, fulltext_retrieved])

#output = {idx[0] : idx[1] for idx in cosine_results}
print(cosine_results)