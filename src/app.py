from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import json
import warnings

# Assuming these imports are available in your project
from embed import Embed
from quantize import quantize
from retreive import FaissRetreiver, CosineRetreiver

# Suppress warnings
warnings.filterwarnings('ignore')

# Load embeddings and chunks from files
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

# Initialize retrievers
faiss_retreiver = FaissRetreiver(topk=50, doc_embeddings=binary_embd, precision='binary', params='auto')
cosine = CosineRetreiver(topk=5)

# FastAPI application
app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/search", response_model=Dict[int, Any])
async def search(query_request: QueryRequest):
    # Generate embeddings for the input query
    emb = Embed(model='gemini', out_dims=256, api_key='AIzaSyBpDFpC9EsOi2vijSnuY9IzM2k-ofV-mCQ')
    result = emb.generate_embeddings(query_request.query)

    # Quantize the query embedding
    binary_query = quantize(result, precision='binary', calibration=None)

    # Retrieve results using FAISS
    faiss_results = faiss_retreiver.retreive(query_embedding=binary_query, documents=[titles, summary, description, fulltext], doc_embeddings=binary_embd)

    # Prepare int8 embeddings for cosine retrieval
    int8_embd_retreived = np.array([int8_embd[keys] for keys, _ in faiss_results.items()])
    titles_retrieved, summary_retrieved, description_retrieved, fulltext_retrieved = zip(
        *(values for values in faiss_results.values())
    )

    titles_retrieved = list(titles_retrieved)
    summary_retrieved = list(summary_retrieved)
    description_retrieved = list(description_retrieved)
    fulltext_retrieved = list(fulltext_retrieved)

    # Retrieve results using cosine similarity
    cosine_results = cosine.retreive(query_embedding=result, doc_embeddings=int8_embd_retreived, documents=[titles_retrieved, description_retrieved, summary_retrieved, fulltext_retrieved])

    # Prepare output
    #output = {idx[0]: idx[1] for idx in cosine_results}
    
    return cosine_results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
