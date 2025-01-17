from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List
import numpy as np
import json
from search.embed import Embed
from search.quantize import quantize
from retrieve import FaissRetreiver, WeightedCosineRetriever
from search.rerank import BiEncoderReranker
from search.keywordsearch import KeywordSearch
from search.utils import keyword_check
from whoosh.index import open_dir
from aiocache import cached
from aiocache.serializers import JsonSerializer
import whoosh
import uvicorn
import warnings
warnings.filterwarnings('ignore')

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    #with open(r'https://github.com/jagadeeshjr5/search-engine.git\main\data\wikipedia-embeddings-binary.json', 'r') as file:
    with open(r'data\wikipedia-embeddings-binary.json', 'r') as file:
        binary_embd = json.load(file)

    binary_embd = np.array(binary_embd).astype('uint8')

    with open(r'data\wikipedia-embeddings-int8.json', 'r') as file:
        int8_embd = json.load(file)

    int8_embd = np.array(int8_embd)

    with open(r'data\wikipedia-dataset-titles.json', 'r') as file:
        titles = json.load(file)

    with open(r'data\wikipedia-dataset-summary.json', 'r') as file:
        summary = json.load(file)
     
    with open(r'data\wikipedia-dataset-description.json', 'r') as file:
        description = json.load(file)

    return binary_embd, int8_embd, titles, summary, description

binary_embd, int8_embd, titles, summary, description = load_data()

def load_retrievers():
    faiss_retreiver = FaissRetreiver(topk=50, doc_embeddings=binary_embd, precision='binary', params='auto')
    emb = Embed(model='gemini', out_dims=256, api_key='AIzaSyBpDFpC9EsOi2vijSnuY9IzM2k-ofV-mCQ')
    wei_cosine = WeightedCosineRetriever(topk=100, threshold=0.75, documents=[titles, summary, description])
    reranker = BiEncoderReranker()
    ks = KeywordSearch()
    index = ks.open_index('data\indexdir')

    return faiss_retreiver, emb, wei_cosine, reranker, ks, index

faiss_retreiver, emb, cosine, reranker, ks, index = load_retrievers()

@cached(ttl=600, serializer=JsonSerializer())
async def embed(query : str):
    result = await emb.generate_embeddings(query)
    return result['embedding']

async def search_index(index : whoosh.index.FileIndex, keywords : List, documents : List[List]):
    search_indexes = await ks.search(index, keywords)
    output = {i: [] for i in search_indexes}    
    for doc in documents:
        for i in search_indexes:
            output[i].append(doc[i])
    return output

@cached(ttl=600, serializer=JsonSerializer())
async def semantic_search(query : str, documents : List[List], binary_embeddings : np.ndarray, int8_embeddings : np.ndarray):
    
    result = await embed(query)
    result = np.array(result)
    binary_query = quantize(result, precision = 'binary', calibration=None)
    
    
    faiss_retreiver_ = FaissRetreiver(topk=500, doc_embeddings = binary_embeddings, precision='binary', params='auto')
    faiss_results = faiss_retreiver_.retreive(query_embedding=binary_query, documents=documents, doc_embeddings=binary_embeddings)
    documents = [list(group) for group in zip(*(values for values in faiss_results.values()))]
    int8_embd_retreived = np.array([int8_embeddings[idx] for idx in faiss_results.keys()])
    wei_cosine_ = WeightedCosineRetriever(topk=100, threshold=0.75, documents=documents)
    wei_cosineresults = wei_cosine_.retrieve(query_embedding=result, doc_embeddings=int8_embd_retreived, documents=documents, query=query)
    return wei_cosineresults

async def main(query : str, index : whoosh.index.FileIndex, documents : List[List], binary_embeddings : np.ndarray, int8_embeddings : np.ndarray):

    response, keywords = keyword_check(query)
    #logger.info(f"Payload: {keywords}")
    
    if response == False and keywords:  
        output = await search_index(index, keywords, documents)
        if output:  
            return output
        else:
            return f"No results found for {', '.join(keywords)}"
        
    elif response == True and keywords:
        output = await search_index(index, keywords, documents)
        logger.info(f"Payload: {output}")
        logger.info(f"Payload: {type(output)}")
        if output:
            documents = [list(group) for group in zip(*(values for values in output.values()))]
            binary_embd_retreived = np.array([binary_embeddings[idx] for idx in output.keys()])
            int8_embd_retreived = np.array([int8_embeddings[idx] for idx in output.keys()])
            output = await semantic_search(query, documents, binary_embd_retreived, int8_embd_retreived)
            return output
        else:
            return f"No results found for {', '.join(keywords)}"

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

from fastapi import HTTPException

@app.post("/search", response_model=Dict[int, Any])
async def search(query_request: str):
    try:
        logger.info(f"Payload: {type(query_request)}")
        
        # Ensure `query_request` is not altered
        output = await main(query_request, index, documents=[titles, summary, description], 
                            binary_embeddings=binary_embd, int8_embeddings=int8_embd)

        if isinstance(output, dict):
            documents = ['|$|'.join(values) for keys, values in output.items()]
            ranked_results = reranker.rerank(query_request, documents)
            output = {idx: doc[0].split('|$|') for idx, doc in enumerate(ranked_results)}
            return json.loads(output)
        else:
            return json.loads(output)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=400, detail="An error occurred while processing the request.")


#if __name__ == "__main__":
#    import uvicorn
#    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

#uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

