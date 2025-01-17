from embed import Embed
from quantize import quantize
from retrieve import FaissRetreiver, CosineRetreiver
import numpy as np
from rerank import BiEncoderReranker
from keywordsearch import KeywordSearch
import warnings
from utils import keyword_check, parse_html, Embeddings, Documents
from whoosh.index import open_dir
from aiocache import cached
from aiocache.serializers import JsonSerializer
import whoosh
from typing import List
import asyncio
import time
import os
import streamlit as st
warnings.filterwarnings('ignore')

api_key = os.environ["API_KEY"] if "API_KEY" in os.environ else st.secrets["API_KEY"]

def load_data():
    binary_embd = asyncio.run(Embeddings.load(file_path=r'data\wikipedia-dataset-embeddings-binary.npy', indices='all', extension='npy'))
    int8_embd = asyncio.run(Embeddings.load(r'data\wikipedia-dataset-embeddings-int8.npy', indices='all', extension='npy'))

    return binary_embd, int8_embd

binary_embd, int8_embd = load_data()

def load_retrievers():
    faiss_retreiver = FaissRetreiver(topk=50, doc_embeddings=binary_embd, precision='binary', params='auto')
    emb = Embed(model='gemini', out_dims=256, api_key=api_key)
    cosine = CosineRetreiver(topk=100, threshold=0.75)
    reranker = BiEncoderReranker()
    ks = KeywordSearch()
    index = ks.open_index('data\indexdir')

    return faiss_retreiver, emb, cosine, reranker, ks, index

faiss_retreiver, emb, cosine, reranker, ks, index = load_retrievers()

@cached(ttl=600, serializer=JsonSerializer())
async def embed(query : str):
    result = await emb.generate_embeddings(query)
    return result['embedding']

async def search_index(index : whoosh.index.FileIndex, keywords : List):
    search_indexes = await ks.search(index, keywords)
    return search_indexes

@cached(ttl=600, serializer=JsonSerializer())
async def semantic_search(query : str, document_indices : List, binary_embeddings : np.ndarray, int8_embeddings : np.ndarray):
    
    result = await embed(query)
    result = np.array(result)
    start_time = time.time()
    binary_query = quantize(result, precision = 'binary', calibration=None)
    
    
    faiss_retreiver_ = FaissRetreiver(topk=500, doc_embeddings = binary_embeddings, precision='binary', params='auto')
    faiss_results = faiss_retreiver_.retrieve(query_embedding=binary_query, document_indices=document_indices, doc_embeddings=binary_embeddings)
    #documents = [list(group) for group in zip(*(values for values in faiss_results.values()))]
    int8_embd_retreived = np.array([int8_embeddings[idx] for idx in faiss_results.keys()])
    cosine = CosineRetreiver(topk=10, threshold=0.75)
    cosineresults = cosine.retrieve(query_embedding=result, doc_embeddings=int8_embd_retreived, document_indices=list(faiss_results.values()))
    end_time = time.time()
    return cosineresults, end_time - start_time

async def main(query : str, index : whoosh.index.FileIndex, binary_embeddings : np.ndarray, int8_embeddings : np.ndarray):

    response, keywords = keyword_check(query)
    
    if response == False and keywords:

        start_time = time.time()

        output = await search_index(index, keywords)

        end_time = time.time()

        return dict(list(output.items())[:25]), end_time - start_time
    
    elif response == True and keywords:

        document_indices = await search_index(index, keywords)
        #documents = [list(group) for group in zip(*(values for values in output.values()))]
        binary_embd_retreived = np.array([binary_embeddings[idx] for idx in document_indices])
        int8_embd_retreived = np.array([int8_embeddings[idx] for idx in document_indices])
        output, search_time = await semantic_search(query, document_indices, binary_embd_retreived, int8_embd_retreived)

        return output, search_time
    
    elif response == False and keywords == []:

        result = await embed(query)
        result = np.array(result)
        start_time = time.time()
        binary_query = quantize(result, precision = 'binary', calibration=None)
        
        faiss_results = faiss_retreiver.retrieve(query_embedding=binary_query, document_indices=list(range(len(binary_embeddings))), doc_embeddings=binary_embeddings)
        int8_embd_retreived = np.array([int8_embeddings[idx] for idx in faiss_results.keys()])
        cosine = CosineRetreiver(topk=10, threshold=0.75)
        cosineresults = cosine.retrieve(query_embedding=result, doc_embeddings=int8_embd_retreived, document_indices=list(faiss_results.values()))
        end_time = time.time()

        return cosineresults, end_time - start_time
    
if __name__ == "__main__":

    query = "Who is the PM of India?"

    file_paths = {'titles' : r'data\wikipedia-dataset-titles.h5',
              'summary' : r'data\wikipedia-dataset-summary.h5',
              'description' : r'data\wikipedia-dataset-description.h5'}
    

    output = asyncio.run(main(query=query, index=index, binary_embeddings=binary_embd, int8_embeddings=int8_embd))


    docs = [asyncio.run(Documents.load(file_path, indices = sorted(list(output[0].values())))) for file_path in file_paths.values()]
    docs = [parse_html(doc) for doc in docs]
    docs = [doc for doc in zip(*docs)]
    rerank_docs = ['\n'.join(doc) for doc in docs]
    rerank_docs = reranker.rerank(query, rerank_docs)
    print(len(rerank_docs))