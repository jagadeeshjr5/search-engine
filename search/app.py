import streamlit as st
import numpy as np
import json
from embed import Embed
from quantize import quantize
from retrieve import FaissRetreiver, CosineRetreiver
from rerank import BiEncoderReranker
from model import Model
import time
from keywordsearch import KeywordSearch
from utils import keyword_check, parse_html, Embeddings, Documents
from whoosh.index import open_dir
from aiocache import cached
from aiocache.serializers import JsonSerializer
import whoosh
from typing import List
import asyncio
import warnings
from search import main
import os
warnings.filterwarnings('ignore')


api_key = os.environ["API_KEY"] if "API_KEY" in os.environ else st.secrets["API_KEY"]

# Load embeddings and chunks
@st.cache_resource
def load_data():
    binary_embd = asyncio.run(Embeddings.load(file_path=r'data\wikipedia-dataset-embeddings-binary.npy', indices='all', extension='npy'))
    int8_embd = asyncio.run(Embeddings.load(r'data\wikipedia-dataset-embeddings-int8.npy', indices='all', extension='npy'))

    return binary_embd, int8_embd

binary_embd, int8_embd = load_data()

# Cache the FaissRetreiver and Embed objects
@st.cache_resource
def load_retrievers():
    faiss_retreiver = FaissRetreiver(topk=50, doc_embeddings=binary_embd, precision='binary', params='auto')
    emb = Embed(model='gemini', out_dims=256, api_key=api_key)
    cosine = CosineRetreiver(topk=100, threshold=0.75)
    reranker = BiEncoderReranker()
    ks = KeywordSearch()
    index = ks.open_index('data\indexdir')
    model = Model(api_key=api_key)

    return faiss_retreiver, emb, cosine, reranker, ks, index, model

faiss_retreiver, emb, cosine, reranker, ks, index, model = load_retrievers()

query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        start_time = time.time()
        file_paths = {'titles' : r'data\wikipedia-dataset-titles.h5',
                'summary' : r'data\wikipedia-dataset-summary.h5',
                'description' : r'data\wikipedia-dataset-description.h5'}
        
        asyncio.run(Documents.add(r'data/U001.h5', query))
        

        output = asyncio.run(main(query=query, index=index, binary_embeddings=binary_embd, int8_embeddings=int8_embd))


        docs = [asyncio.run(Documents.load(file_path, indices = sorted(list(output[0].values())))) for file_path in file_paths.values()]
        docs = [parse_html(doc) for doc in docs]
        docs = [doc for doc in zip(*docs)]
        rerank_docs = ['\n'.join(doc) for doc in docs]
        rerank_docs, similarities = reranker.rerank(query, rerank_docs)

        ranked_results = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
        ranked_results = [tuple(item[0]) for item in ranked_results]
   
        context = '\n\n\n'.join([doc[0] for doc in rerank_docs[:3]])

        fetched_duration = time.time() - start_time

        #answer = model.answer(query, context)
        #st.write_stream(answer)
        

        # Display the titles in expanders
        st.subheader("Retrieved Documents")

        
        total_duration = time.time() - start_time
        
        
        if st.button("Show Recommend"):
            for doc in ranked_results:
                #titles, description, summary = details  # Unpack the document details once for efficiency
                
                with st.expander(f"**{doc[1][:100]}...**", expanded=False):
                    for idx, keys in enumerate(list(file_paths.keys())):
                        st.markdown(f"**{keys.capitalize()}:** {doc[idx]}")

            st.markdown(f"**Total:** {total_duration:.2f} seconds")

    else:
        st.warning("Please enter a query to search.")
