import streamlit as st
import pandas as pd
import numpy as np
import json
from embed import Embed
from quantize import quantize
from retreive import FaissRetreiver, CosineRetreiver
import time

# Load embeddings and chunks
@st.cache_resource
def load_data():
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

    #with open(r'data\wikipedia-dataset-full_text.json', 'r') as file:
    #    fulltext = json.load(file)

    return binary_embd, int8_embd, summary, titles, description

binary_embd, int8_embd, summary, titles, description = load_data()

# Cache the FaissRetreiver and Embed objects
@st.cache_resource
def load_retrievers():
    faiss_retreiver = FaissRetreiver(topk=50, doc_embeddings=binary_embd, precision='binary', params='auto')
    emb = Embed(model='gemini', out_dims=256, api_key='AIzaSyBpDFpC9EsOi2vijSnuY9IzM2k-ofV-mCQ')
    cosine = CosineRetreiver(topk=25)
    return faiss_retreiver, emb, cosine

faiss_retreiver, emb, cosine = load_retrievers()

# Streamlit app layout
st.title("Query-Based Document Retrieval")

query = st.text_input("Enter your query:")

if st.button("Search"):
    if query:
        # Generate embeddings for the query
        start_time = time.time()
        result = emb.generate_embeddings(query)
        binary_query = quantize(result, precision='binary', calibration=None)

        # Retrieve documents
        faiss_results = faiss_retreiver.retreive(query_embedding=binary_query, documents=[titles, summary, description], doc_embeddings=binary_embd)

        # Prepare int8 embeddings for cosine retrieval
        int8_embd_retreived = np.array([int8_embd[keys] for keys, _ in faiss_results.items()])
        titles_retrieved, summary_retrieved, description_retrieved = zip(
            *(values for values in faiss_results.values())
        )

        titles_retrieved = list(titles_retrieved)
        summary_retrieved = list(summary_retrieved)
        description_retrieved = list(description_retrieved)
        #fulltext_retrieved = list(fulltext_retrieved)

        # Retrieve results using cosine similarity
        cosine_results = cosine.retreive(query_embedding=result, doc_embeddings=int8_embd_retreived, documents=[titles_retrieved, description_retrieved, summary_retrieved])

        # Prepare results for display
        #output = {idx[0]: idx[1] for idx in cosine_results}

        # Convert to DataFrame
        #results_df = pd.DataFrame(cosine_results.items(), columns=['Index', 'Text'])

        # Display the titles in expanders
        st.subheader("Retrieved Documents")

        st.markdown(time.time() - start_time)
        
        for doc_id, details in cosine_results.items():
            title, description, summary = details  # Unpack the document details once for efficiency
            
            with st.expander(f"{title} - {description[:30]}...", expanded=False):
                # Use markdown efficiently to display each section of the document
                st.markdown(f"**Description:** {description}")
                st.markdown(f"**Summary:** {summary}")
                #st.markdown(f"**Full Text:** {fulltext}")

    else:
        st.warning("Please enter a query to search.")
