import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from utils import TFIDF
import warnings
warnings.filterwarnings('ignore')

class Retreiver(ABC):
    """
    Abstract base class for document retrieval models.

    Parameters:
    - topk (int): The number of top documents to retrieve based on similarity to the query.
    """

    @abstractmethod
    def __init__(self, 
                 topk: int
                 ):
        """
        Initializes the retriever with the topk value.
        
        Parameters:
        - topk (int): The number of top documents to retrieve.
        """
        self.topk = topk

    @abstractmethod
    def retrieve(self, 
                 query_embedding: np.ndarray, 
                 doc_embeddings: np.ndarray, 
                 documents: Union[List[str], List[List[str]]]
                 ):
        """
        Retrieves the topk documents based on similarity to the query embedding.

        Parameters:
        - query_embedding (np.ndarray): The query vector.
        - doc_embeddings (np.ndarray): The matrix of document embeddings.
        - documents (List): List of document texts or identifiers.

        Returns:
        - List of tuples containing document indices and their corresponding documents.
        """
        pass


class KNNRetreiver(Retreiver):
    """
    K-Nearest Neighbors-based document retriever.
    
    Parameters:
    - topk (int): Number of nearest documents to retrieve.
    - metric (str): Distance metric used by KNN (default is 'hamming').
    """

    def __init__(self, 
                 doc_embeddings : np.ndarray, 
                 topk: int = 50, 
                 metric: str = 'hamming'
                 ):
        """
        Initializes the KNN retriever with the specified topk and distance metric.
        """
        self.knn = NearestNeighbors(n_neighbors=topk, metric=metric)
        self.knn.fit(doc_embeddings)

    def retrieve(self, 
                 query_embedding: np.ndarray, 
                 doc_embeddings: np.ndarray, 
                 document_indices: Union[List[str], List[List[str]]]
                 ) -> dict:
        """
        Retrieves documents using K-Nearest Neighbors.

        Parameters:
        - query_embedding (np.ndarray): The query vector.
        - doc_embeddings (np.ndarray): Matrix of document embeddings.
        - documents (List): List of documents.

        Returns:
        - List of tuples with document index and document text for each top-k result.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
    
        
        _, indices = self.knn.kneighbors(query_embedding)
        knn_results = [(index, rank) for rank, index in enumerate(indices[0])]
        results = {idx[0] : document_indices[idx[0]] for idx in knn_results}
    
        return results


class CosineRetreiver(Retreiver):
    """
    Cosine similarity-based document retriever.

    Parameters:
    - topk (int): Number of top similar documents to retrieve.
    """

    def __init__(self, 
                 topk: int = 50, 
                 threshold : float = 0.80
                 ):
        """
        Initializes the cosine similarity retriever with the topk parameter.
        """
        self.topk = topk
        self.threshold = threshold

    def retrieve(self, 
                 query_embedding: np.ndarray, 
                 doc_embeddings: np.ndarray, 
                 document_indices: List
                 ) -> dict:
        """
        Retrieves documents based on cosine similarity with the query embedding.

        Parameters:
        - query_embedding (np.ndarray): The query vector.
        - doc_embeddings (np.ndarray): Matrix of document embeddings.
        - documents (List): List of documents.

        Returns:
        - List of tuples with document index and document text for each top-k result.
        """
        similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()
        top_k_indices = similarities.argsort()[::-1][:self.topk]
        cosine_results = [
            (index, rank) for rank, index in enumerate(top_k_indices) 
            if similarities[index] >= self.threshold
        ]
        if not cosine_results:
            return "No relevant documents found."
        results = {idx[0] : document_indices[idx[0]] for idx in cosine_results}
        
        return results


class FaissRetreiver(Retreiver):
    """
    FAISS-based document retriever for large-scale and efficient retrieval.

    Parameters:
    - doc_embeddings (np.ndarray): The document embedding matrix to be indexed.
    - topk (int): Number of top similar documents to retrieve.
    - precision (str): Precision level of the index ('binary' or 'int8').
    - params (Union[str, Tuple[int, int]]): Cluster parameters for FAISS index.
      Default is 'auto', which calculates suitable parameters.
    """

    def __init__(self, 
                 doc_embeddings: np.ndarray, 
                 topk: int = 50, 
                 precision: str = 'binary', 
                 params: Union[str, Tuple[int, int]] = 'auto'
                 ):
        """
        Initializes the FAISS retriever with indexing parameters.

        Raises:
        - ValueError: If the embedding data type does not match the precision requirement.
        - ValueError: If the number of training points is insufficient for clustering.

        Parameters:
        - doc_embeddings (np.ndarray): The document embeddings for building the FAISS index.
        - topk (int): Number of top results to retrieve.
        - precision (str): Type of quantization ('binary' or 'int8').
        - params (Union[str, Tuple[int, int]]): Clustering parameters or 'auto' to calculate.
        """
        self.topk = topk
        embedding_dimension = doc_embeddings.shape[1] if precision == 'int8' else doc_embeddings.shape[1] * 8
        params = (int(2 * math.sqrt(len(doc_embeddings))), int(2 * math.sqrt(len(doc_embeddings)) * 0.05)) if params == 'auto' else params

        if doc_embeddings.shape[0] < params[0]:
            raise ValueError(f'Not enough training points: {doc_embeddings.shape[0]} provided, but {params[0]} required. Reduce the value of params')

        if precision == 'binary':
            if not doc_embeddings.dtype == 'uint8':
                raise ValueError('The embeddings data type must be uint8')
            quantizer = faiss.IndexBinaryFlat(embedding_dimension)
            self.ivf_index = faiss.IndexBinaryIVF(quantizer, embedding_dimension, params[0])
            self.ivf_index.train(doc_embeddings)
            self.ivf_index.add(doc_embeddings)
            self.ivf_index.nprobe = params[1]

        elif precision == 'int8':
            if not doc_embeddings.dtype == 'int8':
                raise ValueError('The embeddings data type must be int8')
            quantizer = faiss.IndexFlatL2(embedding_dimension)
            self.ivf_index = faiss.IndexIVFFlat(quantizer, embedding_dimension, params[0])
            self.ivf_index.train(doc_embeddings)
            self.ivf_index.add(doc_embeddings)

    def retrieve(self, 
                 query_embedding: np.ndarray, 
                 document_indices: Union[List[str], List[List[str]]], 
                 doc_embeddings: np.ndarray = None
                 ):
        """
        Retrieves documents using FAISS for approximate nearest neighbor search.

        Parameters:
        - query_embedding (np.ndarray): The query vector.
        - documents (List): List of documents.
        - doc_embeddings (np.ndarray, optional): Matrix of document embeddings for training.

        Returns:
        - List of tuples with document index and document text for each top-k result.
        """
        _, indices = self.ivf_index.search(query_embedding, self.topk)
        results = {idx : document_indices[idx] for idx in indices[0]}
        return results


class WeightedCosineRetriever:
    """
    A retriever that combines cosine similarity with a weighted term frequency factor
    to improve relevance in document search.

    Attributes:
    -----------
    topk : int
        The maximum number of top documents to retrieve.
    threshold : float
        The minimum similarity score for a document to be considered relevant.
    alpha : float
        The weight of the term frequency factor in the final score.
    vectorizer : TfidfVectorizer
        The vectorizer to compute TF-IDF weights for terms.
    tfidf_matrix : np.ndarray
        The matrix representing TF-IDF weights of terms in each document.
    terms : np.ndarray
        Array of terms extracted by the vectorizer from the document corpus.
    """

    def __init__(self, 
                 tfidf_vectorizer_path : str, 
                 tfidf_matrix_path : str
                 ):
        """
        Initialize the WeightedCosineRetriever with documents and parameters.

        Parameters:
        -----------
        documents : List[str]
            A list of documents to fit the TF-IDF vectorizer.
        topk : int, optional
            Number of top results to retrieve, by default 50.
        threshold : float, optional
            Minimum score threshold to filter relevant documents, by default 0.80.
        alpha : float, optional
            Weight for the term frequency adjustment factor, by default 0.7.
        """
        self.vectorizer = TFIDF(tfidf_vectorizer_path)
        self.tfidf_matrix = self.vectorizer.load_matrix(tfidf_matrix_path)

    def retrieve(self, 
                 query_embedding : np.ndarray, 
                 doc_embeddings : np.ndarray, 
                 document_indices : List, 
                 query : str, 
                 topk : int = 50, 
                 threshold : float = 0.80, 
                 alpha : float = 0.7
                 ):
        """
        Retrieve top-k documents similar to the query by combining cosine similarity
        and TF-IDF weighting.

        Parameters:
        -----------
        query_embedding : np.ndarray
            The embedding vector of the query.
        doc_embeddings : np.ndarray
            The embedding matrix for all documents.
        documents : Union[List[str], List[List[str]]]
            The list of documents or a nested list if documents contain multiple parts.
        query : str
            The user's search query text.

        Returns:
        --------
        results : dict
            A dictionary of relevant document indices and their content, sorted by relevance.
        """
        
        cosine_similarities = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings).flatten()

        #print('cosine_similarities: ', cosine_similarities)

        query = ' '.join([word for word in query.split() if word.lower() not in stop_words])
        
        query_tfidf = self.vectorizer.transform([query.lower()])
        query_indices = query_tfidf.nonzero()[1]
        
        tfidf_factor = np.array(self.tfidf_matrix[document_indices][:, query_indices].sum(axis=1)).flatten()
        #tfidf_factor += 1e-8
        #print('tfidf_factor: ', tfidf_factor)
        
        final_scores = cosine_similarities + alpha * tfidf_factor
        #print('final_scores: ', final_scores)
        
        top_k_indices = final_scores.argsort()[::-1][:topk]
        cosine_results = [
            (index, rank) for rank, index in enumerate(top_k_indices) 
            if final_scores[index] >= threshold
        ]
        if not cosine_results:
            return "No relevant documents found."

        print(cosine_results)
        
        results = {idx[0] : document_indices[idx[0]] for idx in cosine_results}
        return results
