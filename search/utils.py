import re
from typing import Tuple, List, Union
import numpy as np
import h5py
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
import asyncio

def parse_html(text : Union[str, List[str]]) -> str:
    if isinstance(text, str):
        text = [text]
    output = [str(BeautifulSoup(doc, 'html.parser')) for doc in text]
    return output

def keyword_check(query: str) -> Tuple[bool, List[str]]:
    keywords = re.findall(r'["\']([^"\']+)["\']', query)
    
    query_without_quotes = re.sub(r'["\'][^"\']+["\']', '', query).strip()
    
    flag = bool(query_without_quotes and query_without_quotes != query)
    
    return flag, keywords

class Content(ABC):
    @abstractmethod
    async def save_async(file_path : str, 
                         item : Union[np.ndarray, List]
                         ):
        pass

    @abstractmethod
    async def add_async(file_path : str, 
                        new_items : Union[np.ndarray, List]
                        ):
        pass

    @abstractmethod
    async def load_async(file_path : str, 
                         indices : Union[str, List[int]]
                         ) -> Union[np.ndarray, List[str]]:
        pass

    @abstractmethod
    async def save(file_path : str, 
                   item : Union[np.ndarray, List]
                   ):
        pass

    @abstractmethod
    async def add(file_path : str, 
                  new_items : Union[np.ndarray, List]
                  ):
        pass

    @abstractmethod
    async def load(file_path : str, 
                   indices : Union[str, List[int]]
                   ) -> Union[np.ndarray, List[str]]:
        pass

class Embeddings(Content):
    
    @staticmethod
    async def save_async(file_path: str, 
                   item: np.ndarray, 
                   extension: str = 'npy'):
        
        if extension == 'h5py':
            await asyncio.to_thread(
                lambda: h5py.File(file_path, 'w').create_dataset('embeddings', data=item, maxshape=(None, item.shape[1]))
            )
        elif extension == 'npy':
            await asyncio.to_thread(np.save, file_path, item)

    @staticmethod
    async def add_async(file_path: str, 
                  new_items: np.ndarray, 
                  extension: str = 'npy'):
        
        if new_items.ndim == 1:
            new_items = new_items.reshape(1, -1)

        if extension == 'h5py':
            def add_to_h5py():
                with h5py.File(file_path, 'a') as f:
                    embeddings = f['embeddings']
                    new_shape = (embeddings.shape[0] + new_items.shape[0], embeddings.shape[1])
                    embeddings.resize(new_shape)
                    embeddings[-new_items.shape[0]:] = new_items

            await asyncio.to_thread(add_to_h5py)

        elif extension == 'npy':
            if not os.path.exists(file_path):
                await asyncio.to_thread(np.save, file_path, new_items)
            else:
                def add_to_npy():
                    existing_embeddings = np.load(file_path, mmap_mode='r')
                    new_shape = (existing_embeddings.shape[0] + new_items.shape[0], existing_embeddings.shape[1])

                    temp_file_path = file_path + '.tmp.npy'
                    with open(temp_file_path, 'wb') as f:
                        np.lib.format.write_array(f, np.empty(new_shape, dtype=existing_embeddings.dtype))

                    with np.load(temp_file_path, mmap_mode='r+') as temp_embeddings:
                        temp_embeddings[:existing_embeddings.shape[0]] = existing_embeddings[:]
                        temp_embeddings[existing_embeddings.shape[0]:] = new_items

                    os.replace(temp_file_path, file_path)

                await asyncio.to_thread(add_to_npy)
                
    @staticmethod
    async def load_async(file_path: str, 
                   indices: Union[str, List[int]], 
                   extension: str = 'npy') -> np.ndarray:
        
        if extension == 'h5py':
            def load_from_h5py():
                with h5py.File(file_path, 'r') as f:
                    if indices == 'all':
                        return np.array(f['embeddings'][:])
                    return np.array(f['embeddings'][indices])

            return await asyncio.to_thread(load_from_h5py)

        elif extension == 'npy':
            def load_from_npy():
                embeddings = np.load(file_path, mmap_mode='r')
                if indices == 'all':
                    return embeddings[:]
                elif isinstance(indices, int):
                    return embeddings[indices]
                else:
                    return embeddings[indices]

            return await asyncio.to_thread(load_from_npy)
            
    @staticmethod
    async def save(file_path: str, 
                   item: np.ndarray, 
                   extension: str = 'npy'):
        await Embeddings.save_async(file_path, item, extension)

    @staticmethod
    async def add(file_path: str, 
                  new_items: np.ndarray, 
                  extension: str = 'npy'):
        await Embeddings.add_async(file_path, new_items, extension)
     
    @staticmethod
    async def load(file_path: str, 
                   indices: Union[str, List[int]], 
                   extension: str = 'npy') -> np.ndarray:
        return await Embeddings.load_async(file_path, indices, extension)
    
class Documents(Content):

    @staticmethod
    async def save_async(file_path: str, 
                   item: List[str]):
        def save_to_h5py():
            with h5py.File(file_path, 'w') as f:
                dt = h5py.string_dtype(encoding='utf-8')  # Specify string encoding
                f.create_dataset('strings', data=item, dtype=dt, maxshape=(None,))

        await asyncio.to_thread(save_to_h5py)

    @staticmethod       
    async def add_async(file_path: str, 
                  new_items: Union[str, List[str]]):
        if isinstance(new_items, str):
            new_documents = np.array([new_items])
        else:
            new_documents = np.array(new_items)

        def add_to_h5py():
            with h5py.File(file_path, 'a') as f:
                if 'strings' not in f:
                    dt = h5py.string_dtype(encoding='utf-8')
                    documents = f.create_dataset('strings', (0,), maxshape=(None,), dtype=dt)
                else:
                    documents = f['strings']
                
                new_size = documents.shape[0] + len(new_documents)
                documents.resize((new_size,))
                documents[-len(new_documents):] = new_documents

        await asyncio.to_thread(add_to_h5py)

    @staticmethod
    async def load_async(file_path: str, 
                   indices: Union[str, List[int]]) -> List[str]:
        def load_from_h5py():
            with h5py.File(file_path, 'r') as f:
                if indices == 'all':
                    return f['strings'][:].tolist()
                else:
                    return f['strings'][indices].tolist()

        return await asyncio.to_thread(load_from_h5py)
        
    @staticmethod
    async def save(file_path: str, 
                   item: List[str]):
        await Documents.save_async(file_path, item)

    @staticmethod
    async def add(file_path: str, 
                  new_items: Union[str, List[str]]):
        await Documents.add_async(file_path, new_items)
     
    @staticmethod
    async def load(file_path: str, 
                   indices: Union[str, List[int]]) -> List[str]:
        return await Documents.load_async(file_path, indices)
        

def reciprocal_rank_fusion(knn_results, 
                           cosine_results, k
                           ):
    """
    Apply Reciprocal Rank Fusion to combine KNN and Cosine Similarity results.

    Parameters:
    - knn_results: List of tuples (document_id, rank) from KNN model.
    - cosine_results: List of tuples (document_id, rank) from Cosine Similarity model.
    - k: Constant to control the influence of ranks.

    Returns:
    - fused_scores: Dictionary with document_ids as keys and their fused RRF scores as values.
    """

    # Initialize a dictionary to hold the fused scores
    fused_scores = {}

    # Process KNN results
    for rank, (doc_id, _) in enumerate(knn_results.items()):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # Process Cosine Similarity results
    for rank, (doc_id, _) in enumerate(cosine_results.items()):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return fused_scores


class TFIDF():
    def __init__(self, vectorizer_path : str):

        if vectorizer_path:
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.vectorizer = TfidfVectorizer()
            
    def transform(self, documents : List):

        return self.vectorizer.fit_transform(documents) 
    
    def save(self, file_path : str):

        joblib.dump(file_path)
        print(f'saved to {file_path}')

    def load_matrix(self, file_path : str):

        self.tfidf_matrix = joblib.load(file_path)
        return self.tfidf_matrix
    
    def add_newdocuments(self, documents : List):
        
        new_vector = self.vectorizer.transform(documents)
        updated_tfidf_matrix = np.vstack([self.tfidf_matrix, new_vector])
        return updated_tfidf_matrix
    
#if __name__ == '__main__':
#    binary_embd = asyncio.run(Embeddings.load(file_path=r'data\wikipedia-dataset-embeddings-binary.npy', indices='all', extension='npy'))
#    print(binary_embd.shape)