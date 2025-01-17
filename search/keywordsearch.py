from tqdm import tqdm
from whoosh.index import open_dir
from whoosh.index import create_in
from whoosh.qparser import QueryParser
import whoosh
from typing import List, Dict, Union
import os
import asyncio
from aiocache import cached
from aiocache.serializers import JsonSerializer

class KeywordSearch:
    
    def __init__(self):
        """
        Initializes the KeywordSearch object.
        """
        pass
        
    def create_index(self, 
                     index_path: str, 
                     schema: whoosh.fields.Schema
                     ):
        """
        Creates a Whoosh search index at the specified path if it does not already exist.
        
        Args:
            index_path (str): The path where the index will be stored.
            schema (whoosh.fields.Schema): The schema to define the structure of the index.
        
        Returns:
            whoosh.index.FileIndex: The created index if it doesn't exist.
            None: If the index already exists at the specified path.
        """
        if not os.path.exists(index_path):
            return create_in(index_path, schema)
        else:
            print("Index already exists at this path")
            
    def open_index(self, 
                   index_path: str
                   ):
        """
        Opens an existing Whoosh index from the specified path.
        
        Args:
            index_path (str): The path where the index is located.
        
        Returns:
            whoosh.index.FileIndex: The opened index.
            None: If the index does not exist at the specified path.
        """
        if os.path.exists(index_path):
            return open_dir(index_path)
        else:
            print("Index does not exist at this path")
            
    def add_document(self, 
                     index: whoosh.index.FileIndex, 
                     documents: Union[List, List[Dict]]
                     ):
        """
        Adds documents to the Whoosh index. This method supports both a list of strings 
        or a list of dictionaries where each dictionary contains multiple fields.
        
        Args:
            index (whoosh.index.FileIndex): The Whoosh index where documents will be added.
            documents (Union[List, List[Dict]]): A list of documents to be indexed. 
                Each document can either be a string or a dictionary containing field names and values.
        
        Returns:
            None: Adds the documents and commits them to the index.
        """
        writer = index.writer()
        if isinstance(documents, list):
            for doc in tqdm(documents, total=len(documents)):
                writer.add_document(content=doc)
        elif isinstance(documents[0], dict):
            for doc in tqdm(documents, total=len(documents)):
                writer.add_document(**doc)
        
        writer.commit()
        print(f"Added {len(documents)} documents to the index.")

    @cached(ttl=600, serializer=JsonSerializer())  
    async def search(self, 
                     index: whoosh.index.FileIndex, 
                     keywords: Union[str, List[str]]
                     ) -> List:
        """
        Searches the Whoosh index for documents containing the given keywords.
        
        Supports both single and multiple keywords. Returns a list of document IDs 
        that match the keywords.
        
        Args:
            index (whoosh.index.FileIndex): The Whoosh index to search.
            keywords (Union[str, List[str]]): A keyword or a list of keywords to search for.
        
        Returns:
            List: A list of document IDs (docnums) that match the keywords.
        """
        if isinstance(keywords, str):
            keywords = [keywords]
        
        search_results = []
    
        def run_search(keyword: str) -> List:
            """
            Runs the search for a single keyword.

            Args:
                keyword (str): The keyword to search for.
                
            Returns:
                List: A list of document IDs that match the keyword.
            """
            with index.searcher() as searcher:
                query = QueryParser("content", index.schema).parse(f'"{keyword}"')
                results = searcher.search(query, limit=None)
                
                if results:
                    print(f"Found {len(results)} results for '{keyword}':")
                    results_dict = [result.docnum for result in results]
                    return results_dict
                else:
                    print(f"No results found for '{keyword}'.")
                    return []
    
        search_tasks = []
        for keyword in keywords:
            search_tasks.append(asyncio.get_event_loop().run_in_executor(None, run_search, keyword))
        
        results = await asyncio.gather(*search_tasks)
        
        # Flatten the results if needed, or just return the list of results
        for result in results:
            search_results.extend(result)  # Combine all doc IDs into one list
        
        return search_results
