import numpy as np
import torch
import google.generativeai as genai
from typing import List
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

class Embed:
    """
    Embedding generator class for models such as 'gemini' and 'sentence-transformers'.
    
    Attributes:
    - model (str): The name of the embedding model to use ('gemini' or a 'sentence-transformers' model. For sentence-transformers, the model name must starts with sentence-transformers.).
    - out_dims (int): The dimensionality of the output embeddings.
    - api_key (str): API key required for 'gemini' model.
    - device (str): Specifies the device for computation ('auto', 'cpu', 'cuda', 'mps', 'npu').
    
    Methods:
    - generate_embeddings(documents: List[str]) -> np.ndarray: Generates embeddings for a list of documents.
    """
    
    def __init__(self, 
                 model: str, 
                 out_dims: int, 
                 api_key: str = None, 
                 device: str = 'auto'
                 ):
        """
        Initializes the Embed class with the specified model, output dimensions, API key, and device type.

        Parameters:
        - model (str): The name of the embedding model to use ('gemini' or a 'sentence-transformers' model. For sentence-transformers, the model name must starts with sentence-transformers.).
        - out_dims (int): The dimensionality of the output embeddings.
        - api_key (str, optional): API key required for the 'gemini' model. Defaults to None.
        - device (str, optional): Specifies the device for computation ('auto', 'cpu', 'cuda', 'mps', 'npu'). Defaults to 'auto'.
        
        Raises:
        - ValueError: If 'gemini' model is selected without an API key, or if an unsupported model is specified.
        """
        self.model = model.lower()
        self.out_dims = out_dims
        self.api_key = api_key
        
        if self.model == "gemini":
            if not api_key:
                raise ValueError("API key is required for the Gemini model.")
        elif not (self.model.startswith("sentence-transformers") or self.model == "gemini"):
            raise ValueError("The specified model is not supported. Please use 'gemini' or a 'sentence-transformers' model.")
        
        if self.model.startswith('sentence-transformers'):
            self.device = self._get_device(device)
            model_name = '/'.join(self.model.split('/')[1:])
            self.st_model = SentenceTransformer(model_name_or_path=model_name, device=self.device, truncate_dim=self.out_dims)        

    def _get_device(self, 
                    device: str
                    ) -> torch.device:
        """
        Determines and returns the appropriate device for model computation.

        Parameters:
        - device (str): The desired device ('auto', 'cpu', 'cuda', 'mps', 'npu').
        
        Returns:
        - torch.device: The selected device for computation.
        
        Raises:
        - ValueError: If an invalid device is specified.
        """
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            elif hasattr(torch.backends, 'npu') and torch.backends.npu.is_available():
                return torch.device("npu")
            else:
                return torch.device("cpu")
        elif device in ['cpu', 'cuda', 'mps', 'npu']:
            return torch.device(device)
        else:
            raise ValueError("Invalid device specified. Choose from 'auto', 'cpu', 'cuda', 'mps', or 'npu'.")

    def generate_embeddings(self, 
                            documents: List[str]
                            ) -> np.ndarray:
        """
        Generates embeddings for a list of documents based on the specified model.

        Parameters:
        - documents (List[str]): A list of document strings to generate embeddings for.
        
        Returns:
        - np.ndarray: An array of generated embeddings.
        
        Raises:
        - ValueError: If an error occurs in processing with an unsupported model.
        """
        if self.model == 'gemini':
            genai.configure(api_key=self.api_key)
            result = genai.embed_content_async(
                model="models/embedding-001",
                content=documents,
                task_type="retrieval_document",
                output_dimensionality=self.out_dims,
            )
            return result
        
        elif self.model.startswith('sentence-transformers'):
            embeddings = self.st_model.encode(documents)
            return embeddings