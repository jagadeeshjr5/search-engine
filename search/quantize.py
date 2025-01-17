import numpy as np
import warnings
from typing import Union, Tuple
warnings.filterwarnings('ignore')


def quantize(doc_embeddings: np.ndarray, 
             precision: str = None, 
             calibration: Union[str, Tuple[float, float]] = 'auto'
             ) -> np.ndarray:
    """
    Quantize document embeddings to either binary or int8 precision.

    This function performs quantization on the input document embeddings based on the specified precision.
    It centers the embeddings for binary quantization or scales them for int8 quantization, using an optional
    calibration range to define the minimum and maximum values.

    Parameters:
    ----------
    doc_embeddings : np.ndarray
        A 2D NumPy array containing the document embeddings to be quantized. 
        Each row represents a document, and each column represents a feature of the embedding.

    precision : str, optional
        The precision for quantization. Must be either 'binary' or 'int8'.
        - 'binary': Converts embeddings to binary values (-1 and 1).
        - 'int8': Scales embeddings to fit within the int8 range (-128 to 127).

    calibration : Union[str, Tuple[float, float]], optional
        Determines the scaling for int8 quantization. 
        - If 'auto', the function automatically calculates the min and max from the input embeddings.
        - If a tuple, it should contain two float values representing the (min, max) range for scaling.

    Returns:
    -------
    np.ndarray
        A NumPy array containing the quantized embeddings. The output type depends on the specified precision:
        - For 'binary', an array of binary values (-1, 1).
        - For 'int8', an array of int8 values scaled between -128 and 127.

    Raises:
    ------
    ValueError
        If the precision is not 'binary' or 'int8'.
        If the calibration parameter is not 'auto' or a valid tuple of (min, max).

    Examples:
    --------
    # Example document embeddings
    doc_embeddings = np.random.rand(10, 128)  # 10 documents, 128-dimensional embeddings

    # Quantization examples
    binary_embeddings = quantize(doc_embeddings, precision='binary')
    int8_embeddings_auto = quantize(doc_embeddings, precision='int8')
    int8_embeddings_custom = quantize(doc_embeddings, precision='int8', calibration=(-1, 1))
    """
    
    if precision not in ['binary', 'int8']:
        raise ValueError("Input precision for quantization must be either 'binary' or 'int8'")
    elif precision == 'binary':
        mean_embeddings = np.mean(doc_embeddings, axis=0)
        document_embeddings_centered = doc_embeddings - mean_embeddings
    
        binary_document_embeddings = np.where(document_embeddings_centered >= 0, 1, -1)
        binary_document_embeddings = (binary_document_embeddings > 0).astype('uint8')
        if binary_document_embeddings.ndim == 1:
            binary_document_embeddings = np.packbits(binary_document_embeddings.reshape(1, -1), axis=1)
        else:
            binary_document_embeddings = np.packbits(binary_document_embeddings, axis=1)
        
        return binary_document_embeddings
    elif precision == 'int8':
        if calibration == 'auto':
            min_val, max_val = doc_embeddings.min(), doc_embeddings.max()
        elif isinstance(calibration, Tuple) and len(calibration) == 2:
            min_val, max_val = calibration
        else:
            raise ValueError("Calibration must be 'auto' or a tuple of (min, max)")
            
        scale = 127 / max(abs(min_val), abs(max_val))
        embeddings_scaled = doc_embeddings * scale
        
        int8_embeddings = np.clip(embeddings_scaled, -128, 127).astype(np.int8)
    
        return int8_embeddings