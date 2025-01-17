from sentence_transformers import SentenceTransformer, util

def reciprocal_rank_fusion(knn_results, cosine_results, k):
    """
    Apply Reciprocal Rank Fusion to combine KNN and Cosine Similarity results.

    Parameters:
    - knn_results: Dictionary (document_id, rank) from KNN model.
    - cosine_results: Dictionary (document_id, rank) from Cosine Similarity model.
    - k: Constant to control the influence of ranks.

    Returns:
    - fused_scores: Dictionary with document_ids as keys and their fused RRF scores as values.
    """

    fused_scores = {}

    for rank, (doc_id, _) in enumerate(knn_results.items()):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    for rank, (doc_id, _) in enumerate(cosine_results.items()):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

    fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    return fused_scores

class BiEncoderReranker:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def rerank(self, query, documents):
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        document_embeddings = self.model.encode(documents, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)

        similarities = similarities[0].tolist()

        ranked_results = sorted(zip(documents, similarities), key=lambda x: x[1], reverse=True)
        return ranked_results, similarities