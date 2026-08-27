from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self):
        # Load https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")        
    
    def embed_chunks(self, code_chunks : list[str]) -> list[list[float]]:
        chunk_embeddings = self.model.encode(code_chunks)
        return chunk_embeddings
    def embed_query(self, query : str) -> list[float]:
        query_embedding = self.model.encode([query])[0]
        return query_embedding   