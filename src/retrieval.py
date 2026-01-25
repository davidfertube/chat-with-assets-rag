from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder

class AdvancedRetriever:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = Chroma(
            persist_directory="../data/vector_store", 
            embedding=self.embeddings
        )
        # Reranker model
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def retrieve(self, query: str, k=10):
        # 1. Semantic Search
        docs = self.vectorstore.similarity_search(query, k=k)
        
        # 2. Re-Ranking
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_docs[:5]]