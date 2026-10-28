import os
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class QdrantHybridSearch:
    """
    Manages vector embeddings and retrieval using Qdrant vector DB.
    """
    def __init__(
        self, 
        collection_name: str = "rag_documents", 
        qdrant_url: str = None, 
        api_key: str = None
    ):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        key = api_key or os.getenv("QDRANT_API_KEY", None)
        
        self.client = QdrantClient(url=url, api_key=key)
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Creates the Qdrant collection if it does not already exist.
        """
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            # text-embedding-3-small uses 1536 dimensions
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

    async def add_documents(self, documents: List[Document]):
        """
        Embeds and indexes a list of chunked documents into Qdrant.
        """
        texts = [doc.page_content for doc in documents]
        vectors = await self.embeddings.aembed_documents(texts)
        
        points = [
            PointStruct(
                id=idx,
                vector=vector,
                payload={"page_content": doc.page_content, "metadata": doc.metadata}
            )
            for idx, (doc, vector) in enumerate(zip(documents, vectors))
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)

    async def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant document chunks for a query.
        """
        query_vector = await self.embeddings.aembed_query(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )
        
        retrieved_docs = []
        for hit in search_result:
            retrieved_docs.append({
                "page_content": hit.payload.get("page_content", ""),
                "metadata": hit.payload.get("metadata", {}),
                "score": hit.score
            })
        
        return retrieved_docs