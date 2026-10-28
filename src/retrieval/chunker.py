from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class SemanticChunker:
    """
    Splits raw text and documents into semantically coherent passages.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of LangChain Document objects into smaller text chunks.
        """
        return self.splitter.split_documents(documents)

    def chunk_text(self, text: str, metadata: dict = None) -> List[Document]:
        """
        Splits raw text strings into chunked Document objects.
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.splitter.split_documents([doc])