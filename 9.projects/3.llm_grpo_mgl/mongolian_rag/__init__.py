"""
Mongolian History RAG System

A complete Retrieval-Augmented Generation system for Mongolian historical data.
Includes embedding generation, FAISS indexing, and question-answering capabilities.
"""

from .embedding_pipeline import MongolianHistoryEmbedder, EmbeddingConfig
from .retrieval_engine import MongolianHistoryRetriever, RetrievalResult, SearchConfig
from .rag_agent import MongolianHistoryRAG, RAGConfig, create_interactive_session

__version__ = "1.0.0"
__author__ = "Mongolian History RAG Team"

__all__ = [
    "MongolianHistoryEmbedder",
    "EmbeddingConfig", 
    "MongolianHistoryRetriever",
    "RetrievalResult",
    "SearchConfig",
    "MongolianHistoryRAG",
    "RAGConfig",
    "create_interactive_session"
]