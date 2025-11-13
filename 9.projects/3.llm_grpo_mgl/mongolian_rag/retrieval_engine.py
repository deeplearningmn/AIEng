#!/usr/bin/env python3
"""
Retrieval Engine for Mongolian Historical RAG System.

This module provides similarity search and retrieval functionality
using the FAISS index created by the embedding pipeline.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging

import faiss
from openai import OpenAI


@dataclass
class RetrievalResult:
    """Result from similarity search."""
    
    text: str
    score: float
    metadata: Dict[str, Any]
    rank: int


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    
    top_k: int = 5
    score_threshold: float = 0.0
    rerank: bool = True
    deduplicate: bool = True
    max_chunk_per_entry: int = 3


class MongolianHistoryRetriever:
    """Retrieval engine for Mongolian historical data."""
    
    def __init__(self, embeddings_dir: str, openai_api_key: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Load components
        self.index = None
        self.metadata = None
        self.config = None
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the FAISS index, metadata, and configuration."""
        try:
            # Load configuration
            config_path = self.embeddings_dir / 'config.json'
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # Load FAISS index
            index_path = self.embeddings_dir / 'mongolian_history.faiss'
            self.index = faiss.read_index(str(index_path))
            
            # Load metadata
            metadata_path = self.embeddings_dir / 'metadata.json'
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            self.logger.info(f"Loaded pipeline with {len(self.metadata)} chunks")
            
        except Exception as e:
            self.logger.error(f"Failed to load pipeline: {e}")
            raise
    
    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for search query."""
        try:
            response = self.client.embeddings.create(
                model=self.config['embedding_model'],
                input=[query],
                encoding_format="float"
            )
            
            embedding = np.array([response.data[0].embedding])
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise
    
    def search(self, query: str, config: SearchConfig = None) -> List[RetrievalResult]:
        """Search for relevant historical content."""
        if config is None:
            config = SearchConfig()
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query)
        
        # Perform similarity search
        scores, indices = self.index.search(query_embedding, config.top_k * 2)
        
        # Process results
        results = []
        seen_entries = set()
        entry_chunk_counts = {}
        
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
            
            if score < config.score_threshold:
                continue
            
            metadata = self.metadata[idx]
            entry_id = metadata['entry_id']
            
            # Deduplication by entry
            if config.deduplicate and entry_id in seen_entries:
                # Check if we've reached max chunks per entry
                if entry_chunk_counts.get(entry_id, 0) >= config.max_chunk_per_entry:
                    continue
                entry_chunk_counts[entry_id] = entry_chunk_counts.get(entry_id, 0) + 1
            else:
                seen_entries.add(entry_id)
                entry_chunk_counts[entry_id] = 1
            
            result = RetrievalResult(
                text=metadata['text'],
                score=float(score),
                metadata=metadata,
                rank=len(results)
            )
            
            results.append(result)
            
            if len(results) >= config.top_k:
                break
        
        # Rerank if requested
        if config.rerank and len(results) > 1:
            results = self._rerank_results(query, results)
        
        return results
    
    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results based on additional criteria."""
        # Simple reranking based on multiple factors
        for result in results:
            metadata = result.metadata
            
            # Boost score based on various factors
            boost = 0.0
            
            # Boost recent periods slightly
            period = metadata.get('period', '')
            if 'XX зуун' in period or 'XXI зуун' in period:
                boost += 0.1
            elif 'XIX зуун' in period:
                boost += 0.05
            
            # Boost if query terms appear in title
            title = metadata.get('title', '').lower()
            query_lower = query.lower()
            if any(word in title for word in query_lower.split()):
                boost += 0.2
            
            # Boost longer, more substantial content
            content_length = metadata.get('content_length', 0)
            if content_length > 1000:
                boost += 0.1
            elif content_length > 500:
                boost += 0.05
            
            # Apply boost (convert similarity to distance and back)
            result.score = result.score - boost
        
        # Re-sort by adjusted score
        results.sort(key=lambda x: x.score)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i
        
        return results
    
    def search_by_period(self, period: str, query: str = "", top_k: int = 10) -> List[RetrievalResult]:
        """Search for content from a specific historical period."""
        if query:
            # Combine period and query search
            enhanced_query = f"{period} {query}"
            results = self.search(enhanced_query, SearchConfig(top_k=top_k * 2))
            
            # Filter by period
            filtered_results = [
                r for r in results 
                if period.lower() in r.metadata.get('period', '').lower()
            ]
            
            return filtered_results[:top_k]
        else:
            # Return all chunks from the period
            period_results = []
            
            for i, metadata in enumerate(self.metadata):
                if period.lower() in metadata.get('period', '').lower():
                    result = RetrievalResult(
                        text=metadata['text'],
                        score=0.0,  # No similarity score for period-only search
                        metadata=metadata,
                        rank=len(period_results)
                    )
                    period_results.append(result)
            
            return period_results[:top_k]
    
    def search_by_source(self, source: str, query: str = "", top_k: int = 10) -> List[RetrievalResult]:
        """Search for content from a specific source."""
        if query:
            results = self.search(query, SearchConfig(top_k=top_k * 2))
            
            # Filter by source
            filtered_results = [
                r for r in results 
                if source.lower() in r.metadata.get('source', '').lower()
            ]
            
            return filtered_results[:top_k]
        else:
            # Return all chunks from the source
            source_results = []
            
            for i, metadata in enumerate(self.metadata):
                if source.lower() in metadata.get('source', '').lower():
                    result = RetrievalResult(
                        text=metadata['text'],
                        score=0.0,
                        metadata=metadata,
                        rank=len(source_results)
                    )
                    source_results.append(result)
            
            return source_results[:top_k]
    
    def get_context_for_rag(self, query: str, max_tokens: int = 4000) -> Tuple[str, List[Dict[str, Any]]]:
        """Get formatted context for RAG applications."""
        results = self.search(query, SearchConfig(top_k=10))
        
        context_parts = []
        sources = []
        total_length = 0
        
        for result in results:
            # Format context entry
            metadata = result.metadata
            
            # Create source info
            source_info = {
                'title': metadata.get('title', 'Unknown'),
                'source': metadata.get('source', 'Unknown'),
                'period': metadata.get('period', 'Unknown'),
                'date': metadata.get('date', ''),
                'chunk_id': metadata.get('chunk_id', 0),
                'score': result.score
            }
            
            # Format text with metadata
            text_with_meta = f"[{metadata.get('period', 'Unknown')}] {result.text}"
            
            # Check token limit (rough estimate: 1 token ≈ 4 characters)
            estimated_tokens = len(text_with_meta) // 4
            if total_length + estimated_tokens > max_tokens:
                break
            
            context_parts.append(text_with_meta)
            sources.append(source_info)
            total_length += estimated_tokens
        
        # Join context
        context = "\n\n---\n\n".join(context_parts)
        
        return context, sources
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retrieval system."""
        if not self.metadata:
            return {}
        
        from collections import Counter
        
        # Period distribution
        periods = [m.get('period', 'Unknown') for m in self.metadata]
        period_dist = dict(Counter(periods))
        
        # Source distribution
        sources = [m.get('source', 'Unknown') for m in self.metadata]
        source_dist = dict(Counter(sources))
        
        # Content statistics
        content_lengths = [len(m.get('text', '')) for m in self.metadata]
        
        return {
            'total_chunks': len(self.metadata),
            'total_entries': len(set(m['entry_id'] for m in self.metadata)),
            'embedding_dimension': self.config.get('embedding_dimension', 0),
            'index_type': self.config.get('index_type', 'Unknown'),
            'period_distribution': period_dist,
            'source_distribution': source_dist,
            'avg_chunk_length': np.mean(content_lengths) if content_lengths else 0,
            'min_chunk_length': min(content_lengths) if content_lengths else 0,
            'max_chunk_length': max(content_lengths) if content_lengths else 0
        }


def main():
    """Demo function for the retrieval engine."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Test Mongolian history retrieval engine")
    parser.add_argument('--embeddings-dir', default='data/embeddings',
                       help='Directory containing embeddings and index')
    parser.add_argument('--query', required=True,
                       help='Search query')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to return')
    parser.add_argument('--period', help='Filter by historical period')
    parser.add_argument('--source', help='Filter by source')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    # Create retriever
    retriever = MongolianHistoryRetriever(args.embeddings_dir, api_key)
    
    # Perform search
    if args.period:
        results = retriever.search_by_period(args.period, args.query, args.top_k)
    elif args.source:
        results = retriever.search_by_source(args.source, args.query, args.top_k)
    else:
        config = SearchConfig(top_k=args.top_k)
        results = retriever.search(args.query, config)
    
    # Display results
    print(f"\nSearch Results for: '{args.query}'")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        metadata = result.metadata
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Title: {metadata.get('title', 'N/A')[:80]}...")
        print(f"   Period: {metadata.get('period', 'N/A')}")
        print(f"   Source: {metadata.get('source', 'N/A')}")
        print(f"   Text: {result.text[:200]}...")
        
        if i < len(results):
            print("-" * 40)
    
    # Show statistics
    stats = retriever.get_statistics()
    print(f"\n\nRetrieval System Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    print(f"  Average chunk length: {stats['avg_chunk_length']:.0f} characters")


if __name__ == "__main__":
    main()