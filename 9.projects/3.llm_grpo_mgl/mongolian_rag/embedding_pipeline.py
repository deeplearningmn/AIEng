#!/usr/bin/env python3
"""
Embedding and Retrieval Pipeline for Mongolian Historical Data.

This module creates embeddings using OpenAI's text-embedding-3-small model
and builds a FAISS index for efficient similarity search in RAG applications.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

import faiss
from openai import OpenAI
from tqdm import tqdm


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding pipeline."""
    
    # OpenAI settings
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    max_tokens: int = 8191  # Max tokens for text-embedding-3-small
    
    # Processing settings
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    batch_size: int = 100  # Embeddings per batch
    
    # FAISS settings
    index_type: str = "IVF"  # IVF, Flat, or HNSW
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    
    # Output settings
    output_dir: str = "data/embeddings"
    index_filename: str = "mongolian_history.faiss"
    metadata_filename: str = "metadata.json"
    config_filename: str = "config.json"


class TextChunker:
    """Split text into overlapping chunks for embedding."""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata."""
        if len(text) <= self.chunk_size:
            return [{
                'text': text,
                'chunk_id': 0,
                'total_chunks': 1,
                **metadata
            }]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end),
                    text.rfind('\n\n', start, end)
                )
                
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'start_pos': start,
                    'end_pos': end,
                    **metadata
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.overlap
            if start >= len(text):
                break
        
        # Add total chunks to all chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks


class EmbeddingGenerator:
    """Generate embeddings using OpenAI's API."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.logger = logging.getLogger(__name__)
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = np.array([item.embedding for item in response.data])
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches to handle rate limits."""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), 
                     desc="Generating embeddings"):
            batch = texts[i:i + self.config.batch_size]
            
            # Truncate texts that are too long
            truncated_batch = []
            for text in batch:
                if len(text) > self.config.max_tokens * 4:  # Rough token estimate
                    text = text[:self.config.max_tokens * 4]
                truncated_batch.append(text)
            
            batch_embeddings = self.generate_embeddings(truncated_batch)
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)


class FAISSIndexBuilder:
    """Build and manage FAISS index for similarity search."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def build_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Build FAISS index from embeddings."""
        dimension = embeddings.shape[1]
        n_vectors = embeddings.shape[0]
        
        self.logger.info(f"Building FAISS index for {n_vectors} vectors of dimension {dimension}")
        
        if self.config.index_type == "Flat":
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(dimension)
            
        elif self.config.index_type == "IVF":
            # Approximate search using IVF (Inverted File)
            quantizer = faiss.IndexFlatL2(dimension)
            nlist = min(self.config.nlist, n_vectors // 10)  # Ensure reasonable nlist
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            self.logger.info("Training IVF index...")
            index.train(embeddings.astype(np.float32))
            index.nprobe = self.config.nprobe
            
        elif self.config.index_type == "HNSW":
            # Hierarchical Navigable Small World
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 128
            
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        # Add vectors to index
        self.logger.info("Adding vectors to index...")
        index.add(embeddings.astype(np.float32))
        
        self.logger.info(f"Index built successfully with {index.ntotal} vectors")
        return index
    
    def save_index(self, index: faiss.Index, output_path: Path):
        """Save FAISS index to disk."""
        faiss.write_index(index, str(output_path))
        self.logger.info(f"Index saved to {output_path}")
    
    def load_index(self, index_path: Path) -> faiss.Index:
        """Load FAISS index from disk."""
        index = faiss.read_index(str(index_path))
        self.logger.info(f"Index loaded from {index_path}")
        return index


class MongolianHistoryEmbedder:
    """Main class for creating embeddings from Mongolian historical data."""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.chunker = TextChunker(config.chunk_size, config.chunk_overlap)
        self.embedder = EmbeddingGenerator(config)
        self.index_builder = FAISSIndexBuilder(config)
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load the unified JSONL dataset."""
        entries = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_num}: {e}")
        
        self.logger.info(f"Loaded {len(entries)} entries from dataset")
        return entries
    
    def prepare_chunks(self, entries: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Prepare text chunks and metadata for embedding."""
        all_chunks = []
        all_metadata = []
        
        for entry_id, entry in enumerate(entries):
            text = entry.get('text', '')
            if not text:
                continue
            
            # Create base metadata
            base_metadata = {
                'entry_id': entry_id,
                'title': entry.get('title', ''),
                'source': entry.get('source', ''),
                'period': entry.get('period', ''),
                'date': entry.get('date', ''),
                'url': entry.get('url', ''),
                'chapter': entry.get('chapter', ''),
                'word_count': entry.get('word_count', 0),
                'content_length': len(text)
            }
            
            # Create chunks
            chunks = self.chunker.chunk_text(text, base_metadata)
            
            for chunk in chunks:
                all_chunks.append(chunk['text'])
                all_metadata.append(chunk)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(entries)} entries")
        return all_chunks, all_metadata
    
    def create_embeddings(self, dataset_path: str) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Create embeddings and FAISS index from dataset."""
        # Load dataset
        entries = self.load_dataset(dataset_path)
        
        # Prepare chunks
        texts, metadata = self.prepare_chunks(entries)
        
        if not texts:
            raise ValueError("No text found in dataset")
        
        # Generate embeddings
        self.logger.info("Generating embeddings...")
        embeddings = self.embedder.generate_batch_embeddings(texts)
        
        # Build FAISS index
        self.logger.info("Building FAISS index...")
        index = self.index_builder.build_index(embeddings)
        
        return index, metadata
    
    def save_pipeline(self, index: faiss.Index, metadata: List[Dict[str, Any]]):
        """Save the complete pipeline to disk."""
        output_dir = Path(self.config.output_dir)
        
        # Save FAISS index
        index_path = output_dir / self.config.index_filename
        self.index_builder.save_index(index, index_path)
        
        # Save metadata
        metadata_path = output_dir / self.config.metadata_filename
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save configuration
        config_path = output_dir / self.config.config_filename
        config_dict = {
            'embedding_model': self.config.embedding_model,
            'embedding_dimension': self.config.embedding_dimension,
            'chunk_size': self.config.chunk_size,
            'chunk_overlap': self.config.chunk_overlap,
            'index_type': self.config.index_type,
            'nlist': self.config.nlist,
            'nprobe': self.config.nprobe,
            'created_at': datetime.now().isoformat(),
            'total_chunks': len(metadata),
            'index_size': index.ntotal
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Pipeline saved to {output_dir}")
        
        # Create summary
        summary = {
            'total_entries': len(set(m['entry_id'] for m in metadata)),
            'total_chunks': len(metadata),
            'embedding_dimension': self.config.embedding_dimension,
            'index_type': self.config.index_type,
            'avg_chunk_length': np.mean([len(m['text']) for m in metadata]),
            'period_distribution': {}
        }
        
        # Calculate period distribution
        from collections import Counter
        periods = [m['period'] for m in metadata if m['period']]
        summary['period_distribution'] = dict(Counter(periods))
        
        summary_path = output_dir / 'pipeline_summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary


def main():
    """Main function to create embeddings pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create embeddings pipeline for Mongolian historical data")
    parser.add_argument('--dataset', default='data/mongolian_history_unified.jsonl',
                       help='Path to the unified JSONL dataset')
    parser.add_argument('--output-dir', default='data/embeddings',
                       help='Output directory for embeddings and index')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Size of text chunks in characters')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help='Overlap between chunks in characters')
    parser.add_argument('--index-type', choices=['Flat', 'IVF', 'HNSW'], default='IVF',
                       help='Type of FAISS index to build')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Batch size for embedding generation')
    
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
    
    # Create configuration
    config = EmbeddingConfig(
        openai_api_key=api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output_dir,
        index_type=args.index_type,
        batch_size=args.batch_size
    )
    
    # Create embedder
    embedder = MongolianHistoryEmbedder(config)
    
    try:
        # Create embeddings and index
        print("Creating embeddings pipeline...")
        index, metadata = embedder.create_embeddings(args.dataset)
        
        # Save pipeline
        print("Saving pipeline...")
        summary = embedder.save_pipeline(index, metadata)
        
        print("\nPipeline Creation Summary:")
        print(f"  Total entries: {summary['total_entries']}")
        print(f"  Total chunks: {summary['total_chunks']}")
        print(f"  Embedding dimension: {summary['embedding_dimension']}")
        print(f"  Index type: {summary['index_type']}")
        print(f"  Average chunk length: {summary['avg_chunk_length']:.0f} characters")
        
        print(f"\nPeriod distribution:")
        for period, count in summary['period_distribution'].items():
            print(f"  {period}: {count} chunks")
        
        print(f"\nFiles created in {args.output_dir}:")
        print(f"  - {config.index_filename} (FAISS index)")
        print(f"  - {config.metadata_filename} (chunk metadata)")
        print(f"  - {config.config_filename} (pipeline configuration)")
        print(f"  - pipeline_summary.json (creation summary)")
        
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        raise


if __name__ == "__main__":
    main()