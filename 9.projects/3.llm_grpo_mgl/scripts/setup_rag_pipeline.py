#!/usr/bin/env python3
"""
Setup script for the complete Mongolian History RAG pipeline.

This script:
1. Installs required dependencies
2. Creates embeddings from the unified dataset
3. Builds FAISS index
4. Tests the retrieval system
5. Provides usage examples
"""

import os
import sys
import subprocess
from pathlib import Path
import json
import logging


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def check_requirements():
    """Check if required packages are installed."""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'openai',
        'faiss-cpu',
        'numpy',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {missing_packages}")
        return False
    
    logger.info("All required packages are installed")
    return True


def install_requirements():
    """Install required packages."""
    logger = logging.getLogger(__name__)
    
    logger.info("Installing RAG requirements...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements_rag.txt'
        ], check=True)
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False


def check_api_key():
    """Check if OpenAI API key is available."""
    logger = logging.getLogger(__name__)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not found")
        logger.info("Please set your OpenAI API key:")
        logger.info("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    logger.info("OpenAI API key found")
    return True


def check_dataset():
    """Check if the unified dataset exists."""
    logger = logging.getLogger(__name__)
    
    # Use filtered dataset (excludes Secret History)
    dataset_path = Path('data/mongolian_history_unified_filtered.jsonl')
    
    # Fallback to original if filtered doesn't exist
    if not dataset_path.exists():
        dataset_path = Path('data/mongolian_history_unified.jsonl')
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please run the data cleaning script first:")
        logger.info("python scripts/clean_and_merge_json.py")
        return False
    
    # Check dataset size
    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = sum(1 for line in f if line.strip())
    
    logger.info(f"Dataset found with {lines} entries")
    return True


def create_embeddings():
    """Create embeddings and FAISS index."""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating embeddings and FAISS index...")
    
    try:
        # Import here to avoid import errors if packages not installed
        from mongolian_rag.embedding_pipeline import MongolianHistoryEmbedder, EmbeddingConfig
        
        # Create configuration
        config = EmbeddingConfig(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            chunk_size=1000,
            chunk_overlap=200,
            output_dir='data/embeddings',
            index_type='IVF',
            batch_size=50
        )
        
        # Create embedder
        embedder = MongolianHistoryEmbedder(config)
        
        # Create embeddings
        # Use filtered dataset
        dataset_file = 'data/mongolian_history_unified_filtered.jsonl'
        if not Path(dataset_file).exists():
            dataset_file = 'data/mongolian_history_unified.jsonl'
        
        index, metadata = embedder.create_embeddings(dataset_file)
        
        # Save pipeline
        summary = embedder.save_pipeline(index, metadata)
        
        logger.info("Embeddings created successfully")
        logger.info(f"Total entries: {summary['total_entries']}")
        logger.info(f"Total chunks: {summary['total_chunks']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        return False


def test_retrieval():
    """Test the retrieval system."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing retrieval system...")
    
    try:
        from mongolian_rag.retrieval_engine import MongolianHistoryRetriever, SearchConfig
        
        # Create retriever
        retriever = MongolianHistoryRetriever('data/embeddings', os.getenv('OPENAI_API_KEY'))
        
        # Test search
        test_query = "Чингис хаан"
        config = SearchConfig(top_k=3)
        results = retriever.search(test_query, config)
        
        logger.info(f"Test search for '{test_query}' returned {len(results)} results")
        
        if results:
            logger.info(f"Top result score: {results[0].score:.4f}")
            logger.info(f"Top result period: {results[0].metadata.get('period', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to test retrieval: {e}")
        return False


def test_rag_agent():
    """Test the RAG agent."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing RAG agent...")
    
    try:
        from mongolian_rag.rag_agent import MongolianHistoryRAG, RAGConfig
        
        # Create configuration
        config = RAGConfig(
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model='gpt-4o-mini',
            temperature=0.3,
            language='mongolian'
        )
        
        # Create RAG agent
        rag = MongolianHistoryRAG(config, 'data/embeddings')
        
        # Test question
        test_question = "Чингис хаан хэзээ төрсөн бэ?"
        result = rag.answer_question(test_question, use_history=False)
        
        logger.info(f"Test question: {test_question}")
        logger.info(f"Answer length: {len(result['answer'])} characters")
        logger.info(f"Sources found: {len(result['sources'])}")
        logger.info(f"Confidence: {result['confidence']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to test RAG agent: {e}")
        return False


def create_usage_examples():
    """Create usage examples and documentation."""
    logger = logging.getLogger(__name__)
    
    examples_dir = Path('examples')
    examples_dir.mkdir(exist_ok=True)
    
    # Create basic usage example
    basic_example = '''#!/usr/bin/env python3
"""
Basic usage example for Mongolian History RAG system.
"""

import os
from mongolian_rag import create_interactive_session

def main():
    # Set your OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        return
    
    # Create RAG session
    rag = create_interactive_session('data/embeddings', api_key)
    
    # Ask a question
    question = "Монголын нууц товчооны талаар ярина уу?"
    result = rag.answer_question(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")

if __name__ == "__main__":
    main()
'''
    
    with open(examples_dir / 'basic_usage.py', 'w', encoding='utf-8') as f:
        f.write(basic_example)
    
    # Create advanced example
    advanced_example = '''#!/usr/bin/env python3
"""
Advanced usage example with custom configuration.
"""

import os
from mongolian_rag import MongolianHistoryRAG, RAGConfig

def main():
    # Custom configuration
    config = RAGConfig(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4o-mini',
        temperature=0.2,
        retrieval_top_k=7,
        language='auto'
    )
    
    # Create RAG agent
    rag = MongolianHistoryRAG(config, 'data/embeddings')
    
    # Ask about specific periods
    questions = [
        "XIII зууны Монголын түүхийн талаар ярина уу?",
        "1921 оны хувьсгалын талаар мэдээлэл өгнө үү?",
        "Богд хааны үеийн Монголын талаар асуумаар байна"
    ]
    
    for question in questions:
        print(f"\\nQuestion: {question}")
        result = rag.answer_question(question)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print("-" * 50)
    
    # Export conversation
    rag.export_conversation('conversation_history.json')
    print("\\nConversation exported to conversation_history.json")

if __name__ == "__main__":
    main()
'''
    
    with open(examples_dir / 'advanced_usage.py', 'w', encoding='utf-8') as f:
        f.write(advanced_example)
    
    logger.info(f"Usage examples created in {examples_dir}")


def main():
    """Main setup function."""
    logger = setup_logging()
    
    print("Mongolian History RAG Pipeline Setup")
    print("=" * 40)
    
    # Step 1: Check and install requirements
    print("\\n1. Checking requirements...")
    if not check_requirements():
        print("Installing missing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements")
            return False
    print("✅ Requirements satisfied")
    
    # Step 2: Check API key
    print("\\n2. Checking OpenAI API key...")
    if not check_api_key():
        print("❌ OpenAI API key not found")
        return False
    print("✅ API key found")
    
    # Step 3: Check dataset
    print("\\n3. Checking dataset...")
    if not check_dataset():
        print("❌ Dataset not found")
        return False
    print("✅ Dataset ready")
    
    # Step 4: Create embeddings
    print("\\n4. Creating embeddings...")
    embeddings_exist = Path('data/embeddings/mongolian_history.faiss').exists()
    
    if embeddings_exist:
        print("✅ Embeddings already exist")
    else:
        if not create_embeddings():
            print("❌ Failed to create embeddings")
            return False
        print("✅ Embeddings created")
    
    # Step 5: Test retrieval
    print("\\n5. Testing retrieval system...")
    if not test_retrieval():
        print("❌ Retrieval test failed")
        return False
    print("✅ Retrieval system working")
    
    # Step 6: Test RAG agent
    print("\\n6. Testing RAG agent...")
    if not test_rag_agent():
        print("❌ RAG agent test failed")
        return False
    print("✅ RAG agent working")
    
    # Step 7: Create examples
    print("\\n7. Creating usage examples...")
    create_usage_examples()
    print("✅ Examples created")
    
    print("\\n" + "=" * 40)
    print("🎉 RAG Pipeline Setup Complete!")
    print("\\nUsage:")
    print("  Interactive mode: python -m mongolian_rag.rag_agent")
    print("  Basic example: python examples/basic_usage.py")
    print("  Advanced example: python examples/advanced_usage.py")
    print("\\nFiles created:")
    print("  - data/embeddings/ (FAISS index and metadata)")
    print("  - examples/ (usage examples)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)