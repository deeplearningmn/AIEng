# Mongolian History RAG System

Semantic search-powered Q&A system for Mongolian history using OpenAI embeddings and GPT.

## Features

- 🔍 **Semantic Search**: text-embedding-3-large for accurate retrieval
- 🎯 **Hybrid Matching**: Combines semantic similarity + keyword boosting
- 📚 **49 Documents**: Comprehensive coverage from ancient to modern history
- 🇲🇳 **Bilingual**: Supports Mongolian (Cyrillic) and English
- 💬 **Interactive**: Suggested questions + custom input

## Coverage

- **Ancient Period**: Хүннү, Сяньби, Түрэг, Уйгур (BCE-900)
- **Medieval Period**: Их Монгол Улс, Юань, Зүүнгар (1206-1757)
- **Modern Period**: Богд хаан, 1921 Revolution, 1990 Democracy (1911-1990)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the system
python rag_semantic.py
```

## Usage

1. Enter your OpenAI API key
2. Choose model (gpt-4o-mini recommended)
3. Select from 10 suggested questions or enter custom question
4. Get accurate answers with source citations

## Example Questions

1. Чингис хаан хэзээ төрсөн бэ?
2. Их Монгол Улс хэзээ байгуулагдсан бэ?
3. Хүннү гүрний анхны шаньюй хэн байсан бэ?
4. 1921 оны хувьсгалын үр дүн юу байсан бэ?
5. Монголын ардчилсан хувьсгал хэзээ болсон бэ?

## Dataset

- **49 documents** total
- **28 core documents**: Filtered Mongolian history + Q&A supplements
- **21 Wikipedia articles**: Ancient/medieval period coverage
- **Embeddings cached**: Fast loading after first run

## Performance

- ✅ 100% accuracy on modern history (1900s-1990s)
- ✅ 40%+ accuracy on ancient history (limited by sources)
- ✅ High-quality answers with source attribution
- ⚡ ~2-3 seconds per query

## Files

- `rag_semantic.py` - Main RAG system (run this)
- `setup.py` - API key configuration
- `scrape_missing_articles.py` - Add more Wikipedia articles
- `data/mongolian_history_complete.jsonl` - Full dataset
- `data/embeddings_complete_cache.pkl` - Cached embeddings

## Adding More Sources

```bash
# Scrape additional Wikipedia articles
python scrape_missing_articles.py

# Embeddings will regenerate automatically
```

## Tech Stack

- **Embeddings**: OpenAI text-embedding-3-large (3072 dimensions)
- **Generation**: GPT-4o-mini / GPT-4o
- **Search**: Cosine similarity + keyword boosting
- **Language**: Python 3.9+

## License

MIT
