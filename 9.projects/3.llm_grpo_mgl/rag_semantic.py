#!/usr/bin/env python3
"""
Semantic RAG System with OpenAI Embeddings
Uses vector embeddings for accurate semantic search and GPT for generation
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from getpass import getpass


class SemanticRAG:
    """RAG system with semantic search using embeddings."""
    
    def __init__(
        self,
        dataset_path: str = "data/mongolian_history_complete.jsonl",
        embeddings_path: str = "data/embeddings_complete_cache.pkl",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large"
    ):
        self.dataset_path = Path(dataset_path)
        self.embeddings_path = Path(embeddings_path)
        self.model = model
        self.embedding_model = embedding_model
        self.documents = []
        self.embeddings = None
        
        # Setup OpenAI
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key parameter.")
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        self.load_dataset()
        self.load_or_create_embeddings()
    
    def load_dataset(self):
        """Load the filtered dataset."""
        print(f"📂 Loading dataset from {self.dataset_path}...")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc)
        
        print(f"✅ Loaded {len(self.documents)} documents")
    
    def create_embeddings(self) -> np.ndarray:
        """Create embeddings for all documents."""
        print("🔄 Creating embeddings for documents...")
        print("⏳ This may take a minute...")
        
        texts = []
        for doc in self.documents:
            title = doc.get('title', '')
            text = doc.get('text', '')
            # Limit to ~2000 tokens (8000 chars) for text-embedding-3-large
            max_chars = 8000
            if len(text) > max_chars:
                text = text[:max_chars]
            combined = f"{title}\n\n{text}"
            texts.append(combined)
        
        # Create embeddings one at a time
        embeddings = []
        
        for i, text in enumerate(texts):
            print(f"  Processing document {i+1}/{len(texts)}...")
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=[text]
            )
            embeddings.append(response.data[0].embedding)
        
        embeddings_array = np.array(embeddings)
        
        # Save embeddings
        print(f"💾 Saving embeddings to {self.embeddings_path}...")
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(embeddings_array, f)
        
        print("✅ Embeddings created and saved")
        return embeddings_array
    
    def load_or_create_embeddings(self):
        """Load existing embeddings or create new ones."""
        if self.embeddings_path.exists():
            print(f"📂 Loading embeddings from {self.embeddings_path}...")
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"✅ Loaded embeddings for {len(self.embeddings)} documents")
        else:
            print("⚠️  No cached embeddings found")
            self.embeddings = self.create_embeddings()
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def keyword_boost(self, query: str, doc: Dict[str, Any]) -> float:
        """Calculate keyword match boost score."""
        query_lower = query.lower()
        text_lower = doc.get('text', '').lower()
        title_lower = doc.get('title', '').lower()
        
        score = 0.0
        
        # Extract key terms from query
        query_words = set(query_lower.split())
        
        # Strong boost for title matches with key terms
        key_terms = [w for w in query_words if len(w) > 3]
        title_matches = sum(1 for term in key_terms if term in title_lower)
        if title_matches > 0:
            score += 0.4 * title_matches  # Much stronger boost
        
        # Boost for exact phrase match in text
        if query_lower in text_lower:
            score += 0.15
        
        # Boost for multiple word matches
        text_words = set(text_lower.split())
        overlap = len(query_words & text_words)
        if overlap > 2:
            score += 0.05 * (overlap - 2)
        
        return min(score, 0.6)  # Higher cap
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Hybrid search: semantic similarity + keyword boosting."""
        # Create query embedding
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=[query]
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Calculate similarities with keyword boost
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            semantic_sim = self.cosine_similarity(query_embedding, doc_embedding)
            keyword_boost = self.keyword_boost(query, self.documents[i])
            
            # Combine semantic and keyword scores
            final_score = semantic_sim + keyword_boost
            
            similarities.append({
                'index': i,
                'similarity': final_score,
                'semantic_score': semantic_sim,
                'keyword_boost': keyword_boost,
                'document': self.documents[i]
            })
        
        # Sort by combined score
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = []
        for item in similarities[:top_k]:
            results.append({
                'document': item['document'],
                'similarity': item['similarity']
            })
        
        return results
    
    def generate_answer(
        self,
        question: str,
        language: str = "mongolian",
        temperature: float = 0.7,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """Generate answer using semantic search and GPT."""
        
        print(f"\n❓ Question: {question}")
        print("=" * 60)
        
        # Semantic search
        print("🔍 Performing semantic search...")
        search_results = self.semantic_search(question, top_k=top_k)
        
        if not search_results:
            return {
                'question': question,
                'answer': 'Уучлаарай, энэ асуултад хариулах мэдээлэл олдсонгүй.',
                'sources': [],
                'similarities': [],
                'context_used': False
            }
        
        # Prepare context
        context_parts = []
        sources = []
        similarities = []
        
        for i, result in enumerate(search_results, 1):
            doc = result['document']
            similarity = result['similarity']
            
            text = doc.get('text', '')
            source = doc.get('source', 'Unknown')
            chapter = doc.get('chapter', '')
            period = doc.get('period', '')
            title = doc.get('title', '')
            
            context_parts.append(f"[Эх сурвалж {i}] (Similarity: {similarity:.3f})\n{text[:800]}")
            sources.append({
                'source': source,
                'title': title,
                'chapter': chapter,
                'period': period
            })
            similarities.append(similarity)
        
        context = "\n\n".join(context_parts)
        
        print(f"✅ Found {len(search_results)} relevant sources")
        print(f"   Best match similarity: {similarities[0]:.3f}")
        print("🤖 Generating answer with GPT...")
        
        # Create prompt based on language
        if language.lower() == "mongolian":
            system_prompt = """Та Монголын түүхийн мэргэжилтэн юм. Өгөгдсөн эх сурвалжийн мэдээлэлд үндэслэн асуултад хариулна уу.

Дүрэм:
1. Зөвхөн өгөгдсөн эх сурвалжийн мэдээлэлийг ашиглана
2. Монгол хэлээр тодорхой, ойлгомжтой хариулна
3. Мэдэхгүй бол "Өгөгдсөн эх сурвалжид энэ тухай мэдээлэл байхгүй байна" гэж хэлнэ
4. Хариултаа байгалийн, хүнлэг яриагаар өгнө
5. Эх сурвалжийн мэдээллийг нэмж тайлбарлаж болно"""
            
            user_prompt = f"""Эх сурвалжийн мэдээлэл:

{context}

Асуулт: {question}

Дээрх эх сурвалжид үндэслэн асуултад монгол хэлээр хариулна уу:"""
        
        else:  # English
            system_prompt = """You are a Mongolian history expert. Answer questions based on the provided source materials.

Rules:
1. Only use information from the provided sources
2. Answer clearly and naturally
3. If you don't know, say "The provided sources don't contain this information"
4. Write in a conversational, human-like tone
5. You can elaborate on the source information"""
            
            user_prompt = f"""Source materials:

{context}

Question: {question}

Based on the sources above, please answer the question:"""
        
        # Generate answer with GPT
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Display results
            print("\n" + "=" * 60)
            print("💬 ANSWER:")
            print("=" * 60)
            print(answer)
            print("\n" + "=" * 60)
            print("📚 SOURCES (with similarity scores):")
            print("=" * 60)
            
            for i, (source, sim) in enumerate(zip(sources, similarities), 1):
                print(f"\n{i}. {source['source']} (Similarity: {sim:.3f})")
                if source['title']:
                    print(f"   Title: {source['title']}")
                if source['period']:
                    print(f"   Period: {source['period']}")
                if source['chapter']:
                    print(f"   Chapter: {source['chapter']}")
            
            return {
                'question': question,
                'answer': answer,
                'sources': sources,
                'similarities': similarities,
                'context_used': True,
                'model': self.model
            }
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return {
                'question': question,
                'answer': f'Error: {str(e)}',
                'sources': sources,
                'similarities': similarities,
                'context_used': False
            }


def main():
    """Interactive Semantic RAG with suggested questions."""
    print("🇲🇳 Mongolian History Semantic RAG")
    print("=" * 60)
    print("Semantic search with 49 historical documents")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("\n🔑 OpenAI API key not found in environment")
        api_key = getpass("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ API key required")
            return
    
    # Test API key
    print("\n🧪 Testing API key...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.models.list()
        print("✅ API key valid")
    except Exception as e:
        print(f"❌ Invalid API key: {e}")
        return
    
    # Choose model
    print("\n🤖 Choose model:")
    print("1. gpt-4o-mini (faster, cheaper)")
    print("2. gpt-4o (better quality)")
    
    choice = input("\nChoice (1-2, default=1): ").strip() or "1"
    model = "gpt-4o-mini" if choice == "1" else "gpt-4o"
    print(f"✅ Using: {model}")
    
    # Initialize RAG
    try:
        rag = SemanticRAG(api_key=api_key, model=model)
    except Exception as e:
        print(f"❌ Error initializing RAG: {e}")
        return
    
    # Suggested questions
    suggested_questions = [
        "Чингис хаан хэзээ төрсөн бэ?",
        "Их Монгол Улс хэзээ байгуулагдсан бэ?",
        "Өгэдэй хааны тухай хэлнэ үү?",
        "Хүннү гүрний анхны шаньюй хэн байсан бэ?",
        "Модун шаньюй Хүннү гүрнийг хэрхэн төвлөрсөн удирдлагатай болгосон бэ?",
        "1921 оны хувьсгалын үр дүн юу байсан бэ?",
        "Богд хаан хэн байсан бэ?",
        "Монголын ардчилсан хувьсгал хэзээ болсон бэ?",
        "Юань улсын тухай хэлнэ үү?",
        "Зүүнгарын тухай хэлнэ үү?"
    ]
    
    # Interactive mode
    print("\n💬 Interactive Mode")
    print("=" * 60)
    
    while True:
        try:
            print("\n📋 Suggested Questions:")
            for i, q in enumerate(suggested_questions, 1):
                print(f"  {i}. {q}")
            print(f"  0. Custom question")
            
            choice = input("\n❓ Choose (1-10) or 0 for custom: ").strip()
            
            if not choice:
                continue
            
            if choice.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Get question
            if choice == '0':
                question = input("Enter your question: ").strip()
                if not question:
                    continue
            elif choice.isdigit() and 1 <= int(choice) <= len(suggested_questions):
                question = suggested_questions[int(choice) - 1]
                print(f"\n❓ {question}")
            else:
                print("❌ Invalid choice")
                continue
            
            # Detect language
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in question)
            language = "mongolian" if has_cyrillic else "english"
            
            result = rag.generate_answer(question, language=language)
            
            print("\n" + "-" * 60)
            input("\nPress Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
