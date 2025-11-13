#!/usr/bin/env python3
"""
RAG Agent for Mongolian Historical Questions.

This module provides a complete RAG (Retrieval-Augmented Generation) system
for answering questions about Mongolian history using the embedded dataset.
"""

import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

from openai import OpenAI
from .retrieval_engine import MongolianHistoryRetriever, SearchConfig


@dataclass
class RAGConfig:
    """Configuration for the RAG agent."""
    
    # OpenAI settings
    openai_api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 1500
    
    # Retrieval settings
    retrieval_top_k: int = 5
    max_context_tokens: int = 3000
    
    # Response settings
    include_sources: bool = True
    language: str = "mongolian"  # mongolian, english, or auto


class MongolianHistoryRAG:
    """RAG agent for Mongolian historical questions."""
    
    def __init__(self, config: RAGConfig, embeddings_dir: str):
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.retriever = MongolianHistoryRetriever(embeddings_dir, config.openai_api_key)
        self.logger = logging.getLogger(__name__)
        
        # Conversation history
        self.conversation_history = []
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the RAG agent."""
        if self.config.language == "mongolian":
            return """Та Монголын түүхийн мэргэжилтэн юм. Танд өгөгдсөн түүхийн баримтуудыг ашиглан Монголын түүхийн талаарх асуултуудад нарийвчлалтай, үнэн зөв хариулт өгнө үү.

Дүрэм:
1. Зөвхөн өгөгдсөн баримтуудад үндэслэн хариулна уу
2. Баримтгүй мэдээлэл өгөхгүй байна уу  
3. Хариултаа тодорхой, ойлгомжтой байлгана уу
4. Эх сурвалжийг дурдана уу
5. Хэрэв мэдээлэл хангалтгүй бол үүнийг тодорхой хэлнэ үү

Монгол хэлээр хариулна уу."""
        
        elif self.config.language == "english":
            return """You are an expert on Mongolian history. Use the provided historical documents to answer questions about Mongolian history with accuracy and detail.

Rules:
1. Base your answers only on the provided documents
2. Do not provide information without documentary evidence
3. Keep your responses clear and understandable
4. Cite your sources
5. If information is insufficient, state this clearly

Respond in English."""
        
        else:  # auto
            return """You are an expert on Mongolian history. Use the provided historical documents to answer questions about Mongolian history with accuracy and detail.

Rules:
1. Base your answers only on the provided documents
2. Do not provide information without documentary evidence
3. Keep your responses clear and understandable
4. Cite your sources
5. If information is insufficient, state this clearly
6. Respond in the same language as the question (Mongolian or English)"""
    
    def _format_context(self, context: str, sources: List[Dict[str, Any]]) -> str:
        """Format the retrieved context for the prompt."""
        formatted_context = f"Түүхийн баримтууд:\n\n{context}"
        
        if self.config.include_sources and sources:
            formatted_context += "\n\nЭх сурвалжууд:\n"
            for i, source in enumerate(sources, 1):
                formatted_context += f"{i}. {source['title']} ({source['period']}) - {source['source']}\n"
        
        return formatted_context
    
    def _create_messages(self, question: str, context: str, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create messages for the OpenAI API."""
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        # Add conversation history
        for entry in self.conversation_history[-4:]:  # Last 4 exchanges
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["answer"]})
        
        # Add current context and question
        formatted_context = self._format_context(context, sources)
        
        user_message = f"{formatted_context}\n\nАсуулт: {question}"
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def answer_question(self, question: str, use_history: bool = True) -> Dict[str, Any]:
        """Answer a question using RAG."""
        try:
            # Retrieve relevant context
            self.logger.info(f"Retrieving context for question: {question}")
            
            search_config = SearchConfig(
                top_k=self.config.retrieval_top_k,
                rerank=True,
                deduplicate=True
            )
            
            context, sources = self.retriever.get_context_for_rag(
                question, 
                max_tokens=self.config.max_context_tokens
            )
            
            if not context:
                return {
                    "answer": "Уучлаарай, энэ асуултын хариултыг олж чадсангүй. Өөр асуулт асуугаарай.",
                    "sources": [],
                    "context_used": "",
                    "confidence": 0.0
                }
            
            # Create messages
            messages = self._create_messages(question, context, sources)
            
            # Generate response
            self.logger.info("Generating response...")
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            answer = response.choices[0].message.content
            
            # Calculate confidence based on retrieval scores
            avg_score = sum(s['score'] for s in sources) / len(sources) if sources else 0
            confidence = max(0, min(1, 1 - avg_score))  # Convert distance to confidence
            
            # Store in conversation history
            if use_history:
                self.conversation_history.append({
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "sources": sources
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "context_used": context,
                "confidence": confidence,
                "retrieval_count": len(sources)
            }
            
        except Exception as e:
            self.logger.error(f"Error answering question: {e}")
            return {
                "answer": f"Алдаа гарлаа: {str(e)}",
                "sources": [],
                "context_used": "",
                "confidence": 0.0
            }
    
    def ask_about_period(self, period: str, question: str = "") -> Dict[str, Any]:
        """Ask a question about a specific historical period."""
        if question:
            enhanced_question = f"{period} үеийн талаар: {question}"
        else:
            enhanced_question = f"{period} үеийн талаар ярина уу"
        
        return self.answer_question(enhanced_question)
    
    def ask_about_person(self, person: str, question: str = "") -> Dict[str, Any]:
        """Ask a question about a specific historical person."""
        if question:
            enhanced_question = f"{person}-ын талаар: {question}"
        else:
            enhanced_question = f"{person}-ын талаар ярина уу"
        
        return self.answer_question(enhanced_question)
    
    def ask_about_event(self, event: str, question: str = "") -> Dict[str, Any]:
        """Ask a question about a specific historical event."""
        if question:
            enhanced_question = f"{event} үйл явдлын талаар: {question}"
        else:
            enhanced_question = f"{event} үйл явдлын талаар ярина уу"
        
        return self.answer_question(enhanced_question)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
    
    def export_conversation(self, filepath: str):
        """Export conversation history to a file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        retrieval_stats = self.retriever.get_statistics()
        
        return {
            "retrieval_system": retrieval_stats,
            "conversation_length": len(self.conversation_history),
            "model": self.config.model,
            "language": self.config.language,
            "last_updated": datetime.now().isoformat()
        }


def create_interactive_session(embeddings_dir: str, api_key: str) -> MongolianHistoryRAG:
    """Create an interactive RAG session."""
    config = RAGConfig(
        openai_api_key=api_key,
        temperature=0.3,
        retrieval_top_k=5,
        language="auto"
    )
    
    return MongolianHistoryRAG(config, embeddings_dir)


def main():
    """Interactive demo of the RAG agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Mongolian History RAG Agent")
    parser.add_argument('--embeddings-dir', default='data/embeddings',
                       help='Directory containing embeddings and index')
    parser.add_argument('--language', choices=['mongolian', 'english', 'auto'], 
                       default='auto', help='Response language')
    parser.add_argument('--model', default='gpt-4o-mini',
                       help='OpenAI model to use')
    parser.add_argument('--export', help='Export conversation to file')
    
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
    
    # Create RAG agent
    config = RAGConfig(
        openai_api_key=api_key,
        model=args.model,
        language=args.language,
        temperature=0.3
    )
    
    rag = MongolianHistoryRAG(config, args.embeddings_dir)
    
    print("Монголын түүхийн RAG систем")
    print("=" * 40)
    print("Монголын түүхийн талаар асуулт асуугаарай.")
    print("Гарахын тулд 'exit' гэж бичнэ үү.")
    print("Түүхийг цэвэрлэхийн тулд 'clear' гэж бичнэ үү.")
    print()
    
    # Show system stats
    stats = rag.get_system_stats()
    print(f"Системийн мэдээлэл:")
    print(f"  Нийт баримт: {stats['retrieval_system']['total_entries']}")
    print(f"  Нийт хэсэг: {stats['retrieval_system']['total_chunks']}")
    print(f"  Модель: {stats['model']}")
    print()
    
    try:
        while True:
            question = input("Асуулт: ").strip()
            
            if question.lower() in ['exit', 'quit', 'гарах']:
                break
            elif question.lower() in ['clear', 'цэвэрлэх']:
                rag.clear_history()
                print("Түүх цэвэрлэгдлээ.\n")
                continue
            elif not question:
                continue
            
            print("Хариулт бэлтгэж байна...")
            
            # Get answer
            result = rag.answer_question(question)
            
            print(f"\nХариулт:")
            print(result['answer'])
            
            if result['sources'] and config.include_sources:
                print(f"\nЭх сурвалж ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. {source['title'][:60]}... ({source['period']})")
            
            print(f"\nИтгэмжлэл: {result['confidence']:.2f}")
            print("-" * 60)
    
    except KeyboardInterrupt:
        print("\n\nСистемээс гарч байна...")
    
    # Export conversation if requested
    if args.export and rag.conversation_history:
        rag.export_conversation(args.export)
        print(f"Харилцаа {args.export} файлд хадгалагдлаа.")


if __name__ == "__main__":
    main()