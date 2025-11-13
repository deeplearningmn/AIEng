#!/usr/bin/env python3
"""
GRPO Dataset Builder for Mongolian Historical Data

Generates a GRPO (Generative Reinforcement Preference Optimization) dataset from 
RAG-generated Mongolian Q&A pairs. Each record contains a user question (prompt) 
and two model responses: a "chosen" (better) and a "rejected" (worse) answer 
for preference-based fine-tuning.

Usage:
    python scripts/build_grpo_dataset.py
    python scripts/build_grpo_dataset.py --source data/custom_corpus.jsonl
    python scripts/build_grpo_dataset.py --pairs-per-topic 20 --output data/custom_grpo.jsonl
"""

import json
import re
import os
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from tqdm import tqdm

try:
    from openai import OpenAI
    from openai import APIConnectionError, RateLimitError, AuthenticationError
except ImportError:
    print("Error: OpenAI library not installed. Run: pip install openai")
    exit(1)


@dataclass
class GRPOStats:
    """Statistics for GRPO dataset generation."""
    total_prompts_generated: int = 0
    valid_pairs: int = 0
    rejected_pairs: int = 0
    api_failures: int = 0
    json_parse_errors: int = 0
    validation_failures: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    avg_chosen_length: float = 0.0
    avg_rejected_length: float = 0.0
    avg_prompt_length: float = 0.0
    mongolian_purity: float = 0.0
    processing_time: float = 0.0
    
    def calculate_averages(self, pairs: List[Dict[str, Any]]):
        """Calculate average lengths and purity from generated pairs."""
        if not pairs:
            return
        
        chosen_lengths = []
        rejected_lengths = []
        prompt_lengths = []
        mongolian_chars = 0
        total_chars = 0
        
        for pair in pairs:
            chosen_lengths.append(len(pair['chosen'].split()))
            rejected_lengths.append(len(pair['rejected'].split()))
            prompt_lengths.append(len(pair['prompt'].split()))
            
            # Count Mongolian characters for purity
            text = f"{pair['prompt']} {pair['chosen']} {pair['rejected']}"
            mongolian_chars += len(re.findall(r'[А-ЯӨҮа-яөү]', text))
            total_chars += len(re.findall(r'[А-ЯӨҮа-яөүA-Za-z]', text))
        
        self.avg_chosen_length = sum(chosen_lengths) / len(chosen_lengths)
        self.avg_rejected_length = sum(rejected_lengths) / len(rejected_lengths)
        self.avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
        
        if total_chars > 0:
            self.mongolian_purity = (mongolian_chars / total_chars) * 100


class QuestionGenerator:
    """Generate questions from historical content."""
    
    def __init__(self):
        """Initialize question generator with templates."""
        self.question_templates = [
            "{topic} хэзээ болсон бэ?",
            "{topic}-ын үндсэн шалтгаан юу байсан бэ?",
            "{topic}-ын үр дүн нь юу байсан бэ?",
            "{topic} яагаад чухал байсан бэ?",
            "{topic}-д хэн оролцсон бэ?",
            "{topic}-ын ач холбогдол юунд оршдог вэ?",
            "{topic} хэрхэн өрнөсөн бэ?",
            "{topic}-ын талаар дэлгэрэнгүй ярина уу?",
            "{topic} Монголын түүхэнд ямар нөлөө үзүүлсэн бэ?",
            "{topic}-тай холбоотой гол үйл явдлууд юу вэ?",
            "{topic}-ын тухай юу мэдэх хэрэгтэй вэ?",
            "{topic} хэрхэн эхэлсэн бэ?",
            "{topic}-ын дараа юу болсон бэ?",
            "{topic}-ын онцлог нь юу вэ?",
            "{topic} яагаад тэр үед болсон бэ?"
        ]
    
    def extract_topics_from_content(self, content: str) -> List[str]:
        """Extract potential topics from historical content."""
        # Common historical terms and patterns
        topic_patterns = [
            r'(\d{4})\s*оны?\s+([^.!?]+)',  # Year-based events
            r'([А-ЯӨҮ][а-яөү\s]+(?:хувьсгал|дайн|хаан|улс|гүрэн))',  # Historical terms
            r'([А-ЯӨҮ][а-яөү\s]+(?:үе|цаг|зуун))',  # Time periods
            r'([А-ЯӨҮ][а-яөү\s]+(?:бодлого|тогтолцоо|засаглал))',  # Political terms
        ]
        
        topics = []
        for pattern in topic_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    topic = ' '.join(str(m) for m in match if m).strip()
                else:
                    topic = str(match).strip()
                
                # Clean and validate topic
                topic = re.sub(r'\s+', ' ', topic)
                if len(topic) > 10 and len(topic) < 100:
                    topics.append(topic)
        
        # Remove duplicates and return
        return list(set(topics))
    
    def generate_questions_for_topic(self, topic: str, count: int = 3) -> List[str]:
        """Generate questions for a specific topic."""
        questions = []
        templates = random.sample(self.question_templates, min(count, len(self.question_templates)))
        
        for template in templates:
            question = template.format(topic=topic)
            questions.append(question)
        
        return questions
    
    def generate_questions_from_content(self, content: str, questions_per_topic: int = 2) -> List[Tuple[str, str]]:
        """Generate questions from content with associated context."""
        topics = self.extract_topics_from_content(content)
        question_pairs = []
        
        for topic in topics[:10]:  # Limit to 10 topics per content
            questions = self.generate_questions_for_topic(topic, questions_per_topic)
            for question in questions:
                question_pairs.append((question, content))
        
        return question_pairs


class GRPOGenerator:
    """Generate GRPO preference pairs using OpenAI."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 temperature: float = 0.4, max_tokens: int = 900, max_retries: int = 2):
        """Initialize GRPO generator."""
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        self.system_prompt = """You are an expert in Mongolian history and language. Your task is to generate two different quality responses to historical questions.

Given a Mongolian historical question and factual context, create:
1. A CHOSEN response: Strong, factual, coherent, and comprehensive
2. A REJECTED response: Weak, vague, incomplete, or slightly inaccurate

Requirements:
- Both responses must be in Mongolian
- CHOSEN should be 80-150 words, well-structured
- REJECTED should be 40-80 words, less informative
- Maintain historical accuracy in CHOSEN
- Make REJECTED obviously inferior but not completely wrong
- Use proper Mongolian grammar and vocabulary

Respond ONLY with valid JSON in this exact format:
{
  "prompt": "the original question",
  "chosen": "strong factual answer in Mongolian",
  "rejected": "weak or vague answer in Mongolian"
}"""
    
    def generate_grpo_pair(self, question: str, context: str) -> Tuple[Optional[Dict[str, str]], bool, Dict[str, Any]]:
        """Generate a GRPO preference pair."""
        metadata = {'tokens_used': 0, 'api_calls': 0, 'error': None}
        
        user_prompt = f"""Question: {question}

Historical Context:
{context[:800]}  

Generate two responses as specified in the system prompt."""
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                content = response.choices[0].message.content.strip()
                metadata['api_calls'] = attempt + 1
                
                if hasattr(response, 'usage') and response.usage:
                    metadata['tokens_used'] = response.usage.total_tokens
                
                # Parse JSON response
                try:
                    grpo_pair = json.loads(content)
                    
                    # Validate required fields
                    required_fields = ['prompt', 'chosen', 'rejected']
                    if all(field in grpo_pair for field in required_fields):
                        # Ensure responses are different
                        if grpo_pair['chosen'] != grpo_pair['rejected']:
                            return grpo_pair, True, metadata
                        else:
                            metadata['error'] = f"Chosen and rejected responses are identical on attempt {attempt + 1}"
                    else:
                        metadata['error'] = f"Missing required fields on attempt {attempt + 1}"
                
                except json.JSONDecodeError as e:
                    metadata['error'] = f"JSON parse error on attempt {attempt + 1}: {e}"
                    continue
                    
            except RateLimitError as e:
                metadata['error'] = f"Rate limit error on attempt {attempt + 1}: {e}"
                if attempt < self.max_retries:
                    delay = (2 ** attempt) * 60
                    time.sleep(delay)
                continue
                
            except APIConnectionError as e:
                metadata['error'] = f"Connection error on attempt {attempt + 1}: {e}"
                continue
                
            except AuthenticationError as e:
                metadata['error'] = f"Authentication error: {e}"
                break
                
            except Exception as e:
                metadata['error'] = f"Unexpected error on attempt {attempt + 1}: {e}"
                continue
        
        return None, False, metadata


class GRPOValidator:
    """Validate GRPO pairs for quality and format."""
    
    def __init__(self, min_length: int = 100):
        """Initialize validator."""
        self.min_length = min_length
        self.mongolian_pattern = re.compile(r'[А-ЯӨҮа-яөү]')
        self.english_pattern = re.compile(r'[A-Za-z]')
    
    def validate_pair(self, pair: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate a GRPO pair."""
        errors = []
        
        # Check required fields
        required_fields = ['prompt', 'chosen', 'rejected']
        for field in required_fields:
            if field not in pair or not pair[field]:
                errors.append(f"Missing or empty field: {field}")
        
        if errors:
            return False, errors
        
        # Check minimum length
        for field in required_fields:
            if len(pair[field]) < self.min_length:
                errors.append(f"Field '{field}' too short: {len(pair[field])} chars (min: {self.min_length})")
        
        # Check Mongolian content
        for field in required_fields:
            text = pair[field]
            mongolian_chars = len(self.mongolian_pattern.findall(text))
            english_chars = len(self.english_pattern.findall(text))
            total_alpha = mongolian_chars + english_chars
            
            if total_alpha > 0:
                mongolian_ratio = mongolian_chars / total_alpha
                if mongolian_ratio < 0.8:  # 80% Mongolian threshold
                    errors.append(f"Field '{field}' has low Mongolian purity: {mongolian_ratio:.1%}")
        
        # Check that chosen != rejected
        if pair['chosen'] == pair['rejected']:
            errors.append("Chosen and rejected responses are identical")
        
        return len(errors) == 0, errors


class GRPODatasetBuilder:
    """Main class for building GRPO datasets."""
    
    def __init__(self, api_key: str, log_file: str = "data/grpo_invalid.log"):
        """Initialize GRPO dataset builder."""
        self.question_generator = QuestionGenerator()
        self.grpo_generator = GRPOGenerator(api_key)
        self.validator = GRPOValidator()
        self.log_file = Path(log_file)
        self.invalid_entries = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def load_source_data(self, source_files: List[Path]) -> List[Dict[str, Any]]:
        """Load source data from files."""
        all_records = []
        
        for file_path in source_files:
            if not file_path.exists():
                self.logger.warning(f"Source file not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix == '.jsonl':
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if line:
                                try:
                                    record = json.loads(line)
                                    all_records.append(record)
                                except json.JSONDecodeError as e:
                                    self.logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {e}")
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_records.extend(data)
                        elif isinstance(data, dict):
                            all_records.append(data)
                
                self.logger.info(f"Loaded {len(all_records)} records from {file_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
        
        return all_records
    
    def load_rag_qa_log(self, rag_log_path: Path) -> List[Tuple[str, str]]:
        """Load existing RAG Q&A pairs if available."""
        qa_pairs = []
        
        if not rag_log_path.exists():
            self.logger.info(f"RAG Q&A log not found: {rag_log_path}")
            return qa_pairs
        
        try:
            with open(rag_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            if 'question' in entry and 'answer' in entry:
                                qa_pairs.append((entry['question'], entry['answer']))
                        except json.JSONDecodeError:
                            continue
            
            self.logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from RAG log")
            
        except Exception as e:
            self.logger.error(f"Failed to load RAG Q&A log: {e}")
        
        return qa_pairs
    
    def generate_questions_from_corpus(self, records: List[Dict[str, Any]], 
                                     pairs_per_topic: int = 12) -> List[Tuple[str, str]]:
        """Generate questions from corpus content."""
        question_pairs = []
        
        for record in records:
            # Extract text content
            content = ""
            for field in ['text', 'content', 'chosen']:
                if field in record and record[field]:
                    content = str(record[field])
                    break
            
            if not content or len(content) < 200:
                continue
            
            # Generate questions for this content
            pairs = self.question_generator.generate_questions_from_content(
                content, pairs_per_topic // 5  # Distribute across topics
            )
            question_pairs.extend(pairs)
        
        # Shuffle and limit
        random.shuffle(question_pairs)
        return question_pairs[:min(pairs_per_topic * 10, len(question_pairs))]
    
    def build_grpo_dataset(self, source_files: List[Path], output_path: Path,
                          rag_log_path: Optional[Path] = None,
                          pairs_per_topic: int = 15) -> GRPOStats:
        """Build complete GRPO dataset."""
        stats = GRPOStats()
        start_time = time.time()
        
        self.logger.info("Starting GRPO dataset generation")
        self.logger.info(f"Source files: {[str(f) for f in source_files]}")
        self.logger.info(f"Output: {output_path}")
        
        # Load source data
        source_records = self.load_source_data(source_files)
        if not source_records:
            self.logger.error("No source data loaded")
            return stats
        
        # Load existing RAG Q&A pairs
        qa_pairs = []
        if rag_log_path:
            qa_pairs = self.load_rag_qa_log(rag_log_path)
        
        # Generate additional questions if needed
        if len(qa_pairs) < pairs_per_topic:
            self.logger.info("Generating questions from corpus content")
            generated_pairs = self.generate_questions_from_corpus(
                source_records, pairs_per_topic - len(qa_pairs)
            )
            qa_pairs.extend(generated_pairs)
        
        self.logger.info(f"Total question-context pairs: {len(qa_pairs)}")
        stats.total_prompts_generated = len(qa_pairs)
        
        # Generate GRPO pairs
        valid_pairs = []
        
        for i, (question, context) in enumerate(tqdm(qa_pairs, desc="Generating GRPO pairs")):
            grpo_pair, success, metadata = self.grpo_generator.generate_grpo_pair(question, context)
            
            # Update statistics
            stats.total_api_calls += metadata.get('api_calls', 0)
            stats.total_tokens_used += metadata.get('tokens_used', 0)
            
            if not success:
                stats.api_failures += 1
                self.invalid_entries.append({
                    'index': i,
                    'question': question,
                    'context': context[:200] + "...",
                    'error': metadata.get('error', 'Unknown error')
                })
                continue
            
            # Validate the pair
            is_valid, validation_errors = self.validator.validate_pair(grpo_pair)
            
            if is_valid:
                valid_pairs.append(grpo_pair)
                stats.valid_pairs += 1
            else:
                stats.validation_failures += 1
                self.invalid_entries.append({
                    'index': i,
                    'question': question,
                    'grpo_pair': grpo_pair,
                    'validation_errors': validation_errors
                })
        
        stats.rejected_pairs = stats.total_prompts_generated - stats.valid_pairs
        
        # Calculate averages
        stats.calculate_averages(valid_pairs)
        
        # Save valid pairs
        if valid_pairs:
            self._save_grpo_dataset(valid_pairs, output_path)
            self.logger.info(f"Saved {len(valid_pairs)} valid GRPO pairs to {output_path}")
        
        # Save invalid entries log
        if self.invalid_entries:
            self._save_invalid_entries_log()
        
        # Calculate processing time
        stats.processing_time = time.time() - start_time
        
        return stats
    
    def _save_grpo_dataset(self, pairs: List[Dict[str, str]], output_path: Path):
        """Save GRPO dataset to JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in pairs:
                json.dump(pair, f, ensure_ascii=False)
                f.write('\n')
    
    def _save_invalid_entries_log(self):
        """Save invalid entries to log file."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"GRPO Invalid Entries Log - Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, entry in enumerate(self.invalid_entries, 1):
                f.write(f"Invalid Entry #{i}\n")
                f.write(f"Index: {entry.get('index', 'Unknown')}\n")
                
                if 'question' in entry:
                    f.write(f"Question: {entry['question']}\n")
                
                if 'context' in entry:
                    f.write(f"Context: {entry['context']}\n")
                
                if 'error' in entry:
                    f.write(f"Error: {entry['error']}\n")
                
                if 'validation_errors' in entry:
                    f.write(f"Validation Errors: {entry['validation_errors']}\n")
                
                if 'grpo_pair' in entry:
                    f.write(f"Generated Pair: {json.dumps(entry['grpo_pair'], ensure_ascii=False, indent=2)}\n")
                
                f.write("-" * 40 + "\n\n")
        
        self.logger.info(f"Invalid entries logged to: {self.log_file}")
    
    def save_stats(self, stats: GRPOStats, stats_path: Path):
        """Save statistics to JSON file."""
        stats_dict = asdict(stats)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Statistics saved to: {stats_path}")
    
    def generate_report(self, stats: GRPOStats) -> str:
        """Generate summary report."""
        success_rate = (stats.valid_pairs / stats.total_prompts_generated * 100) if stats.total_prompts_generated > 0 else 0
        
        report = f"""
📊 GRPO DATASET GENERATION REPORT
{'=' * 50}
Generation Results:
  Total prompts generated: {stats.total_prompts_generated:,}
  Valid pairs: {stats.valid_pairs:,}
  Rejected pairs: {stats.rejected_pairs:,}
  Success rate: {success_rate:.1f}%

Quality Metrics:
  Average prompt length: {stats.avg_prompt_length:.1f} words
  Average chosen length: {stats.avg_chosen_length:.1f} words
  Average rejected length: {stats.avg_rejected_length:.1f} words
  Dataset purity: {stats.mongolian_purity:.1f}% Mongolian

Performance:
  Total processing time: {stats.processing_time:.1f}s
  Total API calls: {stats.total_api_calls:,}
  Total tokens used: {stats.total_tokens_used:,}
  API failures: {stats.api_failures:,}
  Validation failures: {stats.validation_failures:,}

Status: {'✅ SUCCESS' if stats.valid_pairs > 0 else '❌ FAILED'}
{'✅ Ready for GRPO fine-tuning' if stats.valid_pairs >= 50 else '⚠️ Consider generating more pairs for better training'}
"""
        
        return report


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build GRPO dataset from Mongolian historical data")
    parser.add_argument('--source', nargs='*', 
                       help='Source data files (default: auto-detect)')
    parser.add_argument('--rag-log', 
                       help='RAG Q&A log file path')
    parser.add_argument('--output', default='data/mgl_history_grpo.jsonl',
                       help='Output GRPO dataset path')
    parser.add_argument('--stats', default='data/mgl_history_grpo_stats.json',
                       help='Statistics output path')
    parser.add_argument('--pairs-per-topic', type=int, default=15,
                       help='Number of pairs to generate per topic')
    parser.add_argument('--log-file', default='data/grpo_invalid.log',
                       help='Invalid entries log file')
    
    args = parser.parse_args()
    
    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return 1
    
    # Determine source files
    if args.source:
        source_files = [Path(f) for f in args.source]
    else:
        # Auto-detect source files
        data_dir = Path('data')
        source_files = []
        
        # Look for translated dataset first
        for pattern in [
            'mgl_history_translated.jsonl',
            'mongolian_history_unified.jsonl',
            'mgl_history_labeled.jsonl'
        ]:
            file_path = data_dir / pattern
            if file_path.exists():
                source_files.append(file_path)
                break
    
    # Validate source files
    valid_files = []
    for file_path in source_files:
        if file_path.exists():
            valid_files.append(file_path)
        else:
            print(f"⚠️ Warning: Source file not found: {file_path}")
    
    if not valid_files:
        print("❌ No valid source files found")
        print("Expected files: mgl_history_translated.jsonl, mongolian_history_unified.jsonl")
        return 1
    
    print("🔄 GRPO Dataset Builder")
    print("=" * 50)
    print(f"Source files: {len(valid_files)}")
    for f in valid_files:
        print(f"  - {f}")
    print(f"Output: {args.output}")
    print(f"Pairs per topic: {args.pairs_per_topic}")
    if args.rag_log:
        print(f"RAG log: {args.rag_log}")
    print()
    
    # Initialize builder
    builder = GRPODatasetBuilder(api_key, args.log_file)
    
    # Build dataset
    try:
        rag_log_path = Path(args.rag_log) if args.rag_log else None
        
        stats = builder.build_grpo_dataset(
            source_files=valid_files,
            output_path=Path(args.output),
            rag_log_path=rag_log_path,
            pairs_per_topic=args.pairs_per_topic
        )
        
        # Save statistics
        builder.save_stats(stats, Path(args.stats))
        
        # Generate and display report
        report = builder.generate_report(stats)
        print(report)
        
        # Return appropriate exit code
        if stats.valid_pairs >= 10:
            return 0  # Success
        elif stats.valid_pairs > 0:
            return 2  # Partial success
        else:
            return 1  # Failure
            
    except KeyboardInterrupt:
        print("\n⚠️ GRPO generation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())