#!/usr/bin/env python3
"""
Mongolian Dataset Translation Script

Automatically detects and translates English or mixed-language text fields into Mongolian
using OpenAI GPT-4o-mini API. Ensures all historical dataset entries are in clean
Mongolian text before training.

Usage:
    python scripts/translate_mixed_entries.py
    python scripts/translate_mixed_entries.py --files data/custom_dataset.json
    python scripts/translate_mixed_entries.py --threshold 0.15 --output data/custom_output.jsonl
"""

import json
import re
import os
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
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
class TranslationStats:
    """Statistics for translation process."""
    total_records: int = 0
    mixed_language_detected: int = 0
    translation_attempts: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    skipped_records: int = 0
    total_words_processed: int = 0
    total_api_calls: int = 0
    total_tokens_used: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate translation success rate."""
        if self.translation_attempts == 0:
            return 0.0
        return (self.successful_translations / self.translation_attempts) * 100
    
    @property
    def processing_time(self) -> float:
        """Calculate total processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def avg_latency(self) -> float:
        """Calculate average translation latency per entry."""
        if self.successful_translations == 0:
            return 0.0
        return self.processing_time / self.successful_translations


class LanguageDetector:
    """Detect language composition in text."""
    
    def __init__(self, english_threshold: float = 0.2):
        """
        Initialize language detector.
        
        Args:
            english_threshold: Minimum ratio of English characters to trigger translation
        """
        self.english_threshold = english_threshold
        self.mongolian_pattern = re.compile(r'[А-ЯӨҮа-яөү]')
        self.english_pattern = re.compile(r'[A-Za-z]')
        self.number_pattern = re.compile(r'\d')
        self.punctuation_pattern = re.compile(r'[.,!?;:()\[\]{}\"\'-]')
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for language composition.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with language analysis results
        """
        if not text or not text.strip():
            return {
                'needs_translation': False,
                'english_ratio': 0.0,
                'mongolian_ratio': 0.0,
                'english_chars': 0,
                'mongolian_chars': 0,
                'total_chars': 0,
                'is_mixed': False
            }
        
        # Count character types
        mongolian_chars = len(self.mongolian_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        numbers = len(self.number_pattern.findall(text))
        punctuation = len(self.punctuation_pattern.findall(text))
        
        # Total alphabetic characters (excluding numbers and punctuation)
        total_alpha_chars = mongolian_chars + english_chars
        total_chars = len(text)
        
        # Calculate ratios
        english_ratio = (english_chars / total_alpha_chars) if total_alpha_chars > 0 else 0
        mongolian_ratio = (mongolian_chars / total_alpha_chars) if total_alpha_chars > 0 else 0
        
        # Determine if translation is needed
        needs_translation = english_ratio >= self.english_threshold
        is_mixed = 0.1 <= english_ratio <= 0.9  # Mixed if 10-90% English
        
        return {
            'needs_translation': needs_translation,
            'english_ratio': english_ratio,
            'mongolian_ratio': mongolian_ratio,
            'english_chars': english_chars,
            'mongolian_chars': mongolian_chars,
            'total_chars': total_chars,
            'is_mixed': is_mixed
        }
    
    def needs_translation(self, text: str) -> bool:
        """Check if text needs translation."""
        analysis = self.analyze_text(text)
        return analysis['needs_translation']


class MongolianTranslator:
    """OpenAI-powered translator for converting text to Mongolian."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 temperature: float = 0.2, max_tokens: int = 900, max_retries: int = 3):
        """
        Initialize translator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            max_retries: Maximum retry attempts
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        # Translation prompt template
        self.system_prompt = """You are a professional Mongolian translator specializing in historical texts.

Your task is to translate text into clear, natural Mongolian while:
- Preserving academic tone and factual meaning
- Maintaining historical accuracy
- Using proper Mongolian Cyrillic script
- Keeping the same paragraph structure
- Not adding any commentary or explanations

Respond ONLY with the Mongolian translation, nothing else."""
        
        self.user_prompt_template = """Translate the following text into Mongolian:

{text}"""
    
    def translate_text(self, text: str) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Translate text to Mongolian.
        
        Args:
            text: Text to translate
            
        Returns:
            Tuple of (translated_text, success, metadata)
        """
        if not text or not text.strip():
            return text, True, {'tokens_used': 0, 'api_calls': 0}
        
        metadata = {'tokens_used': 0, 'api_calls': 0, 'error': None}
        
        for attempt in range(self.max_retries):
            try:
                # Add small delay between retries
                if attempt > 0:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                
                # Create API request
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt_template.format(text=text)}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract translated text
                translated_text = response.choices[0].message.content.strip()
                
                # Update metadata
                metadata['api_calls'] = attempt + 1
                if hasattr(response, 'usage') and response.usage:
                    metadata['tokens_used'] = response.usage.total_tokens
                
                # Validate translation (should be mostly Mongolian now)
                if self._validate_translation(translated_text):
                    return translated_text, True, metadata
                else:
                    metadata['error'] = f"Translation validation failed on attempt {attempt + 1}"
                    continue
                    
            except RateLimitError as e:
                metadata['error'] = f"Rate limit error on attempt {attempt + 1}: {e}"
                if attempt < self.max_retries - 1:
                    delay = (2 ** attempt) * 60  # Exponential backoff in minutes for rate limits
                    time.sleep(delay)
                continue
                
            except APIConnectionError as e:
                metadata['error'] = f"Connection error on attempt {attempt + 1}: {e}"
                continue
                
            except AuthenticationError as e:
                metadata['error'] = f"Authentication error: {e}"
                break  # Don't retry auth errors
                
            except Exception as e:
                metadata['error'] = f"Unexpected error on attempt {attempt + 1}: {e}"
                continue
        
        # All attempts failed
        return text, False, metadata
    
    def _validate_translation(self, translated_text: str) -> bool:
        """
        Validate that translation is primarily Mongolian.
        
        Args:
            translated_text: Text to validate
            
        Returns:
            True if translation is valid
        """
        if not translated_text or not translated_text.strip():
            return False
        
        # Check for Mongolian characters
        mongolian_chars = len(re.findall(r'[А-ЯӨҮа-яөү]', translated_text))
        english_chars = len(re.findall(r'[A-Za-z]', translated_text))
        total_alpha = mongolian_chars + english_chars
        
        if total_alpha == 0:
            return False
        
        # Should be at least 80% Mongolian after translation
        mongolian_ratio = mongolian_chars / total_alpha
        return mongolian_ratio >= 0.8


class DatasetTranslator:
    """Main class for translating mixed-language datasets."""
    
    def __init__(self, api_key: str, english_threshold: float = 0.2, 
                 log_file: str = "data/mgl_history_translated.log"):
        """
        Initialize dataset translator.
        
        Args:
            api_key: OpenAI API key
            english_threshold: Minimum English ratio to trigger translation
            log_file: Path to log file
        """
        self.detector = LanguageDetector(english_threshold)
        self.translator = MongolianTranslator(api_key)
        self.log_file = Path(log_file)
        self.failed_entries = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler for progress
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
    
    def _extract_text_fields(self, record: Dict[str, Any]) -> List[Tuple[str, str]]:
        """
        Extract text fields from a record.
        
        Args:
            record: Record dictionary
            
        Returns:
            List of (field_name, text_content) tuples
        """
        text_fields = []
        
        # Standard fields
        for field in ['text', 'content', 'title']:
            if field in record and record[field]:
                text_fields.append((field, str(record[field])))
        
        # GRPO format fields
        for field in ['prompt', 'chosen', 'rejected']:
            if field in record and record[field]:
                text_fields.append((field, str(record[field])))
        
        return text_fields
    
    def _load_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Load dataset from JSON or JSONL file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            List of records
        """
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL format
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                else:
                    # Regular JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict):
                        records.append(data)
                    else:
                        self.logger.error(f"Unexpected JSON structure in {file_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
        
        return records
    
    def _save_translated_records(self, records: List[Dict[str, Any]], output_path: Path):
        """
        Save translated records to JSONL file.
        
        Args:
            records: List of translated records
            output_path: Output file path
        """
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\\n')
    
    def translate_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
        """
        Translate a single record.
        
        Args:
            record: Record to translate
            
        Returns:
            Tuple of (translated_record, success, metadata)
        """
        translated_record = record.copy()
        translation_metadata = {
            'fields_translated': [],
            'fields_skipped': [],
            'total_tokens': 0,
            'api_calls': 0,
            'errors': []
        }
        
        # Extract text fields
        text_fields = self._extract_text_fields(record)
        
        if not text_fields:
            return translated_record, True, translation_metadata
        
        overall_success = True
        
        for field_name, text_content in text_fields:
            # Check if translation is needed
            if not self.detector.needs_translation(text_content):
                translation_metadata['fields_skipped'].append(field_name)
                continue
            
            # Translate the field
            translated_text, success, metadata = self.translator.translate_text(text_content)
            
            # Update metadata
            translation_metadata['total_tokens'] += metadata.get('tokens_used', 0)
            translation_metadata['api_calls'] += metadata.get('api_calls', 0)
            
            if success:
                translated_record[field_name] = translated_text
                translation_metadata['fields_translated'].append(field_name)
                self.logger.debug(f"Successfully translated field '{field_name}'")
            else:
                overall_success = False
                error_msg = metadata.get('error', 'Unknown error')
                translation_metadata['errors'].append(f"Field '{field_name}': {error_msg}")
                self.logger.warning(f"Failed to translate field '{field_name}': {error_msg}")
        
        return translated_record, overall_success, translation_metadata
    
    def translate_dataset(self, input_files: List[Path], output_path: Path) -> TranslationStats:
        """
        Translate entire dataset.
        
        Args:
            input_files: List of input file paths
            output_path: Output file path
            
        Returns:
            Translation statistics
        """
        stats = TranslationStats()
        stats.start_time = datetime.now()
        
        self.logger.info(f"Starting translation of {len(input_files)} file(s)")
        self.logger.info(f"Output will be saved to: {output_path}")
        
        all_translated_records = []
        
        # Process each input file
        for file_path in input_files:
            self.logger.info(f"Processing file: {file_path}")
            
            # Load records
            records = self._load_dataset(file_path)
            if not records:
                self.logger.warning(f"No records found in {file_path}")
                continue
            
            self.logger.info(f"Loaded {len(records)} records from {file_path}")
            
            # Process each record
            for i, record in enumerate(tqdm(records, desc=f"Translating {file_path.name}")):
                stats.total_records += 1
                
                # Check if record needs translation
                text_fields = self._extract_text_fields(record)
                needs_translation = False
                
                for field_name, text_content in text_fields:
                    if self.detector.needs_translation(text_content):
                        needs_translation = True
                        stats.mixed_language_detected += 1
                        break
                
                if not needs_translation:
                    # Record is already clean Mongolian
                    all_translated_records.append(record)
                    stats.skipped_records += 1
                    continue
                
                # Translate the record
                stats.translation_attempts += 1
                translated_record, success, metadata = self.translate_record(record)
                
                # Update statistics
                stats.total_tokens_used += metadata.get('total_tokens', 0)
                stats.total_api_calls += metadata.get('api_calls', 0)
                
                # Count words processed
                for field_name, text_content in text_fields:
                    if field_name in metadata.get('fields_translated', []):
                        stats.total_words_processed += len(text_content.split())
                
                if success:
                    all_translated_records.append(translated_record)
                    stats.successful_translations += 1
                    self.logger.debug(f"Successfully translated record {i+1}")
                else:
                    stats.failed_translations += 1
                    self.failed_entries.append({
                        'file': str(file_path),
                        'record_index': i,
                        'record': record,
                        'errors': metadata.get('errors', [])
                    })
                    self.logger.error(f"Failed to translate record {i+1}: {metadata.get('errors', [])}")
        
        # Save translated records
        if all_translated_records:
            self._save_translated_records(all_translated_records, output_path)
            self.logger.info(f"Saved {len(all_translated_records)} translated records to {output_path}")
        
        # Save failed entries log
        if self.failed_entries:
            failed_log_path = output_path.parent / "translate_failed.log"
            self._save_failed_entries_log(failed_log_path)
        
        stats.end_time = datetime.now()
        return stats
    
    def _save_failed_entries_log(self, log_path: Path):
        """Save failed entries to log file."""
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Failed Translation Entries - Generated: {datetime.now().isoformat()}\\n")
            f.write("=" * 80 + "\\n\\n")
            
            for i, entry in enumerate(self.failed_entries, 1):
                f.write(f"Failed Entry #{i}\\n")
                f.write(f"File: {entry['file']}\\n")
                f.write(f"Record Index: {entry['record_index']}\\n")
                f.write(f"Errors: {entry['errors']}\\n")
                f.write(f"Original Record: {json.dumps(entry['record'], ensure_ascii=False, indent=2)}\\n")
                f.write("-" * 40 + "\\n\\n")
        
        self.logger.info(f"Failed entries logged to: {log_path}")
    
    def generate_summary_report(self, stats: TranslationStats, output_path: Path) -> str:
        """
        Generate translation summary report.
        
        Args:
            stats: Translation statistics
            output_path: Output file path
            
        Returns:
            Summary report string
        """
        # Calculate final Mongolian purity
        final_purity = 100.0  # Assume 100% if all translations succeeded
        if stats.failed_translations > 0:
            success_ratio = stats.successful_translations / (stats.successful_translations + stats.failed_translations)
            final_purity = success_ratio * 100
        
        report = f"""
📊 MONGOLIAN TRANSLATION SUMMARY
{'=' * 50}
Input Processing:
  Total records processed: {stats.total_records:,}
  Mixed-language entries detected: {stats.mixed_language_detected:,}
  Records skipped (already Mongolian): {stats.skipped_records:,}

Translation Results:
  Translation attempts: {stats.translation_attempts:,}
  Successful translations: {stats.successful_translations:,}
  Failed translations: {stats.failed_translations:,}
  Success rate: {stats.success_rate:.1f}%

Performance Metrics:
  Total processing time: {stats.processing_time:.1f}s
  Average latency per entry: {stats.avg_latency:.1f}s
  Total API calls: {stats.total_api_calls:,}
  Total tokens used: {stats.total_tokens_used:,}
  Total words processed: {stats.total_words_processed:,}

Output Quality:
  Final Mongolian purity: {final_purity:.1f}%
  Output file: {output_path}
  Total output records: {stats.total_records - stats.failed_translations:,}

{'✅ SUCCESS' if stats.failed_translations == 0 else '⚠️ PARTIAL SUCCESS' if stats.successful_translations > 0 else '❌ FAILED'}
"""
        
        return report


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Translate mixed-language entries to Mongolian")
    parser.add_argument('--files', nargs='*', 
                       help='Specific files to translate')
    parser.add_argument('--output', default='data/mgl_history_translated.jsonl',
                       help='Output file path')
    parser.add_argument('--threshold', type=float, default=0.2,
                       help='English ratio threshold for translation (default: 0.2)')
    parser.add_argument('--log-file', default='data/mgl_history_translated.log',
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        return 1
    
    # Determine input files
    if args.files:
        input_files = [Path(f) for f in args.files]
    else:
        # Default files
        data_dir = Path('data')
        input_files = []
        
        # Look for common dataset files
        for pattern in [
            'mgl_history_labeled.jsonl',
            'modern_history_dataset.json',
            'mgl_history_grpo.jsonl',
            'mongolian_history_unified.jsonl'
        ]:
            file_path = data_dir / pattern
            if file_path.exists():
                input_files.append(file_path)
    
    # Validate input files
    valid_files = []
    for file_path in input_files:
        if file_path.exists():
            valid_files.append(file_path)
        else:
            print(f"⚠️ Warning: File not found: {file_path}")
    
    if not valid_files:
        print("❌ No valid input files found")
        print("Expected files: mgl_history_labeled.jsonl, modern_history_dataset.json, mgl_history_grpo.jsonl")
        return 1
    
    print("🔄 Mongolian Dataset Translator")
    print("=" * 50)
    print(f"Input files: {len(valid_files)}")
    for f in valid_files:
        print(f"  - {f}")
    print(f"Output file: {args.output}")
    print(f"English threshold: {args.threshold}")
    print()
    
    # Initialize translator
    translator = DatasetTranslator(
        api_key=api_key,
        english_threshold=args.threshold,
        log_file=args.log_file
    )
    
    # Perform translation
    try:
        stats = translator.translate_dataset(valid_files, Path(args.output))
        
        # Generate and display summary
        summary = translator.generate_summary_report(stats, Path(args.output))
        print(summary)
        
        # Save summary to log
        with open(args.log_file, 'a', encoding='utf-8') as f:
            f.write("\\n" + summary)
        
        # Return appropriate exit code
        if stats.failed_translations == 0:
            return 0  # Complete success
        elif stats.successful_translations > 0:
            return 2  # Partial success
        else:
            return 1  # Complete failure
            
    except KeyboardInterrupt:
        print("\\n⚠️ Translation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())