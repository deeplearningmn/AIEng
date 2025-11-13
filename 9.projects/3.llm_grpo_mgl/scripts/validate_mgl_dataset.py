#!/usr/bin/env python3
"""
Mongolian Dataset Validator

Validates the integrity, language consistency, and structure of Mongolian historical 
datasets before GRPO or fine-tuning. Ensures data quality and language purity.

Usage:
    python scripts/validate_mgl_dataset.py
    python scripts/validate_mgl_dataset.py --input-dir data --output-log data/validation.log
"""

import json
import re
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


@dataclass
class ValidationStats:
    """Statistics for a single dataset."""
    
    filename: str
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    mongolian_records: int = 0
    mixed_language_records: int = 0
    duplicate_records: int = 0
    empty_records: int = 0
    
    # Content statistics
    total_chars: int = 0
    total_tokens: int = 0
    min_length: int = float('inf')
    max_length: int = 0
    
    # Language purity
    mongolian_purity: float = 0.0
    
    # Field statistics
    missing_fields: List[str] = None
    field_counts: Dict[str, int] = None
    
    def __post_init__(self):
        if self.missing_fields is None:
            self.missing_fields = []
        if self.field_counts is None:
            self.field_counts = {}
    
    @property
    def avg_length(self) -> float:
        return self.total_chars / max(1, self.valid_records)
    
    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / max(1, self.valid_records)
    
    @property
    def validity_rate(self) -> float:
        return (self.valid_records / max(1, self.total_records)) * 100
    
    @property
    def mongolian_rate(self) -> float:
        return (self.mongolian_records / max(1, self.valid_records)) * 100


class MongolianLanguageDetector:
    """Detect and analyze Mongolian language content."""
    
    # Mongolian Cyrillic alphabet including special characters
    MONGOLIAN_PATTERN = re.compile(r'[А-ЯӨҮа-яөү]')
    LATIN_PATTERN = re.compile(r'[A-Za-z]')
    
    # Common Mongolian words for additional validation
    MONGOLIAN_WORDS = {
        'монгол', 'түүх', 'хаан', 'улс', 'орон', 'хүн', 'цаг', 'үе', 'газар',
        'нутаг', 'ард', 'түмэн', 'баатар', 'эзэн', 'гүрэн', 'хувьсгал',
        'тусгаар', 'тогтнол', 'засаг', 'төр', 'соёл', 'иргэншил'
    }
    
    @classmethod
    def detect_mongolian(cls, text: str) -> bool:
        """Check if text contains Mongolian characters."""
        if not text:
            return False
        return bool(cls.MONGOLIAN_PATTERN.search(text))
    
    @classmethod
    def calculate_mongolian_purity(cls, text: str) -> float:
        """Calculate percentage of Mongolian characters in text."""
        if not text:
            return 0.0
        
        mongolian_chars = len(cls.MONGOLIAN_PATTERN.findall(text))
        latin_chars = len(cls.LATIN_PATTERN.findall(text))
        total_letters = mongolian_chars + latin_chars
        
        if total_letters == 0:
            return 0.0
        
        return (mongolian_chars / total_letters) * 100
    
    @classmethod
    def is_mixed_language(cls, text: str, threshold: float = 30.0) -> bool:
        """Check if text contains significant Latin content (>threshold%)."""
        if not text:
            return False
        
        purity = cls.calculate_mongolian_purity(text)
        latin_chars = len(cls.LATIN_PATTERN.findall(text))
        total_chars = len(text)
        
        # Consider mixed if Latin percentage > threshold
        latin_percentage = (latin_chars / max(1, total_chars)) * 100
        
        return latin_percentage > threshold and purity < (100 - threshold)
    
    @classmethod
    def contains_mongolian_words(cls, text: str) -> bool:
        """Check if text contains common Mongolian words."""
        text_lower = text.lower()
        return any(word in text_lower for word in cls.MONGOLIAN_WORDS)


class DatasetValidator:
    """Main validator for Mongolian historical datasets."""
    
    def __init__(self, input_dir: str = "data", log_file: str = "data/invalid_records.log"):
        self.input_dir = Path(input_dir)
        self.log_file = Path(log_file)
        self.logger = self._setup_logging()
        
        # Validation state
        self.all_hashes: Set[str] = set()
        self.global_duplicates = 0
        self.validation_results: Dict[str, ValidationStats] = {}
        self.invalid_records: List[Dict[str, Any]] = []
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger = logging.getLogger('mongolian_validator')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        if not text:
            return 0
        
        # Rough estimation: 1 token ≈ 4 characters for Mongolian
        # This is approximate and may vary with actual tokenizers
        return len(text) // 4
    
    def _calculate_content_hash(self, text: str) -> str:
        """Calculate MD5 hash of normalized text content."""
        if not text:
            return ""
        
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _extract_text_content(self, record: Dict[str, Any]) -> str:
        """Extract text content from various record formats."""
        # Try different field names
        for field in ['text', 'content', 'prompt', 'chosen', 'input']:
            if field in record and record[field]:
                return str(record[field])
        
        # For GRPO format, try to combine prompt and chosen
        if 'prompt' in record and 'chosen' in record:
            prompt = str(record.get('prompt', ''))
            chosen = str(record.get('chosen', ''))
            return f"{prompt} {chosen}".strip()
        
        return ""
    
    def _validate_record_structure(self, record: Dict[str, Any], filename: str) -> Tuple[bool, List[str]]:
        """Validate record structure and required fields."""
        issues = []
        
        # Check if record is a dictionary
        if not isinstance(record, dict):
            issues.append("Record is not a dictionary")
            return False, issues
        
        # Check for required fields
        required_fields = ['text', 'content', 'prompt', 'input']
        has_required_field = any(field in record for field in required_fields)
        
        if not has_required_field:
            issues.append(f"Missing required fields. Expected one of: {required_fields}")
        
        # Check for GRPO format
        if 'prompt' in record:
            if 'chosen' not in record and 'rejected' not in record:
                issues.append("GRPO format missing 'chosen' or 'rejected' field")
        
        # Extract and validate text content
        text_content = self._extract_text_content(record)
        if not text_content or not text_content.strip():
            issues.append("Empty or missing text content")
            return False, issues
        
        return len(issues) == 0, issues
    
    def _validate_single_record(self, record: Dict[str, Any], filename: str, record_idx: int) -> Tuple[bool, ValidationStats]:
        """Validate a single record and update statistics."""
        stats = ValidationStats(filename=filename)
        stats.total_records = 1
        
        # Validate structure
        is_valid, issues = self._validate_record_structure(record, filename)
        
        if not is_valid:
            stats.invalid_records = 1
            self.invalid_records.append({
                'filename': filename,
                'record_index': record_idx,
                'issues': issues,
                'record': record
            })
            self.logger.warning(f"{filename}[{record_idx}]: {'; '.join(issues)}")
            return False, stats
        
        # Extract text content
        text_content = self._extract_text_content(record)
        
        if not text_content:
            stats.empty_records = 1
            return False, stats
        
        # Calculate content statistics
        content_length = len(text_content)
        token_count = self._estimate_tokens(text_content)
        
        stats.valid_records = 1
        stats.total_chars = content_length
        stats.total_tokens = token_count
        stats.min_length = content_length
        stats.max_length = content_length
        
        # Language detection
        is_mongolian = MongolianLanguageDetector.detect_mongolian(text_content)
        is_mixed = MongolianLanguageDetector.is_mixed_language(text_content)
        purity = MongolianLanguageDetector.calculate_mongolian_purity(text_content)
        
        if is_mongolian:
            stats.mongolian_records = 1
            stats.mongolian_purity = purity
        
        if is_mixed:
            stats.mixed_language_records = 1
            self.logger.info(f"{filename}[{record_idx}]: Mixed language detected (purity: {purity:.1f}%)")
        
        # Duplicate detection
        content_hash = self._calculate_content_hash(text_content)
        if content_hash in self.all_hashes:
            stats.duplicate_records = 1
            self.global_duplicates += 1
            self.logger.info(f"{filename}[{record_idx}]: Duplicate content detected")
        else:
            self.all_hashes.add(content_hash)
        
        # Field analysis
        stats.field_counts = {field: 1 for field in record.keys()}
        
        return True, stats
    
    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSON or JSONL file safely."""
        records = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL format - one JSON object per line
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                records.append(record)
                            except json.JSONDecodeError as e:
                                self.logger.error(f"{file_path.name}[{line_num}]: JSON decode error - {e}")
                else:
                    # Regular JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        records.extend(data)
                    elif isinstance(data, dict):
                        records.append(data)
                    else:
                        self.logger.error(f"{file_path.name}: Unexpected JSON structure")
        
        except Exception as e:
            self.logger.error(f"Failed to load {file_path.name}: {e}")
        
        return records
    
    def _merge_stats(self, stats1: ValidationStats, stats2: ValidationStats) -> ValidationStats:
        """Merge two ValidationStats objects."""
        merged = ValidationStats(filename=stats1.filename)
        
        # Sum numeric fields
        merged.total_records = stats1.total_records + stats2.total_records
        merged.valid_records = stats1.valid_records + stats2.valid_records
        merged.invalid_records = stats1.invalid_records + stats2.invalid_records
        merged.mongolian_records = stats1.mongolian_records + stats2.mongolian_records
        merged.mixed_language_records = stats1.mixed_language_records + stats2.mixed_language_records
        merged.duplicate_records = stats1.duplicate_records + stats2.duplicate_records
        merged.empty_records = stats1.empty_records + stats2.empty_records
        
        merged.total_chars = stats1.total_chars + stats2.total_chars
        merged.total_tokens = stats1.total_tokens + stats2.total_tokens
        
        # Min/max handling
        if stats1.min_length != float('inf') and stats2.min_length != float('inf'):
            merged.min_length = min(stats1.min_length, stats2.min_length)
        elif stats1.min_length != float('inf'):
            merged.min_length = stats1.min_length
        elif stats2.min_length != float('inf'):
            merged.min_length = stats2.min_length
        else:
            merged.min_length = 0
        
        merged.max_length = max(stats1.max_length, stats2.max_length)
        
        # Calculate weighted average purity
        total_mongolian = merged.mongolian_records
        if total_mongolian > 0:
            merged.mongolian_purity = (
                (stats1.mongolian_purity * stats1.mongolian_records + 
                 stats2.mongolian_purity * stats2.mongolian_records) / total_mongolian
            )
        
        # Merge field counts
        merged.field_counts = {}
        for field, count in stats1.field_counts.items():
            merged.field_counts[field] = merged.field_counts.get(field, 0) + count
        for field, count in stats2.field_counts.items():
            merged.field_counts[field] = merged.field_counts.get(field, 0) + count
        
        # Merge missing fields
        merged.missing_fields = list(set(stats1.missing_fields + stats2.missing_fields))
        
        return merged
    
    def validate_dataset(self, file_path: Path) -> ValidationStats:
        """Validate a single dataset file."""
        print(f"\n📁 Validating: {file_path.name}")
        
        # Load records
        records = self._load_json_file(file_path)
        
        if not records:
            print(f"   ❌ No valid records found")
            return ValidationStats(filename=file_path.name)
        
        # Initialize stats
        dataset_stats = ValidationStats(filename=file_path.name)
        
        # Validate each record
        for idx, record in enumerate(tqdm(records, desc=f"   Validating {file_path.name}")):
            is_valid, record_stats = self._validate_single_record(record, file_path.name, idx)
            dataset_stats = self._merge_stats(dataset_stats, record_stats)
        
        return dataset_stats
    
    def find_dataset_files(self) -> List[Path]:
        """Find all dataset files to validate."""
        patterns = [
            "*.jsonl",
            "*.json",
            "*history*.jsonl",
            "*history*.json",
            "*mgl*.jsonl",
            "*mgl*.json",
            "*grpo*.jsonl",
            "*grpo*.json"
        ]
        
        files = set()
        for pattern in patterns:
            files.update(self.input_dir.glob(pattern))
        
        # Filter out non-dataset files
        excluded_patterns = [
            "*config*",
            "*metadata*",
            "*stats*",
            "*summary*",
            "*report*"
        ]
        
        filtered_files = []
        for file_path in files:
            if not any(file_path.match(pattern) for pattern in excluded_patterns):
                filtered_files.append(file_path)
        
        return sorted(filtered_files)
    
    def print_dataset_report(self, stats: ValidationStats):
        """Print formatted report for a single dataset."""
        filename = stats.filename
        
        # Determine emoji based on quality
        if stats.validity_rate >= 95 and stats.mongolian_rate >= 95:
            emoji = "📘"  # Blue book - excellent
        elif stats.validity_rate >= 90 and stats.mongolian_rate >= 90:
            emoji = "📗"  # Green book - good
        elif stats.validity_rate >= 80:
            emoji = "📙"  # Orange book - fair
        else:
            emoji = "📕"  # Red book - poor
        
        print(f"\n{emoji} Dataset: {filename}")
        print(f"   Total records: {stats.total_records:,}")
        print(f"   Valid Mongolian entries: {stats.mongolian_records:,} ({stats.mongolian_rate:.1f}%)")
        
        if stats.duplicate_records > 0:
            print(f"   Duplicates found: {stats.duplicate_records:,}")
        
        if stats.invalid_records > 0:
            print(f"   ❌ Invalid records: {stats.invalid_records:,}")
        
        if stats.mixed_language_records > 0:
            print(f"   ⚠️  Mixed-language: {stats.mixed_language_records:,}")
        
        if stats.valid_records > 0:
            print(f"   Average length: {stats.avg_length:.0f} chars")
            if stats.min_length < float('inf'):
                print(f"   Longest: {stats.max_length:,} chars | Shortest: {stats.min_length:,} chars")
            print(f"   Language purity: {stats.mongolian_purity:.1f}% Mongolian")
        
        # Warnings
        if stats.mongolian_purity < 95.0:
            print(f"   ⚠️  Warning: Language purity below 95%")
        
        if stats.mixed_language_records > 0:
            print(f"   💡 Recommendation: Review and clean mixed-language entries")
    
    def print_summary_report(self):
        """Print overall validation summary."""
        if not self.validation_results:
            print("\n❌ No datasets validated")
            return
        
        print(f"\n{'='*60}")
        print("📊 VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        total_files = len(self.validation_results)
        total_records = sum(stats.total_records for stats in self.validation_results.values())
        total_valid = sum(stats.valid_records for stats in self.validation_results.values())
        total_mongolian = sum(stats.mongolian_records for stats in self.validation_results.values())
        total_mixed = sum(stats.mixed_language_records for stats in self.validation_results.values())
        
        print(f"📁 Files validated: {total_files}")
        print(f"📄 Total records: {total_records:,}")
        print(f"✅ Valid records: {total_valid:,} ({(total_valid/max(1,total_records)*100):.1f}%)")
        print(f"🇲🇳 Mongolian records: {total_mongolian:,} ({(total_mongolian/max(1,total_valid)*100):.1f}%)")
        
        if total_mixed > 0:
            print(f"⚠️  Mixed-language records: {total_mixed:,}")
        
        if self.global_duplicates > 0:
            print(f"🔄 Global duplicates: {self.global_duplicates:,}")
        
        # Quality assessment
        high_quality_files = sum(1 for stats in self.validation_results.values() 
                               if stats.validity_rate >= 95 and stats.mongolian_rate >= 95)
        
        print(f"\n🏆 High-quality datasets: {high_quality_files}/{total_files}")
        
        if len(self.invalid_records) > 0:
            print(f"📝 Invalid records logged to: {self.log_file}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        needs_cleaning = [name for name, stats in self.validation_results.items() 
                         if stats.mixed_language_records > 0 or stats.mongolian_purity < 95]
        
        if needs_cleaning:
            print(f"   🧹 Clean mixed-language content in: {', '.join(needs_cleaning)}")
        
        if self.global_duplicates > 0:
            print(f"   🔄 Remove {self.global_duplicates} duplicate records across datasets")
        
        low_quality = [name for name, stats in self.validation_results.items() 
                      if stats.validity_rate < 90]
        
        if low_quality:
            print(f"   ⚠️  Review data quality in: {', '.join(low_quality)}")
        
        if not needs_cleaning and self.global_duplicates == 0 and not low_quality:
            print(f"   ✅ All datasets are ready for GRPO/fine-tuning!")
    
    def validate_all_datasets(self) -> Dict[str, ValidationStats]:
        """Validate all datasets in the input directory."""
        print("🔍 Mongolian Dataset Validator")
        print("=" * 50)
        print(f"📂 Input directory: {self.input_dir}")
        print(f"📝 Log file: {self.log_file}")
        
        # Find dataset files
        dataset_files = self.find_dataset_files()
        
        if not dataset_files:
            print(f"\n❌ No dataset files found in {self.input_dir}")
            return {}
        
        print(f"\n📁 Found {len(dataset_files)} dataset files:")
        for file_path in dataset_files:
            print(f"   - {file_path.name}")
        
        # Validate each dataset
        for file_path in dataset_files:
            try:
                stats = self.validate_dataset(file_path)
                self.validation_results[file_path.name] = stats
                self.print_dataset_report(stats)
            except Exception as e:
                self.logger.error(f"Failed to validate {file_path.name}: {e}")
                print(f"   ❌ Validation failed: {e}")
        
        # Print summary
        self.print_summary_report()
        
        return self.validation_results


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate Mongolian historical datasets for GRPO/fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/validate_mgl_dataset.py
  python scripts/validate_mgl_dataset.py --input-dir data --output-log validation.log
  python scripts/validate_mgl_dataset.py --input-dir datasets --verbose
        """
    )
    
    parser.add_argument(
        '--input-dir',
        default='data',
        help='Directory containing dataset files (default: data)'
    )
    
    parser.add_argument(
        '--output-log',
        default='data/invalid_records.log',
        help='Output log file for invalid records (default: data/invalid_records.log)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Create validator
    validator = DatasetValidator(
        input_dir=args.input_dir,
        log_file=args.output_log
    )
    
    if args.verbose:
        validator.logger.setLevel(logging.DEBUG)
    
    # Run validation
    try:
        results = validator.validate_all_datasets()
        
        # Exit with appropriate code
        if results:
            # Check if any datasets have issues
            has_issues = any(
                stats.mixed_language_records > 0 or 
                stats.mongolian_purity < 95 or 
                stats.validity_rate < 90
                for stats in results.values()
            )
            
            exit_code = 1 if has_issues else 0
        else:
            exit_code = 1
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)