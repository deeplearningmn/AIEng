#!/usr/bin/env python3
"""
Clean and merge JSON outputs from the GPT-4o-mini collector.

This script:
1. Loads all JSON/JSONL files from the data directory
2. Removes duplicates based on content similarity
3. Normalizes date formats to consistent YYYY or YYYY-MM-DD format
4. Combines into one unified JSONL dataset
5. Adds metadata and validation
"""

import json
import re
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime
from difflib import SequenceMatcher
import argparse


class HistoricalDataCleaner:
    """Clean and merge historical data from multiple JSON sources."""
    
    def __init__(self, data_dir: str = "data", output_file: str = "data/mongolian_history_unified.jsonl"):
        self.data_dir = Path(data_dir)
        self.output_file = Path(output_file)
        self.seen_hashes: Set[str] = set()
        self.seen_content: List[str] = []
        self.duplicate_count = 0
        self.processed_count = 0
        
    def normalize_date(self, date_str: str) -> str:
        """Normalize date formats to YYYY or YYYY-MM-DD."""
        if not date_str or not isinstance(date_str, str):
            return ""
        
        date_str = date_str.strip()
        
        # Handle various date formats
        patterns = [
            # YYYY-MM-DD format (keep as is)
            (r'^\d{4}-\d{2}-\d{2}$', lambda m: m.group(0)),
            
            # YYYY format (keep as is)
            (r'^\d{4}$', lambda m: m.group(0)),
            
            # YYYY оны MM сарын DD (Mongolian format)
            (r'(\d{4})\s*оны?\s*(\d{1,2})\s*сарын?\s*(\d{1,2})', 
             lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
            
            # YYYY он (Mongolian year only)
            (r'(\d{4})\s*он', lambda m: m.group(1)),
            
            # MM/DD/YYYY or DD/MM/YYYY
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: m.group(3)),
            
            # YYYY.MM.DD
            (r'(\d{4})\.(\d{1,2})\.(\d{1,2})', 
             lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"),
            
            # Extract first 4-digit year from any string
            (r'(\d{4})', lambda m: m.group(1)),
        ]
        
        for pattern, formatter in patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    result = formatter(match)
                    # Validate year is reasonable (between 800 and 2030)
                    year = int(result.split('-')[0])
                    if 800 <= year <= 2030:
                        return result
                except (ValueError, IndexError):
                    continue
        
        return ""
    
    def normalize_period(self, period_str: str) -> str:
        """Normalize period descriptions to consistent format."""
        if not period_str or not isinstance(period_str, str):
            return "Тодорхойгүй үе"
        
        period_str = period_str.strip()
        
        # Common period mappings
        period_mappings = {
            "XIII зуун": "XIII зуун",
            "XVII–XIX зуун": "XVII-XIX зуун", 
            "Эртнээс орчин үе хүртэл": "Эртнээс орчин үе хүртэл",
            "XX зуун": "XX зуун",
            "XXI зуун": "XXI зуун",
            "Орчин үе": "Орчин үе",
            "Дундад зуун": "Дундад зуун",
            "Эртний үе": "Эртний үе"
        }
        
        # Try exact match first
        if period_str in period_mappings:
            return period_mappings[period_str]
        
        # Extract century information
        century_pattern = r'(\d+(?:-\d+)?)\s*зуун'
        match = re.search(century_pattern, period_str)
        if match:
            return f"{match.group(1)} зуун"
        
        return period_str
    
    def calculate_content_hash(self, text: str) -> str:
        """Calculate hash of normalized content for duplicate detection."""
        # Normalize text for comparison
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def is_similar_content(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """Check if two texts are similar using sequence matching."""
        if len(text1) < 50 or len(text2) < 50:
            return False
        
        # Use first 500 characters for comparison to avoid performance issues
        text1_sample = text1[:500].lower()
        text2_sample = text2[:500].lower()
        
        similarity = SequenceMatcher(None, text1_sample, text2_sample).ratio()
        return similarity >= threshold
    
    def is_duplicate(self, entry: Dict[str, Any]) -> bool:
        """Check if entry is a duplicate based on content hash and similarity."""
        text = entry.get('text', '')
        if not text or len(text.strip()) < 20:
            return True
        
        content_hash = self.calculate_content_hash(text)
        
        # Check exact hash match
        if content_hash in self.seen_hashes:
            return True
        
        # Check similarity with existing content
        for existing_text in self.seen_content:
            if self.is_similar_content(text, existing_text):
                return True
        
        # Not a duplicate - add to tracking
        self.seen_hashes.add(content_hash)
        self.seen_content.append(text)
        return False
    
    def clean_entry(self, entry: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Clean and normalize a single entry."""
        cleaned = {}
        
        # Required fields
        cleaned['text'] = entry.get('text', '').strip()
        if not cleaned['text']:
            return None
        
        # Normalize date
        date_field = entry.get('date', entry.get('period', ''))
        cleaned['date'] = self.normalize_date(str(date_field))
        
        # Title (prefer title, fall back to first 50 chars of text)
        title = entry.get('title', '')
        if not title and cleaned['text']:
            title = cleaned['text'][:50].split('.')[0].strip()
            if len(title) < 10:
                title = cleaned['text'][:50].strip()
        cleaned['title'] = title
        
        # Source information
        cleaned['source'] = entry.get('source', source_file)
        cleaned['dataset_source'] = entry.get('dataset_source', source_file)
        
        # Optional fields
        if 'url' in entry:
            cleaned['url'] = entry['url']
        if 'chapter' in entry:
            cleaned['chapter'] = entry['chapter']
        if 'word_count' in entry:
            cleaned['word_count'] = entry['word_count']
        elif cleaned['text']:
            cleaned['word_count'] = len(cleaned['text'].split())
        
        # Period information
        period = entry.get('period', '')
        if not period and cleaned['date']:
            # Infer period from date
            try:
                year = int(cleaned['date'].split('-')[0])
                if year < 1200:
                    period = "Эртний үе"
                elif year < 1600:
                    period = "Дундад зуун"
                elif year < 1900:
                    period = "XVII-XIX зуун"
                elif year < 2000:
                    period = "XX зуун"
                else:
                    period = "XXI зуун"
            except (ValueError, IndexError):
                period = "Тодорхойгүй үе"
        
        cleaned['period'] = self.normalize_period(period)
        
        # Confidence and matches (if available)
        cleaned['period_confidence'] = entry.get('period_confidence', 0.5)
        cleaned['period_matches'] = entry.get('period_matches', 0)
        
        # Add processing metadata
        cleaned['processed_date'] = datetime.now().isoformat()
        cleaned['content_length'] = len(cleaned['text'])
        
        return cleaned
    
    def load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load entries from a JSON or JSONL file."""
        entries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL format - one JSON object per line
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                entries.append(entry)
                            except json.JSONDecodeError as e:
                                print(f"Warning: Invalid JSON on line {line_num} in {file_path}: {e}")
                else:
                    # Regular JSON format
                    data = json.load(f)
                    if isinstance(data, list):
                        entries.extend(data)
                    elif isinstance(data, dict):
                        entries.append(data)
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        
        return entries
    
    def find_json_files(self) -> List[Path]:
        """Find all JSON and JSONL files in the data directory."""
        json_files = set()  # Use set to avoid duplicates
        
        # Look for files in data directory and subdirectories
        for pattern in ['*.json', '*.jsonl']:
            json_files.update(self.data_dir.glob(pattern))
            json_files.update(self.data_dir.glob(f'**/{pattern}'))
        
        # Filter out output files and reports
        filtered_files = []
        exclude_patterns = ['report', 'demo', 'sample', 'unified']
        
        for file_path in json_files:
            if not any(pattern in file_path.name.lower() for pattern in exclude_patterns):
                filtered_files.append(file_path)
        
        return sorted(filtered_files)
    
    def process_files(self) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Process all JSON files and return cleaned entries with statistics."""
        all_entries = []
        stats = {
            'files_processed': 0,
            'total_entries': 0,
            'duplicates_removed': 0,
            'invalid_entries': 0,
            'final_entries': 0
        }
        
        json_files = self.find_json_files()
        print(f"Found {len(json_files)} JSON/JSONL files to process")
        
        for file_path in json_files:
            print(f"Processing: {file_path}")
            entries = self.load_json_file(file_path)
            stats['files_processed'] += 1
            stats['total_entries'] += len(entries)
            
            for entry in entries:
                # Clean the entry
                cleaned = self.clean_entry(entry, file_path.name)
                if not cleaned:
                    stats['invalid_entries'] += 1
                    continue
                
                # Check for duplicates
                if self.is_duplicate(cleaned):
                    stats['duplicates_removed'] += 1
                    continue
                
                all_entries.append(cleaned)
        
        stats['final_entries'] = len(all_entries)
        return all_entries, stats
    
    def write_output(self, entries: List[Dict[str, Any]], stats: Dict[str, int]):
        """Write cleaned entries to output file."""
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sort entries by date (with empty dates last)
        def sort_key(entry):
            date = entry.get('date', '')
            if not date:
                return '9999'  # Put entries without dates at the end
            return date
        
        entries.sort(key=sort_key)
        
        # Write JSONL output
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        # Write statistics file
        stats_file = self.output_file.with_suffix('.stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'processing_date': datetime.now().isoformat(),
                'output_file': str(self.output_file),
                'statistics': stats,
                'sample_periods': self._get_period_distribution(entries)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nOutput written to: {self.output_file}")
        print(f"Statistics written to: {stats_file}")
    
    def _get_period_distribution(self, entries: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of entries by period."""
        period_counts = {}
        for entry in entries:
            period = entry.get('period', 'Тодорхойгүй үе')
            period_counts[period] = period_counts.get(period, 0) + 1
        return dict(sorted(period_counts.items()))
    
    def run(self):
        """Run the complete cleaning and merging process."""
        print("Mongolian History Data Cleaner and Merger")
        print("=" * 50)
        
        entries, stats = self.process_files()
        
        print(f"\nProcessing Summary:")
        print(f"  Files processed: {stats['files_processed']}")
        print(f"  Total entries found: {stats['total_entries']}")
        print(f"  Invalid entries removed: {stats['invalid_entries']}")
        print(f"  Duplicates removed: {stats['duplicates_removed']}")
        print(f"  Final clean entries: {stats['final_entries']}")
        
        if entries:
            self.write_output(entries, stats)
            
            # Show sample of periods
            period_dist = self._get_period_distribution(entries)
            print(f"\nPeriod Distribution:")
            for period, count in period_dist.items():
                print(f"  {period}: {count} entries")
        else:
            print("No valid entries found to process.")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Clean and merge Mongolian historical data")
    parser.add_argument('--data-dir', default='data', help='Directory containing JSON files')
    parser.add_argument('--output', default='data/mongolian_history_unified.jsonl', 
                       help='Output JSONL file')
    parser.add_argument('--similarity-threshold', type=float, default=0.85,
                       help='Similarity threshold for duplicate detection (0.0-1.0)')
    
    args = parser.parse_args()
    
    cleaner = HistoricalDataCleaner(args.data_dir, args.output)
    cleaner.run()


if __name__ == "__main__":
    main()