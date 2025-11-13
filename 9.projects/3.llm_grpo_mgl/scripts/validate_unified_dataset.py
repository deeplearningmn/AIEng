#!/usr/bin/env python3
"""
Validate the unified Mongolian history dataset.

This script checks:
1. JSON format validity
2. Required fields presence
3. Date format consistency
4. Content quality metrics
5. Duplicate detection
6. Period distribution
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import argparse


class DatasetValidator:
    """Validate the unified historical dataset."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.entries = []
        self.validation_results = {
            'total_entries': 0,
            'valid_entries': 0,
            'invalid_entries': 0,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
    
    def load_dataset(self) -> bool:
        """Load the JSONL dataset."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            self.entries.append(entry)
                        except json.JSONDecodeError as e:
                            self.validation_results['errors'].append(
                                f"Line {line_num}: Invalid JSON - {e}"
                            )
            
            self.validation_results['total_entries'] = len(self.entries)
            return True
            
        except Exception as e:
            self.validation_results['errors'].append(f"Failed to load dataset: {e}")
            return False
    
    def validate_entry_structure(self, entry: Dict[str, Any], index: int) -> bool:
        """Validate the structure of a single entry."""
        required_fields = ['text', 'title', 'source', 'period']
        optional_fields = ['date', 'url', 'chapter', 'word_count', 'dataset_source']
        
        is_valid = True
        
        # Check required fields
        for field in required_fields:
            if field not in entry:
                self.validation_results['errors'].append(
                    f"Entry {index}: Missing required field '{field}'"
                )
                is_valid = False
            elif not entry[field] or (isinstance(entry[field], str) and not entry[field].strip()):
                self.validation_results['warnings'].append(
                    f"Entry {index}: Empty required field '{field}'"
                )
        
        # Check field types
        if 'text' in entry and not isinstance(entry['text'], str):
            self.validation_results['errors'].append(
                f"Entry {index}: 'text' field must be string"
            )
            is_valid = False
        
        if 'word_count' in entry and not isinstance(entry['word_count'], (int, float)):
            self.validation_results['warnings'].append(
                f"Entry {index}: 'word_count' should be numeric"
            )
        
        return is_valid
    
    def validate_date_format(self, date_str: str, index: int) -> bool:
        """Validate date format."""
        if not date_str:
            return True  # Empty dates are allowed
        
        # Valid patterns: YYYY or YYYY-MM-DD
        patterns = [
            r'^\d{4}$',  # YYYY
            r'^\d{4}-\d{2}-\d{2}$'  # YYYY-MM-DD
        ]
        
        for pattern in patterns:
            if re.match(pattern, date_str):
                # Additional validation for year range
                year = int(date_str.split('-')[0])
                if 800 <= year <= 2030:
                    return True
                else:
                    self.validation_results['warnings'].append(
                        f"Entry {index}: Date year {year} outside expected range (800-2030)"
                    )
                    return False
        
        self.validation_results['warnings'].append(
            f"Entry {index}: Invalid date format '{date_str}'"
        )
        return False
    
    def validate_content_quality(self, entry: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Validate content quality metrics."""
        text = entry.get('text', '')
        title = entry.get('title', '')
        
        quality_metrics = {
            'text_length': len(text),
            'word_count': len(text.split()) if text else 0,
            'title_length': len(title),
            'has_mongolian_text': bool(re.search(r'[а-яё]', text.lower())),
            'has_special_chars': bool(re.search(r'[vєvvгvvлрvvн]', text)),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]) if text else 0
        }
        
        # Quality checks
        if quality_metrics['text_length'] < 50:
            self.validation_results['warnings'].append(
                f"Entry {index}: Very short text ({quality_metrics['text_length']} chars)"
            )
        
        if quality_metrics['word_count'] < 10:
            self.validation_results['warnings'].append(
                f"Entry {index}: Very few words ({quality_metrics['word_count']} words)"
            )
        
        if not quality_metrics['has_mongolian_text']:
            self.validation_results['warnings'].append(
                f"Entry {index}: No Mongolian text detected"
            )
        
        return quality_metrics
    
    def check_duplicates(self) -> List[Tuple[int, int]]:
        """Check for potential duplicates."""
        duplicates = []
        
        for i, entry1 in enumerate(self.entries):
            for j, entry2 in enumerate(self.entries[i+1:], i+1):
                # Check title similarity
                title1 = entry1.get('title', '').lower()
                title2 = entry2.get('title', '').lower()
                
                if title1 and title2 and title1 == title2:
                    duplicates.append((i, j))
                    continue
                
                # Check text similarity (first 100 chars)
                text1 = entry1.get('text', '')[:100].lower()
                text2 = entry2.get('text', '')[:100].lower()
                
                if text1 and text2 and text1 == text2:
                    duplicates.append((i, j))
        
        return duplicates
    
    def analyze_statistics(self) -> Dict[str, Any]:
        """Analyze dataset statistics."""
        stats = {
            'period_distribution': Counter(),
            'source_distribution': Counter(),
            'date_distribution': Counter(),
            'content_length_stats': [],
            'word_count_stats': [],
            'quality_metrics': defaultdict(list)
        }
        
        for entry in self.entries:
            # Period distribution
            period = entry.get('period', 'Unknown')
            stats['period_distribution'][period] += 1
            
            # Source distribution
            source = entry.get('source', 'Unknown')
            stats['source_distribution'][source] += 1
            
            # Date distribution (by century)
            date = entry.get('date', '')
            if date and len(date) >= 4:
                try:
                    year = int(date[:4])
                    century = f"{(year // 100) + 1}th century"
                    stats['date_distribution'][century] += 1
                except ValueError:
                    pass
            
            # Content metrics
            text = entry.get('text', '')
            stats['content_length_stats'].append(len(text))
            stats['word_count_stats'].append(len(text.split()) if text else 0)
            
            # Quality metrics
            quality = self.validate_content_quality(entry, 0)  # Index not used here
            for key, value in quality.items():
                stats['quality_metrics'][key].append(value)
        
        # Calculate summary statistics
        if stats['content_length_stats']:
            stats['avg_content_length'] = sum(stats['content_length_stats']) / len(stats['content_length_stats'])
            stats['min_content_length'] = min(stats['content_length_stats'])
            stats['max_content_length'] = max(stats['content_length_stats'])
        
        if stats['word_count_stats']:
            stats['avg_word_count'] = sum(stats['word_count_stats']) / len(stats['word_count_stats'])
            stats['min_word_count'] = min(stats['word_count_stats'])
            stats['max_word_count'] = max(stats['word_count_stats'])
        
        return stats
    
    def validate(self) -> Dict[str, Any]:
        """Run complete validation."""
        print(f"Validating dataset: {self.dataset_path}")
        
        # Load dataset
        if not self.load_dataset():
            return self.validation_results
        
        print(f"Loaded {len(self.entries)} entries")
        
        # Validate each entry
        valid_count = 0
        for i, entry in enumerate(self.entries):
            is_valid = True
            
            # Structure validation
            if not self.validate_entry_structure(entry, i):
                is_valid = False
            
            # Date validation
            date = entry.get('date', '')
            if date and not self.validate_date_format(date, i):
                is_valid = False
            
            # Content quality
            self.validate_content_quality(entry, i)
            
            if is_valid:
                valid_count += 1
        
        self.validation_results['valid_entries'] = valid_count
        self.validation_results['invalid_entries'] = len(self.entries) - valid_count
        
        # Check for duplicates
        duplicates = self.check_duplicates()
        if duplicates:
            self.validation_results['warnings'].append(
                f"Found {len(duplicates)} potential duplicate pairs"
            )
        
        # Generate statistics
        self.validation_results['statistics'] = self.analyze_statistics()
        
        return self.validation_results
    
    def print_report(self):
        """Print validation report."""
        results = self.validation_results
        
        print("\n" + "=" * 60)
        print("DATASET VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\nOverall Statistics:")
        print(f"  Total entries: {results['total_entries']}")
        print(f"  Valid entries: {results['valid_entries']}")
        print(f"  Invalid entries: {results['invalid_entries']}")
        print(f"  Validation rate: {(results['valid_entries']/results['total_entries']*100):.1f}%")
        
        if results['errors']:
            print(f"\nErrors ({len(results['errors'])}):")
            for error in results['errors'][:10]:  # Show first 10
                print(f"  - {error}")
            if len(results['errors']) > 10:
                print(f"  ... and {len(results['errors']) - 10} more")
        
        if results['warnings']:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results['warnings'][:10]:  # Show first 10
                print(f"  - {warning}")
            if len(results['warnings']) > 10:
                print(f"  ... and {len(results['warnings']) - 10} more")
        
        # Statistics
        stats = results['statistics']
        
        print(f"\nPeriod Distribution:")
        for period, count in stats['period_distribution'].most_common():
            print(f"  {period}: {count} entries")
        
        print(f"\nSource Distribution:")
        for source, count in stats['source_distribution'].most_common():
            print(f"  {source}: {count} entries")
        
        print(f"\nContent Statistics:")
        if 'avg_content_length' in stats:
            print(f"  Average content length: {stats['avg_content_length']:.0f} characters")
            print(f"  Content length range: {stats['min_content_length']} - {stats['max_content_length']}")
        
        if 'avg_word_count' in stats:
            print(f"  Average word count: {stats['avg_word_count']:.0f} words")
            print(f"  Word count range: {stats['min_word_count']} - {stats['max_word_count']}")
        
        # Quality assessment
        mongolian_text_count = sum(1 for entry in self.entries 
                                 if re.search(r'[а-яё]', entry.get('text', '').lower()))
        print(f"\nQuality Metrics:")
        print(f"  Entries with Mongolian text: {mongolian_text_count}/{results['total_entries']} ({mongolian_text_count/results['total_entries']*100:.1f}%)")
        
        # Date coverage
        dated_entries = sum(1 for entry in self.entries if entry.get('date'))
        print(f"  Entries with dates: {dated_entries}/{results['total_entries']} ({dated_entries/results['total_entries']*100:.1f}%)")
        
        print("\n" + "=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate unified Mongolian history dataset")
    parser.add_argument('dataset', help='Path to the JSONL dataset file')
    parser.add_argument('--output', help='Output validation report to file')
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.dataset)
    results = validator.validate()
    validator.print_report()
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()