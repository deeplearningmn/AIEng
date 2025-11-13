#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import glob
import os
import re
import hashlib
import random
from collections import defaultdict

def find_input_files():
    """Automatically detect all JSONL files in data/ directory"""
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return []
    
    # Find all JSONL files
    jsonl_pattern = os.path.join(data_dir, '*.jsonl')
    all_jsonl_files = glob.glob(jsonl_pattern)
    
    # Files to skip (output files or previously merged)
    skip_files = {
        'dataset_train.jsonl',
        'dataset_test.jsonl', 
        'mgl_history_merged.jsonl',
        'cleaned_data.jsonl',
        'labeled_data.jsonl'
    }
    
    input_files = []
    skipped_files = []
    
    for file_path in all_jsonl_files:
        filename = os.path.basename(file_path)
        if filename in skip_files:
            skipped_files.append(filename)
        else:
            input_files.append(file_path)
    
    print(f"📁 Found {len(input_files)} input files:")
    for file_path in input_files:
        print(f"   📄 {os.path.basename(file_path)}")
    
    if skipped_files:
        print(f"⏭️  Skipped {len(skipped_files)} output files: {', '.join(skipped_files)}")
    
    return input_files

def is_mongolian_text(text):
    """Check if text contains significant Mongolian Cyrillic content"""
    if not text or len(text.strip()) < 20:
        return False
    
    # Count Cyrillic characters (used in Mongolian)
    cyrillic_chars = len(re.findall(r'[а-яёүө]', text.lower()))
    total_chars = len(re.findall(r'[а-яёүөa-z]', text.lower()))
    
    if total_chars == 0:
        return False
    
    # If more than 60% cyrillic characters, likely Mongolian
    cyrillic_ratio = cyrillic_chars / total_chars
    return cyrillic_ratio > 0.6

def clean_text(text):
    """Advanced text cleaning and normalization"""
    if not text:
        return ""
    
    # Remove invisible characters and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remove zero-width characters and non-breaking spaces
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\u2060\u180e]', ' ', text)
    
    # Remove soft hyphens and line-break hyphens
    text = re.sub(r'[\u00ad\u2010-\u2015]', '', text)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Clean up spaces around punctuation
    text = re.sub(r'\s+([,.!?;:)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    return text.strip()

def get_text_hash(text):
    """Generate MD5 hash for duplicate detection"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def load_jsonl_file(file_path):
    """Load and process a single JSONL file"""
    filename = os.path.basename(file_path)
    print(f"🔗 Merging file: {filename}")
    
    records = []
    line_count = 0
    error_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # Add dataset source information
                    record['dataset_source'] = filename
                    
                    records.append(record)
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 3:  # Only show first few errors
                        print(f"⚠️  JSON error in {filename} line {line_num}: {str(e)}")
    
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return []
    except Exception as e:
        print(f"❌ Error reading {filename}: {str(e)}")
        return []
    
    if error_count > 3:
        print(f"⚠️  ... and {error_count - 3} more JSON errors in {filename}")
    
    print(f"   📊 Loaded {len(records)} records from {filename}")
    return records

def process_and_filter_records(all_records):
    """Clean, filter, and deduplicate all records"""
    print(f"\n🧹 Processing and filtering {len(all_records)} total records...")
    
    processed_records = []
    seen_hashes = set()
    stats = {
        'total_input': len(all_records),
        'too_short': 0,
        'non_mongolian': 0,
        'duplicates': 0,
        'valid': 0
    }
    
    for record in all_records:
        # Extract and clean text
        text = record.get('text', '')
        if not text:
            stats['too_short'] += 1
            continue
        
        cleaned_text = clean_text(text)
        
        # Check minimum length
        if len(cleaned_text) < 100:
            stats['too_short'] += 1
            continue
        
        # Check if text is Mongolian
        if not is_mongolian_text(cleaned_text):
            stats['non_mongolian'] += 1
            continue
        
        # Check for duplicates
        text_hash = get_text_hash(cleaned_text)
        if text_hash in seen_hashes:
            stats['duplicates'] += 1
            continue
        
        seen_hashes.add(text_hash)
        
        # Create processed record
        processed_record = {
            'text': cleaned_text,
            'dataset_source': record.get('dataset_source', 'unknown'),
            'word_count': len(cleaned_text.split())
        }
        
        # Preserve metadata fields if they exist
        metadata_fields = ['source', 'url', 'period', 'chapter', 'title']
        for field in metadata_fields:
            if field in record and record[field]:
                processed_record[field] = record[field]
        
        processed_records.append(processed_record)
        stats['valid'] += 1
    
    print(f"✅ Processing complete:")
    print(f"   📊 Valid entries: {stats['valid']}")
    print(f"   ⚠️  Too short: {stats['too_short']}")
    print(f"   ⚠️  Non-Mongolian: {stats['non_mongolian']}")
    print(f"   ⚠️  Duplicates: {stats['duplicates']}")
    
    return processed_records, stats

def save_merged_dataset(records, output_path):
    """Save merged dataset to JSONL file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"💾 Saved merged dataset to {output_path}")

def print_dataset_summary(records, stats, input_files):
    """Print comprehensive dataset summary"""
    if not records:
        print("📊 No records to summarize")
        return
    
    # Calculate statistics
    total_chars = sum(len(r['text']) for r in records)
    total_words = sum(r['word_count'] for r in records)
    avg_chars = total_chars // len(records) if records else 0
    avg_words = total_words // len(records) if records else 0
    
    # Count by dataset source
    source_counts = defaultdict(int)
    for record in records:
        source_counts[record['dataset_source']] += 1
    
    print(f"\n📊 Final Dataset Summary:")
    print(f"   ✅ Files processed: {len(input_files)}")
    print(f"   ✅ Entries merged: {len(records)}")
    print(f"   ✅ Duplicates removed: {stats['duplicates']}")
    print(f"   ✅ Total characters: {total_chars:,}")
    print(f"   ✅ Total words: {total_words:,}")
    print(f"   ✅ Avg text length: {avg_chars} chars ({avg_words} words)")
    
    print(f"\n📋 Entries by source:")
    for source, count in sorted(source_counts.items()):
        percentage = (count / len(records)) * 100
        print(f"   📄 {source}: {count} entries ({percentage:.1f}%)")
    
    # Content length distribution
    lengths = [len(r['text']) for r in records]
    if lengths:
        print(f"\n📈 Content distribution:")
        print(f"   Shortest: {min(lengths):,} characters")
        print(f"   Longest: {max(lengths):,} characters")
        print(f"   Median: {sorted(lengths)[len(lengths)//2]:,} characters")

def main():
    """Main dataset merging function"""
    print("🔗 Starting Mongolian Historical Dataset Merger...")
    
    try:
        # Find input files
        input_files = find_input_files()
        if not input_files:
            print("❌ No input files found to merge")
            return
        
        # Load all records
        all_records = []
        for file_path in input_files:
            file_records = load_jsonl_file(file_path)
            all_records.extend(file_records)
        
        if not all_records:
            print("❌ No records loaded from input files")
            return
        
        # Process and filter records
        processed_records, stats = process_and_filter_records(all_records)
        
        if not processed_records:
            print("❌ No valid records after processing")
            return
        
        # Optional: Shuffle for randomness
        print("🔀 Shuffling dataset for randomness...")
        random.seed(42)  # For reproducible results
        random.shuffle(processed_records)
        
        # Save merged dataset
        output_path = 'data/mgl_history_merged.jsonl'
        save_merged_dataset(processed_records, output_path)
        
        # Print summary
        print_dataset_summary(processed_records, stats, input_files)
        
        print(f"\n🎉 Dataset merging completed successfully!")
        print(f"📁 Merged dataset: {output_path}")
        print(f"🔗 Ready for labeling and indexing")
        
    except Exception as e:
        print(f"❌ Error during merging: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()