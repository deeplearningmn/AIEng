#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
from collections import defaultdict

def load_textbook_jsonl(file_path):
    """Load textbook JSONL file safely with error handling"""
    print(f"📖 Loading textbook dataset: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    records = []
    error_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"⚠️  JSON decode error on line {line_num}: {str(e)}")
        
        if error_count > 3:
            print(f"⚠️  ... and {error_count - 3} more JSON errors")
        
        print(f"✅ Loaded {len(records)} records")
        return records
    
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return []

def is_mongolian_text(text):
    """Check if text contains significant Mongolian Cyrillic content"""
    if not text or len(text.strip()) < 20:
        return False
    
    # Count Cyrillic characters
    cyrillic_chars = len(re.findall(r'[а-яёүө]', text.lower()))
    total_chars = len(re.findall(r'[а-яёүөa-z]', text.lower()))
    
    if total_chars == 0:
        return False
    
    # If more than 5% cyrillic characters, likely Mongolian
    cyrillic_ratio = cyrillic_chars / total_chars
    return cyrillic_ratio > 0.05

def is_table_of_contents(text):
    """Detect if text is likely a table of contents or header"""
    text_lower = text.lower().strip()
    
    # If text is very short, it might be TOC
    if len(text) < 1000:
        # Check for explicit TOC indicators at the start
        if text_lower.startswith(('агуулга', 'contents', 'гарчиг')):
            return True
        
        # Check if it's mostly numbers and short phrases
        numbers = len(re.findall(r'\d+', text))
        words = len(text.split())
        if words > 0 and numbers / words > 0.4:  # More than 40% numbers
            return True
    
    # Check for patterns like "1.1. Title ... 8" (TOC format)
    toc_pattern_matches = len(re.findall(r'\d+\.\d+\..*\d+\s*$', text, re.MULTILINE))
    lines = text.split('\n')
    if len(lines) > 5 and toc_pattern_matches > len(lines) * 0.3:
        return True
    
    # Check if text has many short lines with numbers (typical TOC)
    short_lines_with_numbers = sum(1 for line in lines 
                                  if len(line.strip()) < 80 and re.search(r'\d+', line))
    if len(lines) > 10 and short_lines_with_numbers / len(lines) > 0.6:
        return True
    
    # Check for excessive use of dots (common in TOC)
    dot_density = text.count('.') / len(text) if len(text) > 0 else 0
    if dot_density > 0.02 and len(text) < 2000:  # More than 2% dots in short text
        return True
    
    return False

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove invisible characters and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remove zero-width characters and non-breaking spaces
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\u2060\u180e]', ' ', text)
    
    # Remove soft hyphens
    text = re.sub(r'[\u00ad]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Collapse multiple blank lines into single blank line
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Clean up spaces around punctuation
    text = re.sub(r'\s+([,.!?;:)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    
    return text.strip()

def deduplicate_chapters(records):
    """Remove duplicate chapters, keeping the longest version"""
    print(f"\n🔍 Deduplicating chapters...")
    
    # Group records by chapter number
    chapters = defaultdict(list)
    for record in records:
        chapter_num = record.get('chapter', 0)
        chapters[chapter_num].append(record)
    
    deduplicated = []
    duplicates_removed = 0
    
    for chapter_num, chapter_records in chapters.items():
        if len(chapter_records) == 1:
            # No duplicates for this chapter
            deduplicated.append(chapter_records[0])
        else:
            # Multiple records for same chapter - keep the longest
            longest_record = max(chapter_records, key=lambda r: len(r.get('text', '')))
            deduplicated.append(longest_record)
            duplicates_removed += len(chapter_records) - 1
            
            print(f"   📋 Chapter {chapter_num}: kept longest ({len(longest_record.get('text', '')):,} chars), removed {len(chapter_records) - 1} duplicates")
    
    print(f"✅ Removed {duplicates_removed} duplicate chapters")
    return deduplicated

def filter_short_and_toc(records):
    """Filter out short segments and table of contents"""
    print(f"\n🧹 Filtering short segments and TOC...")
    
    filtered_records = []
    removed_short = 0
    removed_toc = 0
    removed_non_mongolian = 0
    
    for record in records:
        text = record.get('text', '')
        chapter_num = record.get('chapter', 0)
        
        # Check text length
        if len(text) < 2000:
            removed_short += 1
            print(f"   ⚠️  Removed Chapter {chapter_num}: too short ({len(text)} chars)")
            continue
        
        # Check if it's table of contents
        if is_table_of_contents(text):
            removed_toc += 1
            print(f"   ⚠️  Removed Chapter {chapter_num}: detected as TOC")
            # Debug: show first 200 chars
            print(f"      Preview: {text[:200]}...")
            continue
        
        # Check if it's Mongolian text
        if not is_mongolian_text(text):
            removed_non_mongolian += 1
            print(f"   ⚠️  Removed Chapter {chapter_num}: insufficient Mongolian content")
            continue
        
        # Passed all filters
        print(f"   ✅ Kept Chapter {chapter_num}: {len(text):,} chars")
        filtered_records.append(record)
    
    print(f"✅ Filtering complete:")
    print(f"   📊 Kept: {len(filtered_records)} chapters")
    print(f"   ⚠️  Removed short: {removed_short}")
    print(f"   ⚠️  Removed TOC: {removed_toc}")
    print(f"   ⚠️  Removed non-Mongolian: {removed_non_mongolian}")
    
    return filtered_records

def normalize_records(records):
    """Normalize text content in all records"""
    print(f"\n🔧 Normalizing text content...")
    
    normalized_records = []
    
    for record in records:
        # Create a copy of the record
        normalized_record = record.copy()
        
        # Clean the text
        original_text = record.get('text', '')
        cleaned_text = clean_text(original_text)
        normalized_record['text'] = cleaned_text
        
        # Update word count
        normalized_record['word_count'] = len(cleaned_text.split())
        
        normalized_records.append(normalized_record)
    
    print(f"✅ Normalized {len(normalized_records)} records")
    return normalized_records

def print_chapter_statistics(records):
    """Print detailed statistics for each chapter"""
    print(f"\n📊 Chapter Statistics:")
    
    if not records:
        print(f"   ❌ No chapters to analyze")
        return 0, 0
    
    # Sort by chapter number
    sorted_records = sorted(records, key=lambda r: r.get('chapter', 0))
    
    total_chars = 0
    total_words = 0
    
    print(f"   {'Chapter':<8} {'Characters':<12} {'Words':<8} {'Title':<50}")
    print(f"   {'-'*8} {'-'*12} {'-'*8} {'-'*50}")
    
    for record in sorted_records:
        chapter_num = record.get('chapter', 0)
        text = record.get('text', '')
        char_count = len(text)
        word_count = record.get('word_count', len(text.split()))
        title = record.get('title', record.get('source', ''))[:47] + "..." if len(record.get('title', record.get('source', ''))) > 50 else record.get('title', record.get('source', ''))
        
        total_chars += char_count
        total_words += word_count
        
        print(f"   {chapter_num:<8} {char_count:<12,} {word_count:<8,} {title:<50}")
    
    avg_chars = total_chars // len(records) if records else 0
    avg_words = total_words // len(records) if records else 0
    
    print(f"   {'-'*8} {'-'*12} {'-'*8} {'-'*50}")
    print(f"   {'TOTAL':<8} {total_chars:<12,} {total_words:<8,} {'Average per chapter':<50}")
    print(f"   {'AVG':<8} {avg_chars:<12,} {avg_words:<8,}")
    
    return total_chars, total_words

def save_refined_dataset(records, output_path):
    """Save refined dataset to JSONL file"""
    print(f"\n💾 Saving refined dataset...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Saved {len(records)} refined chapters to {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
        return False

def print_refinement_summary(original_count, final_count, total_chars, original_chars=None):
    """Print comprehensive refinement summary"""
    print(f"\n" + "="*60)
    print(f"📊 TEXTBOOK DATASET REFINEMENT SUMMARY")
    print(f"="*60)
    
    print(f"📈 Processing Results:")
    print(f"   📋 Original chapters: {original_count}")
    print(f"   ✅ Refined chapters: {final_count}")
    print(f"   📉 Chapters removed: {original_count - final_count}")
    print(f"   📊 Retention rate: {(final_count / original_count * 100):.1f}%")
    
    print(f"\n📝 Content Statistics:")
    print(f"   ✅ Total characters preserved: {total_chars:,}")
    if original_chars:
        preservation_rate = (total_chars / original_chars * 100)
        print(f"   📊 Content preservation: {preservation_rate:.1f}%")
    
    print(f"\n🎯 Quality Improvements:")
    print(f"   ✅ Removed duplicate chapters")
    print(f"   ✅ Filtered out table of contents")
    print(f"   ✅ Removed short fragments")
    print(f"   ✅ Normalized text formatting")
    print(f"   ✅ Verified Mongolian content")
    
    if final_count < 4:
        print(f"\n⚠️  Warning: Only {final_count} chapters retained")
        print(f"   Consider reviewing filtering criteria")
    else:
        print(f"\n🎉 Refinement successful!")
        print(f"   📁 Ready for ETL pipeline integration")

def main():
    """Main refinement function"""
    print("🔧 Starting Textbook Dataset Refinement...")
    print("="*60)
    
    try:
        # Define file paths
        input_file = 'data/mongolian_history_textbook.jsonl'
        output_file = 'data/mongolian_history_textbook_refined.jsonl'
        
        # Load original dataset
        original_records = load_textbook_jsonl(input_file)
        if not original_records:
            print("❌ No records to process")
            return
        
        original_count = len(original_records)
        original_chars = sum(len(r.get('text', '')) for r in original_records)
        
        # Step 1: Deduplicate chapters
        deduplicated_records = deduplicate_chapters(original_records)
        
        # Step 2: Filter short segments and TOC
        filtered_records = filter_short_and_toc(deduplicated_records)
        
        # Step 3: Normalize text content
        normalized_records = normalize_records(filtered_records)
        
        # Step 4: Print statistics
        total_chars, total_words = print_chapter_statistics(normalized_records)
        
        # Step 5: Save refined dataset
        if save_refined_dataset(normalized_records, output_file):
            # Print summary
            print_refinement_summary(original_count, len(normalized_records), total_chars, original_chars)
        else:
            print("❌ Failed to save refined dataset")
    
    except Exception as e:
        print(f"❌ Error during refinement: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()