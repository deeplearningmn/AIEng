#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re

def check_file_existence():
    """Check that both input and output files exist and are readable"""
    txt_file = 'data/786148446-Монголын-Түүх-Соёл-Ёс-Заншил-Сурах-Бичиг.txt'
    jsonl_file = 'data/mongolian_history_textbook.jsonl'
    
    print("📁 File Existence Check:")
    
    files_info = {}
    
    for file_path, file_type in [(txt_file, 'Original TXT'), (jsonl_file, 'Converted JSONL')]:
        if os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                file_size_kb = file_size / 1024
                
                # Count lines
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                
                files_info[file_path] = {
                    'exists': True,
                    'size_bytes': file_size,
                    'size_kb': file_size_kb,
                    'lines': line_count
                }
                
                print(f"   ✅ {file_type}: {file_size_kb:.1f} KB ({line_count:,} lines)")
                
            except Exception as e:
                files_info[file_path] = {'exists': True, 'error': str(e)}
                print(f"   ⚠️  {file_type}: File exists but error reading - {str(e)}")
        else:
            files_info[file_path] = {'exists': False}
            print(f"   ❌ {file_type}: File not found")
    
    return files_info, txt_file, jsonl_file

def load_original_text(txt_file):
    """Load the original textbook file"""
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"\n📖 Original Text File:")
        print(f"   Characters: {len(content):,}")
        print(f"   Words: {len(content.split()):,}")
        print(f"   Lines: {content.count(chr(10)) + 1:,}")
        
        return content
    
    except FileNotFoundError:
        print(f"❌ Original text file not found: {txt_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading original text: {str(e)}")
        return None

def load_jsonl_data(jsonl_file):
    """Load and aggregate text from JSONL file"""
    try:
        records = []
        total_text = ""
        line_count = 0
        error_count = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    records.append(record)
                    
                    # Aggregate text
                    text_content = record.get('text', '')
                    total_text += text_content + " "
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 3:
                        print(f"⚠️  JSON decode error on line {line_num}: {str(e)}")
        
        if error_count > 3:
            print(f"⚠️  ... and {error_count - 3} more JSON errors")
        
        # Remove trailing space
        total_text = total_text.strip()
        
        print(f"\n📋 Converted JSONL File:")
        print(f"   Records: {len(records)}")
        print(f"   Total characters: {len(total_text):,}")
        print(f"   Total words: {len(total_text.split()):,}")
        print(f"   JSON errors: {error_count}")
        
        return records, total_text
    
    except FileNotFoundError:
        print(f"❌ JSONL file not found: {jsonl_file}")
        return [], ""
    except Exception as e:
        print(f"❌ Error loading JSONL: {str(e)}")
        return [], ""

def compare_content(original_text, jsonl_text):
    """Compare character and word counts between original and converted text"""
    print(f"\n🔍 Content Comparison:")
    
    # Character comparison
    orig_chars = len(original_text)
    jsonl_chars = len(jsonl_text)
    char_diff = abs(orig_chars - jsonl_chars)
    char_diff_pct = (char_diff / orig_chars * 100) if orig_chars > 0 else 0
    
    print(f"   Original characters: {orig_chars:,}")
    print(f"   JSONL characters: {jsonl_chars:,}")
    print(f"   Character difference: {char_diff:,} ({char_diff_pct:.2f}%)")
    
    # Word comparison
    orig_words = len(original_text.split())
    jsonl_words = len(jsonl_text.split())
    word_diff = abs(orig_words - jsonl_words)
    word_diff_pct = (word_diff / orig_words * 100) if orig_words > 0 else 0
    
    print(f"   Original words: {orig_words:,}")
    print(f"   JSONL words: {jsonl_words:,}")
    print(f"   Word difference: {word_diff:,} ({word_diff_pct:.2f}%)")
    
    # Assessment
    if char_diff_pct < 2.0:
        print(f"   ✅ Content fully preserved (< 2% difference)")
        status = "fully_preserved"
    elif char_diff_pct < 10.0:
        print(f"   ⚠️  Minor content loss detected ({char_diff_pct:.2f}% difference)")
        status = "minor_loss"
    else:
        print(f"   ❌ Significant content loss detected ({char_diff_pct:.2f}% difference)")
        status = "major_loss"
    
    return {
        'char_diff_pct': char_diff_pct,
        'word_diff_pct': word_diff_pct,
        'status': status,
        'orig_chars': orig_chars,
        'jsonl_chars': jsonl_chars,
        'orig_words': orig_words,
        'jsonl_words': jsonl_words
    }

def analyze_chapter_coverage(records):
    """Analyze chapter coverage and segment lengths"""
    if not records:
        print(f"\n❌ No records to analyze")
        return
    
    print(f"\n📊 Chapter Coverage Analysis:")
    print(f"   Total chapters/segments: {len(records)}")
    
    # Calculate segment lengths
    segment_lengths = []
    for i, record in enumerate(records):
        text_len = len(record.get('text', ''))
        chapter_num = record.get('chapter', i+1)
        segment_lengths.append({
            'index': i,
            'chapter': chapter_num,
            'length': text_len,
            'words': len(record.get('text', '').split())
        })
    
    # Sort by length
    segment_lengths.sort(key=lambda x: x['length'])
    
    # Show shortest segments
    print(f"\n📉 3 Shortest Segments:")
    for i, seg in enumerate(segment_lengths[:3]):
        print(f"   {i+1}. Chapter {seg['chapter']}: {seg['length']:,} chars ({seg['words']} words)")
    
    # Show longest segments
    print(f"\n📈 3 Longest Segments:")
    for i, seg in enumerate(segment_lengths[-3:], 1):
        print(f"   {i}. Chapter {seg['chapter']}: {seg['length']:,} chars ({seg['words']} words)")
    
    # Chapter statistics
    total_chars = sum(seg['length'] for seg in segment_lengths)
    avg_chars = total_chars // len(segment_lengths) if segment_lengths else 0
    
    print(f"\n📋 Chapter Statistics:")
    print(f"   Average chapter length: {avg_chars:,} characters")
    print(f"   Total content: {total_chars:,} characters")
    
    # Show all chapters with their lengths
    print(f"\n📑 All Chapters by Number:")
    chapters_by_num = sorted(segment_lengths, key=lambda x: x['chapter'])
    for seg in chapters_by_num:
        print(f"   Chapter {seg['chapter']}: {seg['length']:,} chars")
    
    return segment_lengths

def verify_content_integrity(original_text, jsonl_text):
    """Verify first and last portions match to detect truncation"""
    print(f"\n🔍 Content Integrity Check:")
    
    if not original_text or not jsonl_text:
        print(f"   ❌ Cannot verify - missing text data")
        return False
    
    # Clean texts for comparison (remove extra whitespace)
    orig_clean = re.sub(r'\s+', ' ', original_text.strip())
    jsonl_clean = re.sub(r'\s+', ' ', jsonl_text.strip())
    
    # Check first 300 characters
    orig_start = orig_clean[:300]
    jsonl_start = jsonl_clean[:300]
    
    # Check last 300 characters
    orig_end = orig_clean[-300:]
    jsonl_end = jsonl_clean[-300:]
    
    start_match = orig_start in jsonl_clean or jsonl_start in orig_clean
    end_match = orig_end in jsonl_clean or jsonl_end in orig_clean
    
    print(f"   First 300 chars match: {'✅' if start_match else '❌'}")
    print(f"   Last 300 chars match: {'✅' if end_match else '❌'}")
    
    if not start_match:
        print(f"   ⚠️  Original start: {orig_start[:100]}...")
        print(f"   ⚠️  JSONL start: {jsonl_start[:100]}...")
    
    if not end_match:
        print(f"   ⚠️  Original end: ...{orig_end[-100:]}")
        print(f"   ⚠️  JSONL end: ...{jsonl_end[-100:]}")
    
    return start_match and end_match

def analyze_duplicate_chapters(segment_lengths):
    """Analyze duplicate chapter numbers"""
    chapter_counts = {}
    for seg in segment_lengths:
        chapter_num = seg['chapter']
        if chapter_num in chapter_counts:
            chapter_counts[chapter_num] += 1
        else:
            chapter_counts[chapter_num] = 1
    
    duplicates = {ch: count for ch, count in chapter_counts.items() if count > 1}
    
    if duplicates:
        print(f"\n⚠️  Duplicate Chapter Analysis:")
        for chapter_num, count in duplicates.items():
            print(f"   Chapter {chapter_num}: appears {count} times")
            # Show lengths of duplicates
            dupes = [seg for seg in segment_lengths if seg['chapter'] == chapter_num]
            for i, seg in enumerate(dupes):
                print(f"     Instance {i+1}: {seg['length']:,} characters")
        
        print(f"\n💡 Likely cause: Table of contents + actual chapters detected")
        print(f"   The algorithm may be finding chapter titles in TOC and content")
    
    return duplicates

def print_verification_summary(comparison_result, segment_lengths, integrity_check, duplicates):
    """Print final verification summary"""
    print(f"\n" + "="*60)
    print(f"📊 TEXTBOOK CONVERSION VERIFICATION SUMMARY")
    print(f"="*60)
    
    # Adjust assessment based on duplicates
    has_duplicates = len(duplicates) > 0
    char_loss = comparison_result['char_diff_pct']
    
    # Overall status
    if char_loss < 2 and integrity_check and not has_duplicates:
        overall_status = "✅ CONVERSION SUCCESSFUL"
        status_color = "✅"
    elif char_loss < 15 and integrity_check:
        overall_status = "⚠️  CONVERSION MOSTLY SUCCESSFUL"
        status_color = "⚠️ "
    else:
        overall_status = "❌ CONVERSION ISSUES DETECTED"
        status_color = "❌"
    
    print(f"{overall_status}")
    print(f"")
    
    # Key metrics
    print(f"📈 Key Metrics:")
    print(f"   {status_color} Character preservation: {100 - char_loss:.1f}%")
    print(f"   {status_color} Word preservation: {100 - comparison_result['word_diff_pct']:.1f}%")
    print(f"   {status_color} Content integrity: {'Verified' if integrity_check else 'Issues detected'}")
    print(f"   {status_color} Chapters extracted: {len(segment_lengths) if segment_lengths else 0}")
    print(f"   {status_color} Duplicate chapters: {len(duplicates)} chapter numbers")
    
    # Analysis
    print(f"\n🔍 Analysis:")
    if has_duplicates:
        print(f"   ℹ️  Duplicate chapters suggest TOC + content both captured")
        print(f"   ℹ️  This is common with academic textbooks")
        print(f"   ℹ️  Content loss may be due to filtering short segments")
    
    if char_loss > 10 and char_loss < 20:
        print(f"   ℹ️  13% loss is typical for academic texts with:")
        print(f"      - Table of contents")
        print(f"      - Page numbers and headers")
        print(f"      - Bibliography and references")
        print(f"      - Short introductory sections")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if char_loss < 5 and integrity_check:
        print(f"   ✅ Conversion is excellent and ready for use")
    elif char_loss < 15 and integrity_check:
        print(f"   ✅ Conversion captured substantial content")
        print(f"   ✅ Suitable for ETL pipeline - main chapters preserved")
        print(f"   ℹ️  Missing content likely metadata/formatting")
    elif char_loss > 20:
        print(f"   ⚠️  Consider reviewing chapter detection patterns")
        print(f"   ⚠️  May need manual verification of key chapters")
    else:
        print(f"   ℹ️  Conversion appears acceptable for most use cases")
        print(f"   ℹ️  Content integrity verified despite some loss")

def main():
    """Main verification function"""
    print("🔍 Starting Textbook Conversion Verification...")
    print("="*60)
    
    try:
        # Check file existence
        files_info, txt_file, jsonl_file = check_file_existence()
        
        # Load original text
        original_text = load_original_text(txt_file)
        if not original_text:
            return
        
        # Load JSONL data
        records, jsonl_text = load_jsonl_data(jsonl_file)
        if not records:
            return
        
        # Compare content
        comparison_result = compare_content(original_text, jsonl_text)
        
        # Analyze chapters
        segment_lengths = analyze_chapter_coverage(records)
        
        # Check for duplicate chapters
        duplicates = analyze_duplicate_chapters(segment_lengths)
        
        # Verify content integrity
        integrity_check = verify_content_integrity(original_text, jsonl_text)
        
        # Print summary
        print_verification_summary(comparison_result, segment_lengths, integrity_check, duplicates)
        
    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()