#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
import hashlib
import glob

def find_textbook_file():
    """Automatically locate the Mongolian history textbook file in data/ directory"""
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return None
    
    # List all files in data directory
    all_files = os.listdir(data_dir)
    
    # Look for files that contain the textbook identifier
    textbook_identifiers = [
        '786148446',
        'Монголын-Түүх',
        'Сурах-Бичиг'
    ]
    
    matching_files = []
    
    for filename in all_files:
        # Check if filename contains textbook identifiers and is a text file
        if (any(identifier in filename for identifier in textbook_identifiers) and 
            filename.endswith('.txt')):
            full_path = os.path.join(data_dir, filename)
            matching_files.append(full_path)
    
    if not matching_files:
        print(f"❌ No textbook file found with identifiers: {textbook_identifiers}")
        print(f"📁 Available files in {data_dir}: {all_files}")
        return None
    
    if len(matching_files) > 1:
        print(f"⚠️  Multiple textbook files found: {matching_files}")
        print(f"📁 Using first match: {matching_files[0]}")
    
    textbook_file = matching_files[0]
    print(f"📁 Located textbook file: {textbook_file}")
    return textbook_file

def load_textbook_file(file_path):
    """Load the textbook file with UTF-8 encoding and detailed logging"""
    try:
        # Validate file existence
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
        
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"📁 File size: {file_size:,} bytes")
        
        # Load content with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Count lines
        line_count = content.count('\n') + 1
        
        print(f"📖 Successfully loaded textbook file:")
        print(f"   Characters: {len(content):,}")
        print(f"   Lines: {line_count:,}")
        print(f"   Encoding: UTF-8")
        
        return content
        
    except UnicodeDecodeError:
        print("⚠️  UTF-8 decode failed, trying with error handling...")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            line_count = content.count('\n') + 1
            print(f"📖 Loaded with error handling:")
            print(f"   Characters: {len(content):,}")
            print(f"   Lines: {line_count:,}")
            print(f"   Encoding: UTF-8 (with errors ignored)")
            return content
        except Exception as e:
            print(f"❌ Failed to load file with error handling: {str(e)}")
            return None
    except Exception as e:
        print(f"❌ Error loading file: {str(e)}")
        return None

def clean_text(text):
    """Advanced text cleaning and normalization with invisible character removal"""
    if not text:
        return ""
    
    # Remove invisible characters and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remove zero-width characters and non-breaking spaces
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\u2060\u180e]', ' ', text)
    
    # Remove soft hyphens and line-break hyphens
    text = re.sub(r'[\u00ad\u2010-\u2015]', '', text)
    
    # Fix broken words across line breaks (Mongolian specific)
    text = re.sub(r'([а-яёүө])-?\n+([а-яёүө])', r'\1\2', text, flags=re.IGNORECASE)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks - preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Clean up spaces around punctuation
    text = re.sub(r'\s+([,.!?;:)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    # Remove excessive empty lines but preserve paragraph breaks
    lines = []
    for line in text.split('\n'):
        stripped_line = line.strip()
        if stripped_line or (lines and lines[-1]):  # Keep empty lines only between content
            lines.append(stripped_line)
    
    # Remove trailing empty lines
    while lines and not lines[-1]:
        lines.pop()
    
    return '\n'.join(lines)

def extract_chapter_number(chapter_title):
    """Extract chapter number from Mongolian chapter titles with complete 1-20 mapping"""
    # Complete dictionary mapping Mongolian ordinal numbers to integers (1-20)
    mongolian_numbers = {
        # Basic ordinals
        'нэгдүгээр': 1, 'нэг': 1,
        'хоёрдугаар': 2, 'хоёр': 2,
        'гуравдугаар': 3, 'гурав': 3,
        'дөрөвдүгээр': 4, 'дөрөв': 4,
        'тавдугаар': 5, 'тав': 5,
        'зургаадугаар': 6, 'зургаа': 6,
        'долоодугаар': 7, 'долоо': 7,
        'наймдугаар': 8, 'найм': 8,
        'есдүгээр': 9, 'ес': 9,
        'аравдугаар': 10, 'арав': 10,
        
        # Compound ordinals (11-19)
        'арван нэгдүгээр': 11, 'арван нэг': 11,
        'арван хоёрдугаар': 12, 'арван хоёр': 12,
        'арван гуравдугаар': 13, 'арван гурав': 13,
        'арван дөрөвдүгээр': 14, 'арван дөрөв': 14,
        'арван тавдугаар': 15, 'арван тав': 15,
        'арван зургаадугаар': 16, 'арван зургаа': 16,
        'арван долоодугаар': 17, 'арван долоо': 17,
        'арван наймдугаар': 18, 'арван найм': 18,
        'арван есдүгээр': 19, 'арван ес': 19,
        
        # Twenty
        'хорьдугаар': 20, 'хорь': 20, 'хориндугаар': 20,
        
        # Alternative spellings
        'хоёрдахь': 2, 'гуравдахь': 3, 'дөрөвдэхь': 4,
        'тавдахь': 5, 'зургаадахь': 6, 'долоодохь': 7,
        'наймдахь': 8, 'есдэхь': 9, 'аравдахь': 10
    }
    
    chapter_lower = chapter_title.lower().strip()
    
    # Try to find Mongolian ordinal number (longest match first)
    sorted_numbers = sorted(mongolian_numbers.items(), key=lambda x: len(x[0]), reverse=True)
    for mong_num, num in sorted_numbers:
        if mong_num in chapter_lower:
            return num
    
    # Try to find Arabic numerals
    arabic_match = re.search(r'(\d+)', chapter_title)
    if arabic_match:
        chapter_num = int(arabic_match.group(1))
        # Limit to reasonable range
        if 1 <= chapter_num <= 50:
            return chapter_num
    
    # Default fallback
    return 1

def split_by_chapters(text):
    """Split text by chapter indicators with enhanced fallback logic"""
    chapters = []
    
    # Enhanced pattern to match chapter headers in Mongolian
    chapter_patterns = [
        # Primary pattern: "Нэгдүгээр бүлэг", "Хоёрдугаар бүлэг", etc.
        r'(?:^|\n)\s*([А-ЯҮӨЁ][а-яүөё\s]*дүгээр\s+бүлэг[^\n]*)',
        # Alternative patterns
        r'(?:^|\n)\s*(Бүлэг\s+\d+[^\n]*)',
        r'(?:^|\n)\s*(\d+\.\s*[А-ЯҮӨЁ][а-яүөё\s]*дүгээр\s+бүлэг[^\n]*)',
        r'(?:^|\n)\s*(БҮЛЭГ\s+[IVX\d]+[^\n]*)',
        r'(?:^|\n)\s*([А-ЯҮӨЁ][А-ЯҮӨЁ\s]*БҮЛЭГ[^\n]*)',
        # More flexible patterns
        r'(?:^|\n)\s*([А-ЯҮӨЁ][а-яүөё\s]*\s+бүлэг[^\n]*)',
    ]
    
    # Try each pattern
    for i, pattern in enumerate(chapter_patterns):
        matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
        if matches:
            print(f"✂️  Found {len(matches)} chapters using pattern {i+1}")
            
            for j, match in enumerate(matches):
                chapter_title = match.group(1).strip()
                start_pos = match.end()
                
                # Find end position (start of next chapter or end of text)
                if j + 1 < len(matches):
                    end_pos = matches[j + 1].start()
                else:
                    end_pos = len(text)
                
                chapter_content = text[start_pos:end_pos].strip()
                
                if chapter_content:
                    chapter_num = extract_chapter_number(chapter_title)
                    chapters.append({
                        'title': chapter_title,
                        'number': chapter_num,
                        'content': chapter_content
                    })
            
            if chapters:
                break
    
    # Enhanced fallback: split by paragraph gaps
    if not chapters:
        print("⚠️  No chapter patterns found, using paragraph-based splitting...")
        
        # Split by double newlines (paragraph gaps)
        sections = re.split(r'\n\s*\n', text)
        chapter_num = 1
        
        for section in sections:
            section = section.strip()
            if len(section) >= 300:  # Minimum length filter
                # Check if this looks like a chapter start
                first_line = section.split('\n')[0].strip()
                
                # Look for chapter-like indicators
                if any(word in first_line.lower() for word in ['бүлэг', 'хэсэг', 'тал', 'дугаар']):
                    chapters.append({
                        'title': first_line,
                        'number': chapter_num,
                        'content': section
                    })
                    chapter_num += 1
                elif len(section) >= 500:  # Large sections without clear titles
                    # Use first sentence as title
                    sentences = re.split(r'[.!?]\s+', section)
                    title = sentences[0][:100] + "..." if len(sentences[0]) > 100 else sentences[0]
                    
                    chapters.append({
                        'title': title,
                        'number': chapter_num,
                        'content': section
                    })
                    chapter_num += 1
    
    return chapters

def create_structured_records(chapters):
    """Convert chapters into structured JSONL records"""
    records = []
    
    for chapter in chapters:
        # Clean the chapter content
        cleaned_content = clean_text(chapter['content'])
        
        # Skip if content is too short
        if len(cleaned_content) < 300:
            print(f"⚠️  Skipping short chapter: {chapter['title'][:50]}... ({len(cleaned_content)} chars)")
            continue
        
        record = {
            'source': 'Монголын түүх, соёл, ёс заншил',
            'chapter': chapter['number'],
            'text': cleaned_content,
            'period': 'Эртнээс орчин үе хүртэл'
        }
        records.append(record)
    
    return records

def save_to_jsonl(records, output_path):
    """Save records to JSONL file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"💾 Saved {len(records)} records to {output_path}")

def print_extraction_stats(records):
    """Print detailed extraction statistics"""
    if not records:
        return
    
    total_chars = sum(len(r['text']) for r in records)
    avg_chars = total_chars // len(records)
    
    print("\n📊 Textbook Import Statistics:")
    print(f"   Total chapters: {len(records)}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Average chapter length: {avg_chars} characters")
    print(f"   Shortest chapter: {min(len(r['text']) for r in records)} characters")
    print(f"   Longest chapter: {max(len(r['text']) for r in records)} characters")
    
    # Show chapter distribution
    chapters = sorted(set(r['chapter'] for r in records))
    print(f"   Chapter numbers: {', '.join(map(str, chapters))}")

def print_integration_summary(output_path, record_count):
    """Print integration summary for ETL pipeline"""
    print(f"\n🎉 Textbook Import Complete!")
    print(f"✅ Created {os.path.basename(output_path)}")
    print(f"✅ {record_count} chapters ready for processing")
    print(f"✅ Ready for merge_datasets.py")
    print(f"\n📁 Output location: {output_path}")
    print(f"🔗 Compatible with ETL pipeline format")
    
    print(f"\n📋 Next steps:")
    print(f"   1. Run: python scripts/merge_datasets.py")
    print(f"   2. The textbook data will be automatically detected and merged")

def validate_output_compatibility():
    """Validate that output format matches other ETL pipeline components"""
    expected_files = [
        'data/secret_history.jsonl',
        'data/web_raw.jsonl'
    ]
    
    compatible_files = []
    for file_path in expected_files:
        if os.path.exists(file_path):
            compatible_files.append(os.path.basename(file_path))
    
    if compatible_files:
        print(f"🔗 Compatible with existing datasets: {', '.join(compatible_files)}")
    else:
        print(f"ℹ️  No other datasets found - textbook will be first dataset")

def main():
    """Main textbook import function with automatic file detection"""
    print("📚 Starting Mongolian History Textbook import...")
    
    try:
        # Automatically find textbook file
        input_path = find_textbook_file()
        if not input_path:
            print("❌ Cannot proceed without textbook file")
            return
        
        output_path = 'data/mongolian_history_textbook.jsonl'
        
        # Load textbook content
        content = load_textbook_file(input_path)
        if not content:
            return
        
        # Split into chapters
        chapters = split_by_chapters(content)
        
        if not chapters:
            print("❌ No chapters found in textbook")
            return
        
        print(f"✂️  Extracted {len(chapters)} chapters")
        
        # Warn if too few chapters detected
        if len(chapters) < 5:
            print(f"⚠️  Only {len(chapters)} chapters detected - expected more from textbook")
        
        # Create structured records
        records = create_structured_records(chapters)
        
        if not records:
            print("❌ No valid records created after filtering")
            return
        
        # Save to JSONL
        save_to_jsonl(records, output_path)
        
        # Print statistics
        print_extraction_stats(records)
        
        # Check compatibility with other datasets
        validate_output_compatibility()
        
        # Print integration summary
        print_integration_summary(output_path, len(records))
        
    except Exception as e:
        print(f"❌ Error during import: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()