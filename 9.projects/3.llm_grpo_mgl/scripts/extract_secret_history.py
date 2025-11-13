#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
from io import StringIO

def extract_text_pdfminer(pdf_path):
    """Extract text using pdfminer.six (primary method)"""
    try:
        from pdfminer.high_level import extract_text
        print("📖 Using pdfminer.six for text extraction...")
        text = extract_text(pdf_path)
        return text
    except ImportError:
        print("⚠️  pdfminer.six not available")
        return None
    except Exception as e:
        print(f"⚠️  pdfminer.six failed: {str(e)}")
        return None

def extract_text_pymupdf(pdf_path):
    """Extract text using PyMuPDF (fallback method)"""
    try:
        import fitz  # PyMuPDF
        print("📖 Using PyMuPDF (fitz) for text extraction...")
        
        doc = fitz.open(pdf_path)
        text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        doc.close()
        return text
    except ImportError:
        print("⚠️  PyMuPDF not available")
        return None
    except Exception as e:
        print(f"⚠️  PyMuPDF failed: {str(e)}")
        return None

def clean_text(text):
    """Advanced text cleaning for PDF content"""
    if not text:
        return ""
    
    # Remove invisible characters and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remove zero-width characters and non-breaking spaces
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0]', ' ', text)
    
    # Remove soft hyphens and various dash types used for line breaks
    text = re.sub(r'[\u00ad\u2010-\u2015\u2212]', '', text)
    
    # Fix broken words across line breaks (Mongolian Cyrillic)
    text = re.sub(r'([а-яёүө])\n+([а-яёүө])', r'\1\2', text, flags=re.IGNORECASE)
    
    # Fix broken words with hyphens
    text = re.sub(r'([а-яёүө])-\n+([а-яёүө])', r'\1\2', text, flags=re.IGNORECASE)
    
    # Replace multiple whitespace with single space
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Normalize line breaks - replace multiple newlines with double newline
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Clean up spaces around punctuation
    text = re.sub(r'\s+([,.!?;:)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()

def split_into_segments(text):
    """Split text into numbered segments with smart fallback"""
    segments = []
    
    # Primary method: Split by numbered sections (1., 2., 3., etc.)
    section_pattern = r'(?:^|\n)\s*(\d+)\.\s+'
    parts = re.split(section_pattern, text, flags=re.MULTILINE)
    
    # Process numbered sections
    if len(parts) > 2:  # We have actual numbered sections
        print("✂️  Found numbered sections, splitting by chapter numbers...")
        
        # Skip the first part (content before first number)
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                chapter_num = int(parts[i])
                chapter_text = parts[i + 1].strip()
                
                # Only include segments with sufficient content
                if len(chapter_text) >= 100:
                    segments.append({
                        'chapter': chapter_num,
                        'text': chapter_text
                    })
    
    # Fallback method: Split by paragraphs if no numbered sections
    if not segments:
        print("⚠️  No numbered sections found, using paragraph-based splitting...")
        
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        chapter_num = 1
        
        for paragraph in paragraphs:
            clean_paragraph = paragraph.strip()
            if len(clean_paragraph) >= 100:
                segments.append({
                    'chapter': chapter_num,
                    'text': clean_paragraph
                })
                chapter_num += 1
    
    # Alternative fallback: Split by sentence groups if still no segments
    if not segments:
        print("⚠️  Paragraph splitting failed, using sentence-based splitting...")
        
        # Split by periods followed by whitespace and capital letters
        sentences = re.split(r'\.(?=\s+[А-ЯҮӨЁ])', text)
        chapter_num = 1
        current_segment = ""
        
        for sentence in sentences:
            current_segment += sentence.strip() + ". "
            
            # If segment is long enough, save it
            if len(current_segment) >= 200:
                segments.append({
                    'chapter': chapter_num,
                    'text': current_segment.strip()
                })
                chapter_num += 1
                current_segment = ""
        
        # Add remaining content if any
        if current_segment.strip() and len(current_segment.strip()) >= 100:
            segments.append({
                'chapter': chapter_num,
                'text': current_segment.strip()
            })
    
    return segments

def extract_pdf_content(pdf_path):
    """Main function to extract and process PDF content"""
    print(f"🔍 Processing PDF: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return []
    
    # Try pdfminer.six first, then PyMuPDF as fallback
    raw_text = extract_text_pdfminer(pdf_path)
    
    if not raw_text or len(raw_text.strip()) < 100:
        print("🔄 Trying fallback method...")
        raw_text = extract_text_pymupdf(pdf_path)
    
    if not raw_text:
        print("❌ Failed to extract text with both methods")
        return []
    
    print(f"📄 Extracted {len(raw_text)} characters of raw text")
    
    # Clean the text
    cleaned_text = clean_text(raw_text)
    print(f"🧹 Cleaned text: {len(cleaned_text)} characters")
    
    if len(cleaned_text) < 100:
        print("⚠️  Cleaned text too short, extraction may have failed")
        return []
    
    # Split into segments
    segments = split_into_segments(cleaned_text)
    print(f"✂️  Split into {len(segments)} segments")
    
    return segments

def create_structured_records(segments):
    """Convert segments into structured JSONL records"""
    records = []
    
    for segment in segments:
        record = {
            'source': 'Монголын нууц товчоо',
            'chapter': segment['chapter'],
            'text': segment['text'],
            'period': 'XIII зуун'
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
    min_chapter = min(r['chapter'] for r in records)
    max_chapter = max(r['chapter'] for r in records)
    
    print("\n📊 Extraction Statistics:")
    print(f"   Total segments: {len(records)}")
    print(f"   Chapter range: {min_chapter} - {max_chapter}")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Average segment length: {avg_chars} characters")
    print(f"   Shortest segment: {min(len(r['text']) for r in records)} characters")
    print(f"   Longest segment: {max(len(r['text']) for r in records)} characters")

def main():
    """Main extraction function"""
    print("🏛️  Starting Secret History of the Mongols extraction...")
    
    # Define paths
    pdf_path = 'data/Монголын_нууц_товчоо.pdf'
    output_path = 'data/secret_history.jsonl'
    
    try:
        # Extract content from PDF
        segments = extract_pdf_content(pdf_path)
        
        if not segments:
            print("❌ No segments extracted from PDF")
            return
        
        # Create structured records
        records = create_structured_records(segments)
        
        # Save to JSONL
        save_to_jsonl(records, output_path)
        
        # Print detailed statistics
        print_extraction_stats(records)
        
        print(f"\n✅ Done! Successfully extracted Secret History content")
        print(f"🔗 Ready for integration with ETL pipeline")
        
    except Exception as e:
        print(f"❌ Error during extraction: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()