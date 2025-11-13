#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import hashlib
from langdetect import detect, DetectorFactory
import os

# Set seed for consistent language detection
DetectorFactory.seed = 0

def load_raw_data():
    """Load raw scraped data from JSONL file"""
    docs = []
    try:
        with open('data/raw_data.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        print(f"📥 Loaded {len(docs)} raw documents")
        return docs
    except FileNotFoundError:
        print("❌ raw_data.jsonl not found. Run crawl.py first.")
        return []

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def is_mongolian_text(text):
    """Check if text is primarily Mongolian using language detection"""
    if not text or len(text.strip()) < 10:
        return False
    
    try:
        detected_lang = detect(text)
        # Mongolian might be detected as 'mn' or sometimes misclassified
        # We'll also check for Cyrillic characters which are common in Mongolian
        cyrillic_chars = len(re.findall(r'[а-яё]', text.lower()))
        total_chars = len(re.findall(r'[а-яёa-z]', text.lower()))
        
        if total_chars > 0:
            cyrillic_ratio = cyrillic_chars / total_chars
            # If more than 30% cyrillic characters, likely Mongolian
            return detected_lang == 'mn' or cyrillic_ratio > 0.3
        
        return detected_lang == 'mn'
    except:
        # If detection fails, check for Cyrillic characters
        cyrillic_chars = len(re.findall(r'[а-яё]', text.lower()))
        return cyrillic_chars > 10

def get_text_hash(text):
    """Generate hash for duplicate detection"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def main():
    """Main cleaning function"""
    print("🧹 Starting text cleaning...")
    
    # Load raw data
    raw_docs = load_raw_data()
    if not raw_docs:
        return
    
    cleaned_docs = []
    seen_hashes = set()
    
    for doc in raw_docs:
        # Clean title and text
        clean_title = clean_text(doc.get('title', ''))
        clean_content = clean_text(doc.get('text', ''))
        
        # Skip if no meaningful content
        if not clean_content or len(clean_content.split()) < 10:
            continue
        
        # Check if text is Mongolian
        if not is_mongolian_text(clean_content):
            print(f"⚠️  Skipping non-Mongolian text: {clean_title[:50]}...")
            continue
        
        # Check for duplicates
        content_hash = get_text_hash(clean_content)
        if content_hash in seen_hashes:
            print(f"⚠️  Skipping duplicate: {clean_title[:50]}...")
            continue
        
        seen_hashes.add(content_hash)
        
        # Create cleaned document
        cleaned_doc = {
            'url': doc.get('url', ''),
            'title': clean_title,
            'text': clean_content,
            'word_count': len(clean_content.split()),
            'hash': content_hash
        }
        
        cleaned_docs.append(cleaned_doc)
    
    # Save cleaned data
    os.makedirs('data', exist_ok=True)
    output_file = 'data/cleaned_data.jsonl'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in cleaned_docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Done cleaning! {len(cleaned_docs)} documents remain")
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    main()