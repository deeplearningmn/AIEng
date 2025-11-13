#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import yaml
import json
import os
from urllib.parse import urljoin, urlparse
import time

def load_sources():
    """Load URLs from sources.yaml"""
    with open('sources.yaml', 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data.get('urls', [])

def extract_text(url):
    """Extract title and paragraph text from a URL"""
    try:
        print(f"📥 Crawling: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        
        # Extract all paragraph text
        paragraphs = soup.find_all('p')
        text_content = []
        
        for p in paragraphs:
            # Remove nested tags but keep text
            text = p.get_text().strip()
            if text:
                text_content.append(text)
        
        full_text = ' '.join(text_content)
        
        # Check minimum word count
        word_count = len(full_text.split())
        if word_count < 50:
            print(f"⚠️  Skipping {url} - only {word_count} words")
            return None
            
        return {
            'url': url,
            'title': title,
            'text': full_text,
            'word_count': word_count
        }
        
    except Exception as e:
        print(f"❌ Error crawling {url}: {str(e)}")
        return None

def main():
    """Main crawling function"""
    print("🚀 Starting web crawling...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load URLs
    urls = load_sources()
    print(f"📋 Found {len(urls)} URLs to crawl")
    
    crawled_docs = []
    
    for url in urls:
        doc = extract_text(url)
        if doc:
            crawled_docs.append(doc)
        
        # Be respectful - add small delay
        time.sleep(1)
    
    # Save to JSONL
    output_file = 'data/raw_data.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in crawled_docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Done! Crawled {len(crawled_docs)} documents")
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    main()