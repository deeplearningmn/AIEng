#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
import yaml
import json
import os
import time
import re
from urllib.parse import urlparse, urljoin
from collections import defaultdict, Counter
import random

def load_urls_from_yaml(yaml_path='sources.yaml'):
    """Load URLs from YAML configuration file"""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        urls = data.get('urls', [])
        print(f"📋 Loaded {len(urls)} URLs from {yaml_path}")
        return urls
    except FileNotFoundError:
        print(f"❌ File not found: {yaml_path}")
        return []
    except Exception as e:
        print(f"❌ Error loading YAML: {str(e)}")
        return []

def get_domain_name(url):
    """Extract domain name from URL"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "unknown"

def is_mongolian_text(text):
    """Check if text contains significant Mongolian content"""
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
    """Clean and normalize extracted text"""
    if not text:
        return ""
    
    # Remove invisible characters and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remove zero-width characters and non-breaking spaces
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0]', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Clean up spaces around punctuation
    text = re.sub(r'\s+([,.!?;:)])', r'\1', text)
    text = re.sub(r'([(])\s+', r'\1', text)
    
    return text.strip()

def extract_content(url, session, timeout=10):
    """Extract title and text content from a webpage"""
    try:
        print(f"🕸️  Scraping: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'mn,en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        response = session.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Check if content is HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            print(f"⚠️  Skipping non-HTML content: {url}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'advertisement', 'ads', 'menu', 'sidebar']):
            element.decompose()
        
        # Extract title
        title = ""
        # Try h1 first, then title tag
        h1_tag = soup.find('h1')
        if h1_tag:
            title = h1_tag.get_text().strip()
        else:
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
        
        # Extract main content
        text_content = []
        
        # Look for main content areas first
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|article', re.I))
        
        if main_content:
            paragraphs = main_content.find_all('p')
        else:
            paragraphs = soup.find_all('p')
        
        for p in paragraphs:
            # Skip paragraphs that are likely navigation or ads
            if p.find_parent(['nav', 'header', 'footer', 'aside']):
                continue
                
            text = p.get_text().strip()
            if text and len(text.split()) >= 5:  # At least 5 words
                text_content.append(text)
        
        # Combine all text
        full_text = ' '.join(text_content)
        full_text = clean_text(full_text)
        
        # Check if we have enough meaningful content
        word_count = len(full_text.split())
        if word_count < 50:
            print(f"⚠️  Insufficient content: {url} ({word_count} words)")
            return None
        
        # Check if content is primarily Mongolian
        if not is_mongolian_text(full_text):
            print(f"⚠️  Non-Mongolian content: {url}")
            return None
        
        return {
            'url': url,
            'title': clean_text(title),
            'text': full_text,
            'source': get_domain_name(url),
            'word_count': word_count
        }
        
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout: {url}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"🔌 Connection error: {url}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"🚫 HTTP error {e.response.status_code}: {url}")
        return None
    except Exception as e:
        print(f"❌ Error scraping {url}: {str(e)}")
        return None

def scrape_with_retries(url, session, max_retries=3, base_delay=1):
    """Scrape URL with retry logic"""
    for attempt in range(max_retries):
        result = extract_content(url, session)
        if result is not None:
            return result
        
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"🔄 Retry {attempt + 1}/{max_retries} for {url} in {delay:.1f}s")
            time.sleep(delay)
    
    return None

def save_to_jsonl(records, output_path):
    """Save records to JSONL file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"💾 Saved {len(records)} records to {output_path}")

def print_scraping_stats(records, skipped_count, total_urls):
    """Print detailed scraping statistics"""
    if not records:
        print("📊 No records scraped successfully")
        return
    
    # Basic stats
    total_scraped = len(records)
    total_words = sum(r['word_count'] for r in records)
    avg_words = total_words // total_scraped if total_scraped > 0 else 0
    
    # Domain statistics
    domain_counts = Counter(r['source'] for r in records)
    top_domains = domain_counts.most_common(5)
    
    print(f"\n📊 Scraping Statistics:")
    print(f"   ✅ {total_scraped} pages scraped successfully")
    print(f"   ⚠️  {skipped_count} pages skipped")
    print(f"   📋 {total_urls} total URLs processed")
    print(f"   📝 {total_words:,} total words")
    print(f"   📊 {avg_words} average words per page")
    
    print(f"\n🏆 Top domains:")
    for domain, count in top_domains:
        percentage = (count / total_scraped) * 100
        print(f"   {domain}: {count} pages ({percentage:.1f}%)")
    
    # Word count distribution
    word_counts = [r['word_count'] for r in records]
    if word_counts:
        print(f"\n📈 Content distribution:")
        print(f"   Shortest: {min(word_counts)} words")
        print(f"   Longest: {max(word_counts)} words")
        print(f"   Median: {sorted(word_counts)[len(word_counts)//2]} words")

def main():
    """Main scraping function"""
    print("🌐 Starting Mongolian historical sites scraping...")
    
    # Load URLs
    urls = load_urls_from_yaml('sources.yaml')
    if not urls:
        print("❌ No URLs to scrape")
        return
    
    # Remove duplicates while preserving order
    seen = set()
    unique_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique_urls.append(url)
    
    print(f"🔗 Processing {len(unique_urls)} unique URLs")
    
    # Create session for connection reuse
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    scraped_records = []
    skipped_count = 0
    
    try:
        for i, url in enumerate(unique_urls, 1):
            print(f"\n[{i}/{len(unique_urls)}] Processing: {url}")
            
            # Add random delay to be respectful
            if i > 1:
                delay = random.uniform(1, 3)
                time.sleep(delay)
            
            result = scrape_with_retries(url, session)
            
            if result:
                scraped_records.append(result)
                print(f"✅ Success: {result['word_count']} words from {result['source']}")
            else:
                skipped_count += 1
                print(f"⚠️  Skipped: {url}")
        
        # Save results
        if scraped_records:
            output_path = 'data/web_raw.jsonl'
            save_to_jsonl(scraped_records, output_path)
            
            # Print statistics
            print_scraping_stats(scraped_records, skipped_count, len(unique_urls))
            
            print(f"\n🎉 Scraping completed successfully!")
            print(f"📁 Results saved to: {output_path}")
        else:
            print("❌ No content was successfully scraped")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Scraping interrupted by user")
        if scraped_records:
            output_path = 'data/web_raw.jsonl'
            save_to_jsonl(scraped_records, output_path)
            print(f"💾 Partial results saved to: {output_path}")
    
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        session.close()

if __name__ == "__main__":
    main()