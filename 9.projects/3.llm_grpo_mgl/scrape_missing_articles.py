#!/usr/bin/env python3
"""
Scrape missing Mongolian Wikipedia articles for comprehensive history coverage
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

# Key articles needed for 100 questions coverage
ARTICLES = [
    # Ancient period (Хүннү, Сяньби, Жужан)
    "Хүннү",
    "Модун_шаньюй",
    "Сяньби",
    "Жужан",
    
    # Turkic period (Түрэг, Уйгур, Киргиз)
    "Түрэг",
    "Көктүрк",
    "Билгэ_хаан",
    "Тоныүкүк",
    "Уйгур",
    "Орхон_бичиг",
    
    # Pre-Chinggis (Хамаг Монгол)
    "Хамаг_Монгол",
    "Есүхэй_баатар",
    "Жамуха",
    
    # Mongol Empire expansion
    "Хархорум",
    "Бат_хаан",
    "Мөнх_хаан",
    
    # Yuan Dynasty
    "Юань_улс",
    
    # Post-Yuan period
    "Даян_хаан",
    "Алтан_хаан",
    "Занабазар",
    
    # Oirat-Dzungar
    "Ойрад",
    "Зүүнгар",
    "Галдан_Бошигт",
    
    # Manchu period
    "Манж_улс",
    
    # National revolution
    "Богд_хаан",
    "Дамдины_Сүхбаатар",
]

def scrape_wikipedia_article(title):
    """Scrape a Mongolian Wikipedia article"""
    url = f"https://mn.wikipedia.org/wiki/{title}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None
        
        # Extract paragraphs
        paragraphs = content_div.find_all('p')
        text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if len(text) < 100:  # Too short
            return None
        
        return {
            'title': title.replace('_', ' '),
            'text': text,
            'url': url,
            'source': 'mn.wikipedia.org',
            'word_count': len(text.split())
        }
        
    except Exception as e:
        print(f"   ❌ Error scraping {title}: {e}")
        return None

def main():
    print("🌐 Scraping Missing Wikipedia Articles")
    print("=" * 60)
    
    output_file = Path("data/wikipedia_missing.jsonl")
    
    print(f"\n📋 Articles to scrape: {len(ARTICLES)}")
    print(f"💾 Output: {output_file}")
    print("\n" + "=" * 60)
    
    results = []
    success = 0
    failed = 0
    
    for i, article in enumerate(ARTICLES, 1):
        print(f"\n[{i}/{len(ARTICLES)}] Scraping: {article}")
        
        data = scrape_wikipedia_article(article)
        
        if data:
            results.append(data)
            success += 1
            print(f"   ✅ Success ({data['word_count']} words)")
        else:
            failed += 1
            print(f"   ❌ Failed")
        
        # Be nice to Wikipedia
        time.sleep(1)
    
    # Save results
    print("\n" + "=" * 60)
    print("💾 Saving results...")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(results)} articles to {output_file}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total attempted: {len(ARTICLES)}")
    print(f"✅ Success: {success}")
    print(f"❌ Failed: {failed}")
    print(f"Success rate: {success/len(ARTICLES)*100:.1f}%")
    
    total_words = sum(r['word_count'] for r in results)
    print(f"\nTotal words scraped: {total_words:,}")
    print(f"Average per article: {total_words/len(results):.0f} words")
    
    print("\n✅ Done! Next steps:")
    print("1. Review data/wikipedia_missing.jsonl")
    print("2. Merge with existing dataset")
    print("3. Regenerate embeddings")
    print("4. Re-test 100 questions")

if __name__ == "__main__":
    main()
