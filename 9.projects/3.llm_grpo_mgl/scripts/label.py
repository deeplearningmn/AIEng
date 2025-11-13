#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os

def load_cleaned_data():
    """Load cleaned data from JSONL file"""
    docs = []
    try:
        with open('data/cleaned_data.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        print(f"📥 Loaded {len(docs)} cleaned documents")
        return docs
    except FileNotFoundError:
        print("❌ cleaned_data.jsonl not found. Run clean.py first.")
        return []

def classify_historical_period(text):
    """Classify text into historical periods using regex patterns"""
    text_lower = text.lower()
    
    # Define period patterns with Mongolian terms
    patterns = {
        'Хүннү': [
            r'хүннү', r'хунну', r'hunnu', r'xiongnu',
            r'модун шаньюй', r'шаньюй', r'хүннүгийн эзэнт гүрэн'
        ],
        'XIII зуун': [
            r'чингис\s*хаан', r'чингисхаан', r'genghis', r'чингис',
            r'монгол\s*эзэнт\s*гүрэн', r'их\s*монгол\s*улс',
            r'өгөдэй', r'мөнх', r'хубилай', r'xiii\s*зуун', r'13.*зуун'
        ],
        'XVII–XIX зуун': [
            r'манж', r'цин', r'qing', r'маньчжур',
            r'xvii.*зуун', r'xviii.*зуун', r'xix.*зуун',
            r'17.*зуун', r'18.*зуун', r'19.*зуун',
            r'богд\s*хаан', r'автономи', r'хятад'
        ],
        'XX зуун': [
            r'ардын\s*хувьсгал', r'социализм', r'зөвлөлт',
            r'бнмау', r'сүхбаатар', r'чойбалсан',
            r'xx\s*зуун', r'20.*зуун', r'1900', r'1910', r'1920',
            r'1930', r'1940', r'1950', r'1960', r'1970', r'1980', r'1990'
        ],
        'Орчин үе': [
            r'ардчилал', r'зах\s*зээл', r'шинэ\s*үе',
            r'21.*зуун', r'xxi\s*зуун', r'2000', r'2010', r'2020',
            r'орчин\s*үе', r'өнөөгийн', r'хөгжил'
        ]
    }
    
    # Score each period
    period_scores = {}
    
    for period, period_patterns in patterns.items():
        score = 0
        for pattern in period_patterns:
            matches = len(re.findall(pattern, text_lower))
            score += matches
        period_scores[period] = score
    
    # Return period with highest score, or 'Орчин үе' as default
    if max(period_scores.values()) > 0:
        return max(period_scores, key=period_scores.get)
    else:
        return 'Орчин үе'

def main():
    """Main labeling function"""
    print("🏷️  Starting historical period labeling...")
    
    # Load cleaned data
    cleaned_docs = load_cleaned_data()
    if not cleaned_docs:
        return
    
    labeled_docs = []
    period_counts = {}
    
    for doc in cleaned_docs:
        # Combine title and text for classification
        full_text = f"{doc.get('title', '')} {doc.get('text', '')}"
        
        # Classify historical period
        period = classify_historical_period(full_text)
        
        # Count periods
        period_counts[period] = period_counts.get(period, 0) + 1
        
        # Add period to document
        labeled_doc = doc.copy()
        labeled_doc['period'] = period
        
        labeled_docs.append(labeled_doc)
    
    # Save labeled data
    os.makedirs('data', exist_ok=True)
    output_file = 'data/labeled_data.jsonl'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc in labeled_docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"✅ Done labeling! {len(labeled_docs)} documents processed")
    print("📊 Period distribution:")
    for period, count in sorted(period_counts.items()):
        print(f"   {period}: {count} documents")
    print(f"💾 Saved to {output_file}")

if __name__ == "__main__":
    main()