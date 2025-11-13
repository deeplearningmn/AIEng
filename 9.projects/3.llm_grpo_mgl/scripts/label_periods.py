#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import os
from collections import defaultdict, Counter

def load_merged_dataset(file_path):
    """Load the merged dataset from JSONL file"""
    print(f"📖 Loading merged dataset: {file_path}")
    
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

def define_period_patterns():
    """Define regex patterns for each historical period"""
    patterns = {
        'Эртний үе': [
            # Ancient period keywords
            r'хүннү', r'модун', r'шаньюй', r'сяньби', r'жужан',
            r'түрэг\s+улс', r'уйгур\s+улс', r'кидан', r'ляо',
            r'эртний\s+улс', r'балар\s+эрт', r'чулуун\s+зэвсэг',
            r'хүрэл\s+зэвсэг', r'төмрийн\s+үе', r'археологи'
        ],
        'XIII зуун': [
            # 13th century - Mongol Empire
            r'чингис\s*хаан', r'чингисхаан', r'тэмүжин', r'өгөдэй',
            r'мөнх\s+хаан', r'хубилай', r'их\s+монгол\s+улс',
            r'монголын\s+эзэнт\s+гүрэн', r'юань\s+улс', r'нууц\s+товчоо',
            r'монголын\s+нууц', r'xiii\s*зуун', r'13.*зуун'
        ],
        'XVII–XIX зуун': [
            # 17th-19th century - Manchu period
            r'манж', r'цин\s+улс', r'маньчжур', r'богд\s+хаан',
            r'төвөд', r'далай\s+лам', r'гадаад\s+монгол', r'өвөр\s+монгол',
            r'автономи', r'xvii.*зуун', r'xviii.*зуун', r'xix.*зуун',
            r'17.*зуун', r'18.*зуун', r'19.*зуун'
        ],
        'XX зуун': [
            # 20th century - Socialist period
            r'бнмау', r'сүхбаатар', r'чойбалсан', r'ардын\s+хувьсгал',
            r'социализм', r'коммунист', r'коминтерн', r'зөвлөлт',
            r'ленин', r'сталин', r'xx\s*зуун', r'20.*зуун',
            r'1900', r'1910', r'1920', r'1930', r'1940', r'1950',
            r'1960', r'1970', r'1980', r'1990'
        ]
    }
    
    return patterns

def classify_historical_period(text, patterns):
    """Classify text into historical periods using regex patterns"""
    text_lower = text.lower()
    
    # Score each period based on pattern matches
    period_scores = {}
    period_matches = {}
    
    for period, period_patterns in patterns.items():
        matches = []
        score = 0
        
        for pattern in period_patterns:
            found_matches = re.findall(pattern, text_lower)
            if found_matches:
                matches.extend(found_matches)
                score += len(found_matches)
        
        period_scores[period] = score
        period_matches[period] = matches
    
    # Find the period with highest score
    if max(period_scores.values()) > 0:
        best_period = max(period_scores, key=period_scores.get)
        best_score = period_scores[best_period]
        best_matches = period_matches[best_period]
        
        # Calculate confidence based on number of matches
        if best_score >= 3:
            confidence = 1.0
        elif best_score == 2:
            confidence = 0.8
        else:
            confidence = 0.5
        
        return best_period, confidence, best_matches
    else:
        # Default to modern period if no matches
        return 'Орчин үе', 0.3, []

def label_records(records):
    """Label all records with historical periods"""
    print(f"\n🏷️  Labeling historical periods...")
    
    patterns = define_period_patterns()
    labeled_records = []
    period_stats = defaultdict(int)
    confidence_stats = defaultdict(int)
    
    for i, record in enumerate(records, 1):
        text = record.get('text', '')
        title = record.get('title', '')
        
        # Combine title and text for classification
        full_text = f"{title} {text}"
        
        # Classify the period
        period, confidence, matches = classify_historical_period(full_text, patterns)
        
        # Create labeled record
        labeled_record = record.copy()
        labeled_record['period'] = period
        labeled_record['period_confidence'] = confidence
        labeled_record['period_matches'] = len(matches)
        
        labeled_records.append(labeled_record)
        
        # Update statistics
        period_stats[period] += 1
        confidence_range = f"{confidence:.1f}"
        confidence_stats[confidence_range] += 1
        
        # Print progress for first few and every 10th record
        if i <= 5 or i % 10 == 0:
            source = record.get('dataset_source', 'unknown')[:20]
            print(f"   [{i:2d}] {period:<15} (conf: {confidence:.1f}) - {source}")
    
    print(f"✅ Labeled {len(labeled_records)} records")
    
    return labeled_records, period_stats, confidence_stats

def print_labeling_statistics(period_stats, confidence_stats, total_records):
    """Print detailed labeling statistics"""
    print(f"\n📊 Historical Period Distribution:")
    
    # Sort periods chronologically
    period_order = ['Эртний үе', 'XIII зуун', 'XVII–XIX зуун', 'XX зуун', 'Орчин үе']
    
    for period in period_order:
        count = period_stats.get(period, 0)
        percentage = (count / total_records * 100) if total_records > 0 else 0
        print(f"   📅 {period:<15}: {count:2d} records ({percentage:4.1f}%)")
    
    print(f"\n📈 Confidence Distribution:")
    for conf_level in sorted(confidence_stats.keys(), reverse=True):
        count = confidence_stats[conf_level]
        percentage = (count / total_records * 100) if total_records > 0 else 0
        print(f"   🎯 Confidence {conf_level}: {count:2d} records ({percentage:4.1f}%)")
    
    # Calculate quality metrics
    high_conf_count = sum(count for conf, count in confidence_stats.items() 
                         if float(conf) >= 0.8)
    high_conf_pct = (high_conf_count / total_records * 100) if total_records > 0 else 0
    
    print(f"\n🎯 Quality Metrics:")
    print(f"   ✅ High confidence (≥0.8): {high_conf_count} records ({high_conf_pct:.1f}%)")
    print(f"   📊 Average confidence: {sum(float(conf) * count for conf, count in confidence_stats.items()) / total_records:.2f}")

def save_labeled_dataset(records, output_path):
    """Save labeled dataset to JSONL file"""
    print(f"\n💾 Saving labeled dataset...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Saved {len(records)} labeled records to {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error saving file: {str(e)}")
        return False

def analyze_period_keywords(records, patterns):
    """Analyze which keywords were most effective for classification"""
    print(f"\n🔍 Keyword Analysis:")
    
    keyword_usage = defaultdict(int)
    
    for record in records:
        text = record.get('text', '').lower()
        period = record.get('period', '')
        
        if period in patterns:
            for pattern in patterns[period]:
                if re.search(pattern, text):
                    keyword_usage[f"{period}:{pattern}"] += 1
    
    # Show top keywords for each period
    for period in patterns.keys():
        period_keywords = [(k.split(':')[1], v) for k, v in keyword_usage.items() 
                          if k.startswith(period)]
        period_keywords.sort(key=lambda x: x[1], reverse=True)
        
        if period_keywords:
            print(f"   📅 {period}:")
            for keyword, count in period_keywords[:3]:  # Top 3 keywords
                print(f"      🔑 '{keyword}': {count} matches")

def print_labeling_summary(total_records, period_stats, output_path):
    """Print comprehensive labeling summary"""
    print(f"\n" + "="*60)
    print(f"📊 HISTORICAL PERIOD LABELING SUMMARY")
    print(f"="*60)
    
    print(f"📈 Processing Results:")
    print(f"   📋 Total records processed: {total_records}")
    print(f"   🏷️  Successfully labeled: {sum(period_stats.values())}")
    print(f"   📁 Output file: {output_path}")
    
    print(f"\n🎯 Labeling Quality:")
    if total_records > 0:
        # Find most and least common periods
        most_common = max(period_stats.items(), key=lambda x: x[1])
        least_common = min(period_stats.items(), key=lambda x: x[1])
        
        print(f"   📊 Most common period: {most_common[0]} ({most_common[1]} records)")
        print(f"   📊 Least common period: {least_common[0]} ({least_common[1]} records)")
    
    print(f"\n🎉 Labeling completed successfully!")
    print(f"🔗 Ready for final dataset preparation")

def main():
    """Main period labeling function"""
    print("🏷️  Starting Historical Period Labeling...")
    print("="*60)
    
    try:
        # Define file paths
        input_file = 'data/mgl_history_merged.jsonl'
        output_file = 'data/mgl_history_labeled.jsonl'
        
        # Load merged dataset
        records = load_merged_dataset(input_file)
        if not records:
            print("❌ No records to process")
            return
        
        # Label records with historical periods
        labeled_records, period_stats, confidence_stats = label_records(records)
        
        # Print statistics
        print_labeling_statistics(period_stats, confidence_stats, len(labeled_records))
        
        # Analyze keyword effectiveness
        patterns = define_period_patterns()
        analyze_period_keywords(labeled_records, patterns)
        
        # Save labeled dataset
        if save_labeled_dataset(labeled_records, output_file):
            # Print summary
            print_labeling_summary(len(records), period_stats, output_file)
        else:
            print("❌ Failed to save labeled dataset")
    
    except Exception as e:
        print(f"❌ Error during labeling: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()