#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import os

def load_labeled_data():
    """Load labeled data from JSONL file"""
    docs = []
    try:
        with open('data/labeled_data.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
        print(f"📥 Loaded {len(docs)} labeled documents")
        return docs
    except FileNotFoundError:
        print("❌ labeled_data.jsonl not found. Run label.py first.")
        return []

def split_dataset(docs, train_ratio=0.9):
    """Split dataset into train and test sets"""
    # Shuffle documents randomly
    random.seed(42)  # For reproducible results
    shuffled_docs = docs.copy()
    random.shuffle(shuffled_docs)
    
    # Calculate split point
    total_docs = len(shuffled_docs)
    train_size = int(total_docs * train_ratio)
    
    # Split the data
    train_docs = shuffled_docs[:train_size]
    test_docs = shuffled_docs[train_size:]
    
    return train_docs, test_docs

def save_dataset_split(docs, filename):
    """Save dataset split to JSONL file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for doc in docs:
            json.dump(doc, f, ensure_ascii=False)
            f.write('\n')

def analyze_split(train_docs, test_docs):
    """Analyze and print statistics about the dataset splits"""
    def count_periods(docs):
        period_counts = {}
        for doc in docs:
            period = doc.get('period', 'Unknown')
            period_counts[period] = period_counts.get(period, 0) + 1
        return period_counts
    
    train_periods = count_periods(train_docs)
    test_periods = count_periods(test_docs)
    
    print("📊 Dataset split analysis:")
    print(f"   Total documents: {len(train_docs) + len(test_docs)}")
    print(f"   Training set: {len(train_docs)} documents ({len(train_docs)/(len(train_docs)+len(test_docs))*100:.1f}%)")
    print(f"   Test set: {len(test_docs)} documents ({len(test_docs)/(len(train_docs)+len(test_docs))*100:.1f}%)")
    
    print("\n📈 Period distribution in training set:")
    for period, count in sorted(train_periods.items()):
        percentage = (count / len(train_docs)) * 100
        print(f"   {period}: {count} docs ({percentage:.1f}%)")
    
    print("\n📈 Period distribution in test set:")
    for period, count in sorted(test_periods.items()):
        percentage = (count / len(test_docs)) * 100 if len(test_docs) > 0 else 0
        print(f"   {period}: {count} docs ({percentage:.1f}%)")

def main():
    """Main dataset building function"""
    print("🏗️  Building final dataset splits...")
    
    # Load labeled data
    labeled_docs = load_labeled_data()
    if not labeled_docs:
        return
    
    # Split dataset
    train_docs, test_docs = split_dataset(labeled_docs, train_ratio=0.9)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save splits
    train_file = 'data/dataset_train.jsonl'
    test_file = 'data/dataset_test.jsonl'
    
    save_dataset_split(train_docs, train_file)
    save_dataset_split(test_docs, test_file)
    
    # Print analysis
    analyze_split(train_docs, test_docs)
    
    print(f"\n✅ Done! Dataset splits created:")
    print(f"💾 Training set: {train_file}")
    print(f"💾 Test set: {test_file}")

if __name__ == "__main__":
    main()