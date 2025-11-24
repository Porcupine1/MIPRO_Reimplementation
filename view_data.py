#!/usr/bin/env python3
"""
Simple script to view HotpotQA dataset in human-readable format.
"""

from datasets import load_from_disk
import sys

def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)

def print_example(example, index):
    """Print a single example in a readable format."""
    print_separator("=")
    print(f"EXAMPLE #{index}")
    print_separator("=")
    
    print(f"\n📋 ID: {example['id']}")
    print(f"📊 Type: {example['type']}")
    print(f"🎯 Level: {example['level']}")
    
    print(f"\n❓ QUESTION:")
    print(f"   {example['question']}")
    
    print(f"\n✅ ANSWER:")
    print(f"   {example['answer']}")
    
    print(f"\n📚 CONTEXT:")
    for i, (title, sentences) in enumerate(zip(example['context']['title'], 
                                                 example['context']['sentences'])):
        print(f"\n   [{i+1}] {title}")
        for j, sentence in enumerate(sentences):
            print(f"       {j}. {sentence}")
    
    print(f"\n🔍 SUPPORTING FACTS:")
    for title, sent_id in zip(example['supporting_facts']['title'], 
                              example['supporting_facts']['sent_id']):
        print(f"   - {title} (sentence {sent_id})")
    
    print()

def main():
    data_dir = "data/hotpotqa"
    
    print("Loading HotpotQA dataset...")
    dataset = load_from_disk(data_dir)
    
    print(f"\n📦 Dataset loaded successfully!")
    print(f"   - Train examples: {len(dataset['train'])}")
    print(f"   - Validation examples: {len(dataset['validation'])}")
    print(f"   - Test examples: {len(dataset['test'])}")
    
    # Ask user which split to view
    print("\nWhich split would you like to view?")
    print("  1. train")
    print("  2. validation")
    print("  3. test")
    
    choice = input("\nEnter choice (1-3) or press Enter for validation: ").strip()
    
    split_map = {"1": "train", "2": "validation", "3": "test", "": "validation"}
    split = split_map.get(choice, "validation")
    
    data = dataset[split]
    print(f"\n📂 Viewing {split} split ({len(data)} examples)")
    
    # Ask how many examples to view
    num_str = input("\nHow many examples to view? (default: 3): ").strip()
    num_examples = int(num_str) if num_str.isdigit() else 3
    num_examples = min(num_examples, len(data))
    
    # Ask starting index
    start_str = input(f"Starting from index? (0-{len(data)-1}, default: 0): ").strip()
    start_idx = int(start_str) if start_str.isdigit() else 0
    start_idx = max(0, min(start_idx, len(data) - 1))
    
    print(f"\n🔎 Showing {num_examples} examples starting from index {start_idx}...\n")
    
    # Display examples
    for i in range(num_examples):
        idx = start_idx + i
        if idx >= len(data):
            break
        print_example(data[idx], idx)
        
        # Ask if user wants to continue after each example (except the last one)
        if i < num_examples - 1:
            cont = input("Press Enter to see next example (or 'q' to quit): ").strip().lower()
            if cont == 'q':
                break

if __name__ == "__main__":
    main()

