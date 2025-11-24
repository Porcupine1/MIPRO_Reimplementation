#!/usr/bin/env python3
"""
Test script to demonstrate grounding: learning queries from context.
"""

from QADataset import QADataset
from GroundingHelper import GroundingHelper
from LMBackend import LMBackend

def main():
    print("="*60)
    print("Testing Grounding: Learning Queries from Context")
    print("="*60)
    
    # Load dataset
    print("\n[1/3] Loading dataset...")
    dataset = QADataset().load()
    
    # Sample examples
    examples = dataset.sample_batch(3, split="train")
    
    # Initialize grounding helper
    print("\n[2/3] Initializing grounding helper...")
    grounding = GroundingHelper(lm=LMBackend())
    
    # Generate grounded queries
    print("\n[3/3] Generating grounded queries...\n")
    
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}")
        print(f"{'='*60}")
        
        question = example["question"]
        answer = example["answer"]
        
        # Get context
        contexts = []
        for ctx in example["context"]["sentences"]:
            contexts.extend(ctx)
        context = " ".join(contexts)
        
        print(f"\nQuestion: {question}")
        print(f"\nAnswer: {answer}")
        print(f"\nContext: {context}...")
        
        # Generate grounded query
        print("\n[Generating grounded query...]")
        grounded_query = grounding.generate_grounded_query(question, context, answer)
        
        print(f"\nGrounded Query: {grounded_query}")
        print(f"\n→ This is what the model should learn to generate!")
        print(f"→ This query would retrieve the context containing '{answer}'")
    
    print("\n" + "="*60)
    print("How This Helps MIPRO:")
    print("="*60)
    print("""
1. We show the instruction proposer examples of GOOD queries
2. Good queries = queries that would retrieve relevant context
3. The proposer generates instructions based on these examples
4. MIPRO optimizes to find instructions that produce queries like these
5. Result: Better query generation → Better retrieval → Better answers
    """)

if __name__ == "__main__":
    main()

