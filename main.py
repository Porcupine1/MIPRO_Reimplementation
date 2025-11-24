#!/usr/bin/env python3
"""
Main script to run MIPRO optimization on QA program.
"""

from LMBackend import LMBackend
from QADataset import QADataset
from QAProgram import QAProgram
from MIPROOpt import MIPROOptimizer
from config import MODEL_NAME


def main():
    print("Initializing MIPRO Optimization...")
    print(f"Model: {MODEL_NAME}")
    
    # 1. Load dataset
    print("\n[1/4] Loading dataset...")
    dataset = QADataset()
    dataset.load()
    print(f"  Train: {len(dataset.train)} examples")
    print(f"  Validation: {len(dataset.validation)} examples")
    
    # 2. Initialize program
    print("\n[2/4] Initializing QA program...")
    backend = LMBackend()
    program = QAProgram(backend=backend)
    print(f"  Modules: {program.get_module_names()}")
    
    # 3. Test baseline (optional)
    print("\n[3/4] Testing baseline program...")
    test_batch = dataset.sample_batch(1, split="validation")
    if not test_batch:
        print("  Warning: No validation examples available")
    else:
        test_example = test_batch[0]
        try:
            baseline_answer = program.forward(test_example)
            print(f"  Question: {test_example['question'][:100]}...")
            print(f"  Baseline Answer: {baseline_answer[:100]}...")
            print(f"  Ground Truth: {test_example['answer'][:100]}...")
        except Exception as e:
            print(f"  Error in baseline: {e}")
    
    # 4. Run MIPRO optimization
    print("\n[4/4] Running MIPRO optimization...")
    optimizer = MIPROOptimizer(
        program=program,
        dataset=dataset,
        n_trials=10,
        batch_size=20,
        eval_batch_size=50
    )
    
    optimized_program = optimizer.optimize(
        task_description="Answer multi-hop questions using retrieved context from Wikipedia"
    )
    
    # 5. Test optimized program
    print("\n[5/5] Testing optimized program...")
    if test_batch:
        try:
            optimized_answer = optimized_program.forward(test_example)
            print(f"  Optimized Answer: {optimized_answer[:100]}...")
        except Exception as e:
            print(f"  Error in optimized: {e}")
    
    # 6. Show best instructions
    print("\n" + "="*60)
    print("Best Instructions Found:")
    print("="*60)
    best_instructions = optimizer.get_best_instructions()
    for module_name, instruction in best_instructions.items():
        print(f"\n{module_name}:")
        print(f"  {instruction}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

