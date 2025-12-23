from typing import List, Dict, Any
from backend import LMBackend


class GroundingHelper:
    """
    grounding: characterize task dynamics from dataset and program
    
    - characterize patterns in the raw dataset
    - summarize the program's control flow
    - bootstrapped program demonstrations
    - per-stage prompts with scores
    """
    
    def __init__(self, lm: LMBackend = None):
        self.lm = lm or LMBackend()
    
    def generate_grounded_query(
        self,
        question: str,
        context: str,
        answer: str
    ) -> str:
        """
        Given question, context, and answer,
        generate what search query would have retrieved this context
        
        This is the "ground truth" query we want to learn to generate
        """
        prompt = f"""Given a question and relevant context, generate the search query that would have been used to retrieve this context.

Question: {question}

Relevant Context: {context}...

Answer found in context: {answer}

What search query would retrieve this context? Generate a concise, effective search query:

Search Query:"""
        
        grounded_query = self.lm.generate(prompt, temperature=0.3)
        return grounded_query.strip()
    
    def summarize_dataset(
        self,
        examples: List[Dict[str, Any]],
        n_samples: int = 10,
        batch_size: int = 5
    ) -> str:
        """
        Characterize patterns in the dataset using iterative batch process.
        
        returns a summary of:
        - question types
        - answer types
        - common patterns
        """
        sample_text = []
        for i, ex in enumerate(examples[:n_samples], 1):
            sample_text.append(f"{i}. Question: {ex['question']}... Answer: {ex['answer']}...")
        
        prompt = f"""Analyze these question-answer examples and describe the key patterns:

{chr(10).join(sample_text)}

Describe in 2-3 sentences:
1. What types of questions are asked?
2. What types of answers are expected?
3. What makes a good answer?

Summary:"""
        
        summary = self.lm.generate(prompt, temperature=0.3)
        return summary.strip()
    
    def summarize_program(self, program) -> str:
        """
        summarize the program's control flow.
        
        returns description of what each module does.
        """
        modules = program.get_module_names()
        
        summary = f"""This program has {len(modules)} modules:
1. Query Module: Generates search queries from questions
2. Retrieval: Retrieves relevant context
3. Answer Module: Generates answers from question + context

The program flow: Question -> Query -> Retrieve Context -> Answer"""
        
        return summary
