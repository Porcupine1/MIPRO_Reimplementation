from typing import List, Dict, Any, Optional
import copy
from backend import LMBackend
from QADataset import QADataset
from .PromptMod import PromptModule, QueryModule, AnswerModule


class QAProgram:
    """multi-stage QA pipeline: query generation -> document retrieval -> answer generation"""
    
    def __init__(
        self,
        backend: Optional[LMBackend] = None,
        modules: Optional[Dict[str, PromptModule]] = None
    ):
        self.backend = backend or LMBackend()
        self.modules = modules or {
            "query": QueryModule(),
            "answer": AnswerModule()
        }
        self.trace = []  # List of module traces: [{"module": name, "input": {...}, "output": ...}, ...]
    
    def generate_query(self, question: str) -> str:
        """generate search query from question"""
        
        if "query" not in self.modules:
            raise ValueError("Query module not found in program")
        module = self.modules["query"]
        query = module.forward(question=question)
        # Record trace
        self.trace.append({
            "module": module.name,
            "input": {"question": question},
            "output": query
        })
        return query
    
    def retrieve_context(self, query: str, example: Dict[str, Any]) -> str:
        """retrive context from example. in reality, this would use a retriever to get the context"""


        # mock retrieval: use context from example
        # query would be used to retrieve the context
        if "context" not in example or "sentences" not in example["context"]:
            raise ValueError("Example must contain 'context' with 'sentences' key")
        contexts = []
        for ctx in example["context"]["sentences"]:
            contexts.extend(ctx)
        return " ".join(contexts)
    
    def generate_answer(self, question: str, context: str) -> str:
        """generate answer from question and context"""

        if "answer" not in self.modules:
            raise ValueError("Answer module not found in program")
        module = self.modules["answer"]
        answer = module.forward(question=question, context=context)
        # Record trace
        self.trace.append({
            "module": module.name,
            "input": {"question": question, "context": context},
            "output": answer
        })
        return answer
    
    def reset_trace(self):
        """reset the trace for a new run"""
        self.trace = []
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """get the current trace (list of module executions)"""
        return self.trace
    
    def forward(self, example: Dict[str, Any]) -> str:
        """run full pipeline on single example"""

        # Reset trace at the start of each run
        self.reset_trace()
        
        if "question" not in example:
            raise ValueError("Example must contain 'question' key")
        question = example["question"]
        
        query = self.generate_query(question)
        context = self.retrieve_context(query, example)
        answer = self.generate_answer(question, context)
        return answer
    
    def process_batch(self, batch: List[Dict[str, Any]], parallel: bool = True) -> List[str]:
        """
        process batch of examples
        
        Args:
            batch: List of examples to process
            parallel: If True, process examples in parallel (default: True)
        """
        if not parallel or len(batch) == 1:
            # Sequential processing for single items or when parallel is disabled
            predictions = []
            for example in batch:
                prediction = self.forward(example)
                predictions.append(prediction)
            return predictions
        
        # Parallel processing using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        predictions = [None] * len(batch)
        
        with ThreadPoolExecutor(max_workers=min(4, len(batch))) as executor:
            future_to_idx = {
                executor.submit(self.forward, example): idx 
                for idx, example in enumerate(batch)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    predictions[idx] = future.result()
                except Exception as e:
                    print(f"Error processing example {idx}: {e}")
                    predictions[idx] = ""
        
        return predictions
    
    def apply_configuration(self, instructions: Dict[str, str], demos: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        """apply instruction and demo configuration to modules"""

        for module_name, instruction in instructions.items():
            if module_name in self.modules:
                self.modules[module_name].set_instruction(instruction)
        
        if demos:
            for module_name, demo_list in demos.items():
                if module_name in self.modules:
                    # Deep copy to avoid shared state
                    self.modules[module_name].set_demos(copy.deepcopy(demo_list))
    
    def get_module_names(self) -> List[str]:
        """get list of module names"""

        return list(self.modules.keys())
    
    def clone(self) -> "QAProgram":
        """create a copy of this program"""

        cloned_modules = {
            name: module.clone() 
            for name, module in self.modules.items()
        }
        cloned = QAProgram(backend=self.backend, modules=cloned_modules)
        cloned.trace = []  # Initialize empty trace
        return cloned

