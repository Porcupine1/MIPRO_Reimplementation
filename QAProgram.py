from typing import List, Dict, Any, Optional
import copy
from LMBackend import LMBackend
from QADataset import QADataset
from PromptMod import PromptModule, QueryModule, AnswerModule


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
    
    def generate_query(self, question: str) -> str:
        """generate search query from question"""
        
        if "query" not in self.modules:
            raise ValueError("Query module not found in program")
        query = self.modules["query"].forward(question=question)
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
        answer = self.modules["answer"].forward(question=question, context=context)
        return answer
    
    def forward(self, example: Dict[str, Any]) -> str:
        """run full pipeline on single example"""

        if "question" not in example:
            raise ValueError("Example must contain 'question' key")
        question = example["question"]
        
        query = self.generate_query(question)
        context = self.retrieve_context(query, example)
        answer = self.generate_answer(question, context)
        return answer
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """process batch of examples"""

        predictions = []
        for example in batch:
            prediction = self.forward(example)
            predictions.append(prediction)
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
        return QAProgram(backend=self.backend, modules=cloned_modules)

