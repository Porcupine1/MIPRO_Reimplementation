from typing import List, Dict, Any, Optional
import copy
from LMBackend import LMBackend


class PromptModule:
    """base class for prompt modules with instruction and demonstrations"""
    
    def __init__(
        self,
        name: str,
        lm: Optional[LMBackend] = None,
        instruction: str = "",
        demos: Optional[List[Dict[str, Any]]] = None
    ):
        self.name = name
        self.lm = lm or LMBackend()
        self.instruction = instruction
        self.demos = demos or []
    
    def build_prompt(self, **inputs) -> str:
        """build prompt from instruction, demos, and inputs"""

        raise NotImplementedError
    
    def forward(self, **inputs) -> str:
        """generate output by building prompt and calling LM"""

        prompt = self.build_prompt(**inputs)
        output = self.lm.generate(prompt)
        return self.parse_output(output)
    
    def parse_output(self, output: str) -> str:
        """parse LM output (override this)"""

        return output.strip()
    
    def set_instruction(self, instruction: str):
        """update instruction"""

        self.instruction = instruction
    
    def set_demos(self, demos: List[Dict[str, Any]]):
        """update demonstrations"""

        self.demos = demos
    
    def clone(self) -> "PromptModule":
        """create a copy of this module"""

        raise NotImplementedError


class QueryModule(PromptModule):
    """module for query generation"""
    
    def __init__(
        self,
        lm: Optional[LMBackend] = None,
        instruction: str = "Generate a search query to answer the question.",
        demos: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(name="query", lm=lm, instruction=instruction, demos=demos)
    
    def build_prompt(self, question: str, **kwargs) -> str:
        """build query generation prompt"""
        parts = [self.instruction, ""]
        
        # add demos
        for demo in self.demos:
            parts.append(f"Question: {demo['question']}")
            parts.append(f"Search Query: {demo['query']}\n")
        
        # add current input
        parts.append(f"Question: {question}")
        parts.append("Search Query:")
        
        return "\n".join(parts)
    
    def clone(self) -> "QueryModule":
        return QueryModule(
            lm=self.lm,
            instruction=self.instruction,
            demos=copy.deepcopy(self.demos)
        )


class AnswerModule(PromptModule):
    """module for answer generation"""
    
    def __init__(
        self,
        lm: Optional[LMBackend] = None,
        instruction: str = "answer the question based on the context",
        demos: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(name="answer", lm=lm, instruction=instruction, demos=demos)
    
    def build_prompt(self, question: str, context: str, **kwargs) -> str:
        """build answer generation prompt"""
        parts = [self.instruction, ""]
        
        # add demos
        for demo in self.demos:
            parts.append(f"Question: {demo['question']}")
            parts.append(f"Context: {demo.get('context', '')}")
            parts.append(f"Answer: {demo['answer']}\n")
        
        # add current input
        parts.append(f"Question: {question}")
        parts.append(f"Context: {context}")
        parts.append("Answer:")
        
        return "\n".join(parts)
    
    def clone(self) -> "AnswerModule":
        return AnswerModule(
            lm=self.lm,
            instruction=self.instruction,
            demos=copy.deepcopy(self.demos)
        )

