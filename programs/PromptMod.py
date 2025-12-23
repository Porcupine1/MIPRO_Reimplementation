from typing import List, Dict, Any, Optional
import copy
from backend import LMBackend


class PromptModule:
    """base class for prompt modules with instruction and demonstrations"""

    def __init__(
        self,
        name: str,
        lm: Optional[LMBackend] = None,
        instruction: str = "",
        demos: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name = name
        self.lm = lm or LMBackend()
        self.instruction = instruction
        self.demos = demos or []

    def build_prompt(self, **inputs) -> str:
        """build prompt from instruction, demos, and inputs"""

        raise NotImplementedError

    def forward(self, **inputs) -> str:
        """
        generate output by building prompt and calling LM

        Args:
            **inputs: Input arguments for the module
        """

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
        demos: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(name="query", lm=lm, instruction=instruction, demos=demos)

    def build_prompt(self, question: str, **kwargs) -> str:
        """build query generation prompt"""
        parts = [self.instruction, ""]

        # add demos
        for demo in self.demos:
            # Handle both trace format and flat format
            if "input" in demo:
                # Trace format: {"module": "...", "input": {"question": "..."}, "output": "..."}
                demo_question = demo["input"].get("question", "")
                demo_query = demo.get("output", "")
            else:
                # Flat format: {"question": "...", "query": "..."} or {"question": "...", "output": "..."}
                demo_question = demo.get("question", "")
                demo_query = demo.get("query") or demo.get("output", "")

            parts.append(f"Question: {demo_question}")
            parts.append(f"Search Query: {demo_query}\n")

        # add current input
        parts.append(f"Question: {question}")
        parts.append("Search Query:")

        return "\n".join(parts)

    def clone(self) -> "QueryModule":
        return QueryModule(
            lm=self.lm, instruction=self.instruction, demos=copy.deepcopy(self.demos)
        )


class AnswerModule(PromptModule):
    """module for answer generation"""

    def __init__(
        self,
        lm: Optional[LMBackend] = None,
        instruction: str = "answer the question based on the context",
        demos: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(name="answer", lm=lm, instruction=instruction, demos=demos)

    def build_prompt(self, question: str, context: str, **kwargs) -> str:
        """build answer generation prompt"""
        parts = [self.instruction, ""]

        # add demos
        for demo in self.demos:
            # Handle both trace format and flat format
            if "input" in demo:
                # Trace format: {"module": "...", "input": {"question": "...", "context": "..."}, "output": "..."}
                demo_question = demo["input"].get("question", "")
                demo_context = demo["input"].get("context", "")
                demo_answer = demo.get("output", "")
            else:
                # Flat format: {"question": "...", "context": "...", "answer": "..."} or {"question": "...", "context": "...", "output": "..."}
                demo_question = demo.get("question", "")
                demo_context = demo.get("context", "")
                demo_answer = demo.get("answer") or demo.get("output", "")

            parts.append(f"Question: {demo_question}")
            parts.append(f"Context: {demo_context}")
            parts.append(f"Answer: {demo_answer}\n")

        # add current input
        parts.append(f"Question: {question}")
        parts.append(f"Context: {context}")
        parts.append("Answer:")

        return "\n".join(parts)

    def clone(self) -> "AnswerModule":
        return AnswerModule(
            lm=self.lm, instruction=self.instruction, demos=copy.deepcopy(self.demos)
        )
