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
        """update demonstrations (base implementation, may be overridden by subclasses)"""

        self.demos = demos

    def clone(self) -> "PromptModule":
        """create a copy of this module"""

        raise NotImplementedError


class QueryModule(PromptModule):
    """module for query generation"""

    def __init__(
        self,
        lm: Optional[LMBackend] = None,
        instruction: str = "Write a search query for Wikipedia. Output only the query, nothing else.",
        demos: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(name="query", lm=lm, instruction=instruction, demos=demos)

    def build_prompt(self, question: str, **kwargs) -> str:
        """build query generation prompt"""
        parts = [self.instruction, ""]

        # add demos
        for demo in self.demos:
            # Demos are normalized to:
            # {"module", "input", "output", "score?", "context?"}
            input_data = demo.get("input", {}) or {}
            demo_question = input_data.get("question", "")
            demo_query = demo.get("output", "")

            parts.append(f"Question: {demo_question}")
            parts.append(f"Search Query: {demo_query}\n")

        # add current input
        parts.append(f"Question: {question}")
        parts.append("Search Query:")

        return "\n".join(parts)

    def parse_output(self, output: str) -> str:
        """
        Extract clean query from LLM output.
        Handles verbose responses that include explanations.
        """
        output = output.strip()

        # If output contains quotes, extract the quoted text
        import re

        quoted = re.findall(r'"([^"]+)"', output)
        if quoted:
            # Return the first quoted string (usually the actual query)
            return quoted[0].strip()

        # If multiple lines, try to find the line that looks like a query
        lines = output.split("\n")
        if len(lines) > 1:
            # Look for lines that don't start with common explanation phrases
            skip_phrases = ["here", "try", "you can", "this query", "or,"]
            for line in lines:
                line_lower = line.lower().strip()
                if line_lower and not any(
                    line_lower.startswith(phrase) for phrase in skip_phrases
                ):
                    # Check if it looks like a query (no colons, not a sentence with period at end)
                    if ":" not in line and not line.endswith("."):
                        return line.strip()

        # Fallback: return the first line or the original output
        return lines[0].strip() if lines else output

    def clone(self) -> "QueryModule":
        return QueryModule(
            lm=self.lm, instruction=self.instruction, demos=copy.deepcopy(self.demos)
        )

    def set_demos(self, demos: List[Dict[str, Any]]):
        """
        Normalize demos to a module-agnostic schema:
        { "module", "input", "output", "score?", "context?" }
        """

        normalized: List[Dict[str, Any]] = []
        for demo in demos:
            # Already in (or close to) normalized/trace format
            if "module" in demo and "input" in demo and "output" in demo:
                module_name = demo.get("module", self.name)
                input_data = demo.get("input") or {}
                # Ensure question lives under input
                if "question" in demo and "question" not in input_data:
                    input_data = {**input_data, "question": demo["question"]}
                output = demo.get("output", "")
                score = demo.get("score")
                context = demo.get("context")
            else:
                # Legacy/flat formats:
                # {"question", "query", "score?", "context?"}
                module_name = self.name
                question = demo.get("question") or demo.get("input", {}).get(
                    "question", ""
                )
                output = demo.get("query") or demo.get("output", "")
                score = demo.get("score")
                context = demo.get("context")
                input_data = {"question": question}

            norm_demo: Dict[str, Any] = {
                "module": module_name,
                "input": input_data,
                "output": output,
            }
            if score is not None:
                norm_demo["score"] = score
            if context is not None:
                norm_demo["context"] = context

            normalized.append(norm_demo)

        self.demos = normalized


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
            # Demos are normalized to:
            # {"module", "input", "output", "score?", "context?"}
            input_data = demo.get("input", {}) or {}
            demo_question = input_data.get("question", "")
            # Prefer top-level context if present, fall back to input["context"]
            demo_context = demo.get("context") or input_data.get("context", "")
            demo_answer = demo.get("output", "")

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

    def set_demos(self, demos: List[Dict[str, Any]]):
        """
        Normalize demos to a module-agnostic schema:
        { "module", "input", "output", "score?", "context?" }
        """

        normalized: List[Dict[str, Any]] = []
        for demo in demos:
            # Already in (or close to) normalized/trace format
            if "module" in demo and "input" in demo and "output" in demo:
                module_name = demo.get("module", self.name)
                input_data = demo.get("input") or {}
                # Ensure question/context live under input
                if "question" in demo and "question" not in input_data:
                    input_data = {**input_data, "question": demo["question"]}
                if "context" in demo and "context" not in input_data:
                    input_data = {**input_data, "context": demo["context"]}
                output = demo.get("output", "")
                score = demo.get("score")
                # Keep a top-level context copy if available
                context = demo.get("context") or input_data.get("context")
            else:
                # Legacy/flat formats:
                # {"question", "context", "answer"/"output", "score?"}
                module_name = self.name
                question = demo.get("question") or demo.get("input", {}).get(
                    "question", ""
                )
                context = demo.get("context") or demo.get("input", {}).get(
                    "context", ""
                )
                output = demo.get("answer") or demo.get("output", "")
                score = demo.get("score")
                input_data = {"question": question, "context": context}

            norm_demo: Dict[str, Any] = {
                "module": module_name,
                "input": input_data,
                "output": output,
            }
            if score is not None:
                norm_demo["score"] = score
            if context is not None:
                norm_demo["context"] = context

            normalized.append(norm_demo)

        self.demos = normalized
