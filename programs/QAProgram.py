from typing import List, Dict, Any, Optional
import threading
import copy
import logging
from backend import LMBackend
from retrievers import build_retriever, Retriever
from QADataset import QADataset
from config import MAX_CONTEXT_CHARS
from .PromptMod import PromptModule, QueryModule, AnswerModule


logger = logging.getLogger(__name__)


class QAProgram:
    """multi-stage QA pipeline: query generation -> document retrieval -> answer generation"""

    def __init__(
        self,
        backend: Optional[LMBackend] = None,
        modules: Optional[Dict[str, PromptModule]] = None,
        retriever: Optional[Retriever] = None,
        selector: Optional[object] = None,
    ):
        self.backend = backend or LMBackend()
        self.modules = modules or {"query": QueryModule(), "answer": AnswerModule()}
        self.retriever = retriever or build_retriever()
        # Selector is kept for backward compatibility but is no longer used for
        # truncating context – we now pass the full retrieved context through.
        self.selector = selector
        # Thread-local trace storage so concurrent batch execution does not share state.
        self._trace_local = threading.local()
        self.trace = []  # List of module traces per thread

    def _get_thread_trace(self) -> List[Dict[str, Any]]:
        """Return the trace list scoped to the current thread."""
        if not hasattr(self._trace_local, "trace"):
            self._trace_local.trace = []
        return self._trace_local.trace

    @property
    def trace(self) -> List[Dict[str, Any]]:
        return self._get_thread_trace()

    @trace.setter
    def trace(self, value: List[Dict[str, Any]]):
        self._trace_local.trace = value

    def generate_query(self, question: str) -> str:
        """generate search query from question"""

        logger.debug("Generating query for question: %s", question)
        if "query" not in self.modules:
            raise ValueError("Query module not found in program")
        module = self.modules["query"]
        
        # Build and log the prompt
        prompt = module.build_prompt(question=question)
        logger.instr("\n" + "="*80)
        logger.instr("QUERY GENERATION PROMPT:")
        logger.instr("-"*80)
        logger.instr("%s", prompt)
        logger.instr("-"*80)
        
        # Generate the query
        query = module.forward(question=question)
        
        logger.instr("GENERATED QUERY: %s", query)
        logger.instr("="*80 + "\n")
        
        # Record trace
        self.trace.append(
            {"module": module.name, "input": {"question": question}, "output": query}
        )
        return query

    def retrieve_context(
        self, question: str, query: str, example: Dict[str, Any]
    ) -> str:
        """
        Retrieve context using the configured retriever and pack retrieved
        sentences, truncating to MAX_CONTEXT_CHARS to prevent exceeding model limits.
        Uses sentence-aware truncation to avoid cutting mid-tag or mid-sentence.
        """

        logger.debug(
            "Retrieving context with retriever '%s' for question: %s | query: %s",
            getattr(self.retriever, "name", type(self.retriever).__name__),
            question,
            query,
        )
        passages = self.retriever.retrieve(
            question=question, query=query, example=example
        )
        logger.debug("Retriever returned %d passages", len(passages))

        # Pack all sentences from all passages, tagged by title.
        lines = []
        for passage in passages:
            if hasattr(passage, "as_tagged_sentences"):
                lines.extend(passage.as_tagged_sentences())
            elif getattr(passage, "sentences", None):
                lines.extend(
                    f"[{getattr(passage, 'title', '')} | {s.strip()}]"
                    for s in passage.sentences
                    if s and s.strip()
                )

        # Sentence-aware truncation: keep complete tagged sentences up to the limit
        if not lines:
            return ""

        total_chars = sum(len(line) + 1 for line in lines)  # +1 for newline
        if total_chars <= MAX_CONTEXT_CHARS:
            context = "\n".join(lines)
        else:
            # Truncate by keeping complete lines (sentences) until we hit the limit
            kept_lines = []
            current_length = 0
            for line in lines:
                line_length = len(line) + 1  # +1 for newline
                if current_length + line_length > MAX_CONTEXT_CHARS:
                    break
                kept_lines.append(line)
                current_length += line_length

            context = "\n".join(kept_lines)
            logger.debug(
                "Context truncated (sentence-aware): kept %d/%d lines, %d/%d chars",
                len(kept_lines),
                len(lines),
                len(context),
                total_chars - 1,  # -1 to account for final newline
            )

        logger.debug(
            "Packed context with %d lines and %d characters",
            len(context.split("\n")),
            len(context),
        )
        
        # Log the retrieved context
        logger.instr("\n" + "="*80)
        logger.instr("RETRIEVED CONTEXT (%d chars, %d lines):", len(context), len(context.split("\n")))
        logger.instr("-"*80)
        # Truncate context display if too long (show first 1000 chars)
        if len(context) > 1000:
            logger.instr("%s\n... [truncated, %d more chars] ...", context[:1000], len(context) - 1000)
        else:
            logger.instr("%s", context)
        logger.instr("="*80 + "\n")
        
        return context

    def generate_answer(self, question: str, context: str) -> str:
        """generate answer from question and context"""

        if "answer" not in self.modules:
            raise ValueError("Answer module not found in program")
        module = self.modules["answer"]
        logger.debug(
            "Generating answer with context length %d chars for question: %s",
            len(context),
            question,
        )
        
        # Build and log the prompt
        prompt = module.build_prompt(question=question, context=context)
        logger.instr("\n" + "="*80)
        logger.instr("ANSWER GENERATION PROMPT:")
        logger.instr("-"*80)
        # Truncate prompt display if too long (show first 1500 chars)
        if len(prompt) > 1500:
            logger.instr("%s\n... [truncated, %d more chars] ...", prompt[:1500], len(prompt) - 1500)
        else:
            logger.instr("%s", prompt)
        logger.instr("-"*80)
        
        # Generate the answer
        answer = module.forward(question=question, context=context)
        
        logger.instr("GENERATED ANSWER: %s", answer)
        logger.instr("="*80 + "\n")
        
        # Record trace
        self.trace.append(
            {
                "module": module.name,
                "input": {"question": question, "context": context},
                "output": answer,
            }
        )
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
        
        logger.instr("\n" + "="*80)
        logger.instr("QA PIPELINE RUN STARTED")
        logger.instr("="*80)
        logger.instr("QUESTION: %s", question)
        logger.instr("="*80)

        query = self.generate_query(question)
        context = self.retrieve_context(question, query, example)
        answer = self.generate_answer(question, context)
        
        logger.instr("\n" + "="*80)
        logger.instr("QA PIPELINE COMPLETED")
        logger.instr("Question: %s", question)
        logger.instr("Query: %s", query)
        logger.instr("Answer: %s", answer)
        logger.instr("="*80 + "\n")
        
        return answer

    def process_batch(
        self, batch: List[Dict[str, Any]], parallel: bool = True
    ) -> List[str]:
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
                    # Avoid importing logging here to keep this module lightweight;
                    # callers can handle/log empty predictions if needed.
                    predictions[idx] = ""

        return predictions

    def apply_configuration(
        self,
        instructions: Dict[str, str],
        demos: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
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

        # Create a NEW backend instance for thread safety
        # Each clone needs its own LMBackend to avoid race conditions in parallel execution
        cloned_backend = LMBackend(
            model_name=self.backend.model_name,
            base_url=self.backend.base_url,
            temperature=self.backend.temperature,
            max_tokens=self.backend.max_tokens,
            max_workers=self.backend.max_workers,
            request_timeout=self.backend.request_timeout,
        )
        
        # Clone modules with the new backend
        cloned_modules = {}
        for name, module in self.modules.items():
            cloned_module = module.clone()
            # Update the module to use the new backend
            cloned_module.lm = cloned_backend
            cloned_modules[name] = cloned_module
        
        cloned = QAProgram(
            backend=cloned_backend,
            modules=cloned_modules,
            retriever=self.retriever,
            selector=self.selector,
        )
        cloned.trace = []  # Initialize empty trace
        return cloned
