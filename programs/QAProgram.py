from typing import List, Dict, Any, Optional
import threading
import copy
import logging
from backend import LMBackend
from retrievers import build_retriever, Retriever
from my_selectors import SentenceSelector
from config import TOP_SENTS_TOTAL, MAX_SENTS_PER_TITLE
from QADataset import QADataset
from .PromptMod import PromptModule, QueryModule, AnswerModule


logger = logging.getLogger(__name__)


class QAProgram:
    """multi-stage QA pipeline: query generation -> document retrieval -> answer generation"""

    def __init__(
        self,
        backend: Optional[LMBackend] = None,
        modules: Optional[Dict[str, PromptModule]] = None,
        retriever: Optional[Retriever] = None,
        selector: Optional[SentenceSelector] = None,
    ):
        self.backend = backend or LMBackend()
        self.modules = modules or {"query": QueryModule(), "answer": AnswerModule()}
        self.retriever = retriever or build_retriever()
        self.selector = selector or SentenceSelector()
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

        logger.debug("Generating query for question: %s", question[:200])
        if "query" not in self.modules:
            raise ValueError("Query module not found in program")
        module = self.modules["query"]
        query = module.forward(question=question)
        # Record trace
        self.trace.append(
            {"module": module.name, "input": {"question": question}, "output": query}
        )
        return query

    def retrieve_context(
        self, question: str, query: str, example: Dict[str, Any]
    ) -> str:
        """Retrieve context using the configured retriever and pack selected sentences."""

        logger.debug(
            "Retrieving context with retriever '%s' for question: %s | query: %s",
            getattr(self.retriever, "name", type(self.retriever).__name__),
            question[:200],
            query[:200],
        )
        passages = self.retriever.retrieve(
            question=question, query=query, example=example
        )
        logger.debug("Retriever returned %d passages", len(passages))
        # For the local Hotpot retriever, do not score sentences – just pack all
        # sentences in title-tagged order. This preserves the provided context
        # structure while still giving the answer module a clean format.
        if getattr(self.retriever, "name", "") == "hotpot_local":
            lines = []
            for passage in passages:
                lines.extend(passage.as_tagged_sentences())
            context = "\n".join(lines)
            logger.debug(
                "Packed local context with %d lines and %d characters",
                len(lines),
                len(context),
            )
            return context

        # For other retrievers (e.g., Wikipedia, mock), apply sentence selection
        # to reduce noise and keep only the top-K evidence sentences.
        selected = self.selector.select(
            question=question,
            query=query,
            passages=passages,
            total_limit=TOP_SENTS_TOTAL,
            per_title_limit=MAX_SENTS_PER_TITLE,
        )
        context = self.selector.pack(selected)
        logger.debug(
            "Packed selected context with %d sentences and %d characters",
            len(selected),
            len(context),
        )
        return context

    def generate_answer(self, question: str, context: str) -> str:
        """generate answer from question and context"""

        if "answer" not in self.modules:
            raise ValueError("Answer module not found in program")
        module = self.modules["answer"]
        logger.debug(
            "Generating answer with context length %d chars for question: %s",
            len(context),
            question[:200],
        )
        answer = module.forward(question=question, context=context)
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
        logger.info("QAProgram forward start. Question: %s", question[:200])

        query = self.generate_query(question)
        context = self.retrieve_context(question, query, example)
        answer = self.generate_answer(question, context)
        logger.info("QAProgram forward complete. Answer (truncated): %s", answer[:200])
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

        cloned_modules = {name: module.clone() for name, module in self.modules.items()}
        cloned = QAProgram(
            backend=self.backend,
            modules=cloned_modules,
            retriever=self.retriever,
            selector=self.selector,
        )
        cloned.trace = []  # Initialize empty trace
        return cloned
