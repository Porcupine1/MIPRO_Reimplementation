from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Passage:
    """Lightweight representation of a document snippet."""

    title: str
    sentences: List[str]
    source: str = "local"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_tagged_sentences(self) -> List[str]:
        """Format sentences with a title tag for downstream packing."""
        tagged = []
        for sent in self.sentences:
            cleaned = sent.strip()
            if not cleaned:
                continue
            tagged.append(f"[{self.title} | {cleaned}]")
        return tagged


class Retriever(abc.ABC):
    """Abstract retriever interface."""

    name: str = "base"

    @abc.abstractmethod
    def retrieve(
        self,
        question: str,
        query: Optional[str] = None,
        example: Optional[Dict[str, Any]] = None,
    ) -> List[Passage]:
        """Return a list of passages relevant to the question/query."""
        raise NotImplementedError


class MockRetriever(Retriever):
    """
    Baseline retriever that flattens provided example context.

    Useful for A/B against the previous behavior.
    """

    name = "mock"

    def retrieve(
        self,
        question: str,
        query: Optional[str] = None,
        example: Optional[Dict[str, Any]] = None,
    ) -> List[Passage]:
        if example is None:
            return []
        context = example.get("context", {})
        sentences = []
        if isinstance(context, dict) and "sentences" in context:
            for ctx in context["sentences"]:
                sentences.extend(ctx)
        elif isinstance(context, list):
            for item in context:
                if isinstance(item, list):
                    sentences.extend(item)
                elif isinstance(item, str):
                    sentences.append(item)
        else:
            return []

        return [Passage(title="context", sentences=sentences, source="example")]
