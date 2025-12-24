from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from retrievers.base import Passage


@dataclass
class SelectedSentence:
    title: str
    sentence: str
    score: float


class SentenceSelector:
    """Simple lexical-overlap sentence selector."""

    def __init__(self):
        self.token_pattern = re.compile(r"\b\w+\b")

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self.token_pattern.findall(text or "")]

    def _score(
        self, question: str, sentence: str, query: Optional[str] = None
    ) -> float:
        q_tokens = set(self._tokenize(question))
        if query:
            q_tokens |= set(self._tokenize(query))
        s_tokens = set(self._tokenize(sentence))
        if not q_tokens or not s_tokens:
            return 0.0
        overlap = len(q_tokens & s_tokens)
        return overlap / math.sqrt(len(q_tokens) * len(s_tokens))

    def select(
        self,
        question: str,
        passages: Iterable[Passage],
        query: Optional[str] = None,
        total_limit: int = 8,
        per_title_limit: int = 3,
    ) -> List[SelectedSentence]:
        scored: List[SelectedSentence] = []
        for passage in passages:
            taken = 0
            for sent in passage.sentences:
                if taken >= per_title_limit:
                    break
                score = self._score(question, sent, query=query)
                scored.append(SelectedSentence(passage.title, sent.strip(), score))
                taken += 1

        # Keep top by score up to total_limit
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:total_limit]

    @staticmethod
    def pack(selected: List[SelectedSentence]) -> str:
        if not selected:
            return ""
        lines = [
            f"[{item.title} | {item.sentence}]" for item in selected if item.sentence
        ]
        return "\n".join(lines)
