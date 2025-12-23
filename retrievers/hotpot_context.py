from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Passage, Retriever


class HotpotContextRetriever(Retriever):
    """
    Retriever that operates on the provided HotpotQA context.
    """

    name = "hotpot_local"

    def _build_passages(self, example: Dict[str, Any]) -> List[Passage]:
        if "context" not in example:
            raise ValueError("Example must contain 'context'")
        ctx = example["context"]
        sentences = ctx.get("sentences")
        if not sentences:
            raise ValueError("Example context must contain 'sentences'")

        titles = ctx.get("titles") or ctx.get("title") or []
        passages: List[Passage] = []
        for idx, sent_list in enumerate(sentences):
            title = (
                titles[idx]
                if isinstance(titles, list) and idx < len(titles)
                else f"Doc{idx+1}"
            )
            if not isinstance(sent_list, list):
                continue
            passages.append(Passage(title=title, sentences=sent_list, source="example"))
        return passages

    def retrieve(
        self,
        question: str,
        query: Optional[str] = None,
        example: Optional[Dict[str, Any]] = None,
    ) -> List[Passage]:
        if example is None:
            return []
        return self._build_passages(example)
