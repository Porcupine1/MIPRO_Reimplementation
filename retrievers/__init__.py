from __future__ import annotations

import os
from typing import Optional

from config import (
    CACHE_DIR,
    CACHE_TTL_HOURS,
    HOPS,
    MAX_SENTS_PER_TITLE,
    RETRIEVER,
    TOP_SENTS_TOTAL,
    TOP_TITLES_HOP1,
    TOP_TITLES_HOP2,
    USE_RETRIEVER_CACHE,
)
from .base import MockRetriever, Passage, Retriever
from .hotpot_context import HotpotContextRetriever
from .wikipedia_api import WikipediaRetriever
from .cache import SimpleCache


def build_cache() -> SimpleCache:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "retriever_cache.json")
    return SimpleCache(cache_path, ttl_hours=CACHE_TTL_HOURS)


def build_retriever(name: Optional[str] = None) -> Retriever:
    """Factory to construct a retriever based on config."""

    selected = (name or RETRIEVER).lower()

    if selected == "mock":
        return MockRetriever()

    if selected == "hotpot_local":
        # Local Hotpot retriever operates directly over the provided example
        # context; it does not need configuration knobs like top_sentences.
        return HotpotContextRetriever()

    if selected == "wiki_online":
        cache = build_cache() if USE_RETRIEVER_CACHE else None
        return WikipediaRetriever(
            hops=HOPS,
            top_titles_hop1=TOP_TITLES_HOP1,
            top_titles_hop2=TOP_TITLES_HOP2,
            top_sentences=TOP_SENTS_TOTAL,
            max_per_title=MAX_SENTS_PER_TITLE,
            cache=cache,
        )

    # Default fallback: use the local Hotpot context retriever.
    # It does not take any configuration arguments.
    return HotpotContextRetriever()


__all__ = [
    "Passage",
    "Retriever",
    "MockRetriever",
    "HotpotContextRetriever",
    "WikipediaRetriever",
    "build_retriever",
]
