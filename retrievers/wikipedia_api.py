from __future__ import annotations

import itertools
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set

import requests

from config import (
    MAX_WIKI_TITLES_TOTAL,
    WIKI_FETCH_FULL_ARTICLE,
    WIKI_MAX_EXTRACT_CHARS,
    WIKI_PAGINATE_LINKS,
    WIKI_MAX_LINK_ITERATIONS,
)
from .base import Passage, Retriever
from .cache import SimpleCache
from my_selectors import SentenceSelector

logger = logging.getLogger(__name__)


SEARCH_URL = "https://en.wikipedia.org/w/api.php"
EXTRACT_URL = "https://en.wikipedia.org/w/api.php"

# Use a descriptive User-Agent per Wikipedia API guidelines. This also helps
# avoid 403s from some proxies or network middleboxes that block the default
# Python user agent.
WIKI_HEADERS = {
    "User-Agent": "cis0099-ai-sys-is/1.0 (student project; contact: local)",
    "Accept": "application/json",
}


class WikipediaRetriever(Retriever):
    """
    Two-hop Wikipedia retriever with lightweight caching.
    """

    name = "wiki_online"

    def __init__(
        self,
        hops: int = 2,
        top_titles_hop1: int = 4,
        top_titles_hop2: int = 4,
        top_sentences: int = 10,
        max_per_title: int = 3,
        cache: Optional[SimpleCache] = None,
        request_timeout: float = 8.0,
    ):
        self.hops = max(1, hops)
        self.top_titles_hop1 = top_titles_hop1
        self.top_titles_hop2 = top_titles_hop2
        # Keep these for compatibility, but we now return full passages and let
        # downstream components decide how much context to use.
        self.top_sentences = top_sentences
        self.max_per_title = max_per_title
        self.cache = cache
        self.request_timeout = request_timeout
        self.selector = SentenceSelector()

    def _cache_get(self, key: str) -> Optional[Any]:
        if not self.cache:
            return None
        return self.cache.get(key)

    def _cache_set(self, key: str, value: Any) -> None:
        if not self.cache:
            return
        self.cache.set(key, value)

    def _search(self, query: str, limit: int) -> List[str]:
        cache_key = f"search::{query}::{limit}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug(
                "Wikipedia search (cache hit) for %r -> %d titles", query, len(cached)
            )
            return cached

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        try:
            logger.debug("Wikipedia search (API) for %r (limit=%d)", query, limit)
            resp = requests.get(
                SEARCH_URL,
                params=params,
                headers=WIKI_HEADERS,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            titles = [item["title"] for item in data.get("query", {}).get("search", [])]
        except Exception as e:
            # Show long/prompty queries and errors on separate, indented lines
            logger.warning(
                "Wikipedia search failed.\nQuery:\n%s\nError: %s",
                query,
                e,
            )
            titles = []

        self._cache_set(cache_key, titles)
        logger.debug("Wikipedia search for %r returned %d titles", query, len(titles))
        return titles

    def _fetch_extract(self, title: str) -> str:
        cache_key = f"extract::{title}::intro={not WIKI_FETCH_FULL_ARTICLE}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug(
                "Wikipedia extract (cache hit) for %r (len=%d)", title, len(cached)
            )
            return cached

        # Configurable: fetch intro only (fast) or full article (better coverage)
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": not WIKI_FETCH_FULL_ARTICLE,  # False = full article
            "explaintext": True,
            "titles": title,
            "format": "json",
        }

        # Add character limit if fetching full article to prevent excessive text
        if WIKI_FETCH_FULL_ARTICLE and WIKI_MAX_EXTRACT_CHARS:
            params["exchars"] = WIKI_MAX_EXTRACT_CHARS

        try:
            logger.debug(
                "Wikipedia extract (API) for %r (full_article=%s)",
                title,
                WIKI_FETCH_FULL_ARTICLE,
            )
            resp = requests.get(
                EXTRACT_URL,
                params=params,
                headers=WIKI_HEADERS,
                timeout=self.request_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("pages", {})
            # pages is dict keyed by pageid
            extract = next(iter(pages.values()), {}).get("extract", "")
        except Exception as e:
            logger.warning("Wikipedia extract failed for %s: %s", title, e)
            extract = ""

        self._cache_set(cache_key, extract)
        logger.debug(
            "Wikipedia extract for %r produced %d characters", title, len(extract)
        )
        return extract

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        # Basic split; keeps it simple for local use.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _fetch_links(self, title: str) -> List[str]:
        """
        Fetch outgoing links from a Wikipedia page (namespace 0 only).
        Used for hyperlink / link-graph driven bridging.

        If WIKI_PAGINATE_LINKS is True, this will paginate through all links
        using the continuation token. Otherwise, returns only the first batch.
        """
        cache_key = f"links::{title}::paginate={WIKI_PAGINATE_LINKS}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.debug(
                "Wikipedia links (cache hit) for %r -> %d links", title, len(cached)
            )
            return cached

        links: List[str] = []
        params = {
            "action": "query",
            "prop": "links",
            "titles": title,
            "plnamespace": 0,  # main/article namespace only
            "pllimit": "max",
            "format": "json",
        }

        try:
            logger.debug(
                "Wikipedia links (API) for %r (paginate=%s)", title, WIKI_PAGINATE_LINKS
            )

            # Fetch links, optionally paginating through all results
            continue_token = None
            max_iterations = WIKI_MAX_LINK_ITERATIONS if WIKI_PAGINATE_LINKS else 1
            iteration = 0
            hit_limit = False

            while iteration < max_iterations:
                iteration += 1
                request_params = params.copy()

                # Add continuation token if we have one
                if continue_token:
                    request_params.update(continue_token)

                resp = requests.get(
                    EXTRACT_URL,
                    params=request_params,
                    headers=WIKI_HEADERS,
                    timeout=self.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                # Extract links from response
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    for link in page.get("links", []) or []:
                        link_title = link.get("title")
                        if link_title:
                            links.append(link_title)

                # Check if there are more results
                if not WIKI_PAGINATE_LINKS or "continue" not in data:
                    break

                continue_token = data["continue"]
                logger.debug(
                    "Wikipedia links pagination: fetched %d links so far, continuing...",
                    len(links),
                )

                # Check if we hit the iteration limit
                if iteration >= max_iterations and "continue" in data:
                    hit_limit = True

            if hit_limit:
                logger.warning(
                    "Wikipedia links pagination hit max iteration limit (%d) for %r; "
                    "some links may be missing. Fetched %d links total.",
                    max_iterations,
                    title,
                    len(links),
                )

        except Exception as e:
            logger.warning("Wikipedia links fetch failed for %s: %s", title, e)

        self._cache_set(cache_key, links)
        logger.debug("Wikipedia links for %r produced %d links", title, len(links))
        return links

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        pattern = re.compile(r"\b\w+\b")
        return {t.lower() for t in pattern.findall(text or "")}

    def _extract_bridge_titles(
        self,
        question: str,
        hop1_titles: List[str],
        limit: int = 3,
    ) -> List[str]:
        """
        Use outgoing links from hop-1 pages as bridge candidates.

        Strategy:
        - Fetch links for each hop-1 title.
        - Score each linked title by lexical overlap with question tokens.
        - Keep the top-k unique linked titles as bridge pages.
        """
        q_tokens = self._tokenize(question)
        if not q_tokens:
            return []

        scored: List[tuple[float, str]] = []
        seen_links: Set[str] = set()

        for src_title in hop1_titles:
            links = self._fetch_links(src_title)
            for link_title in links:
                if link_title in seen_links:
                    continue
                seen_links.add(link_title)
                link_tokens = self._tokenize(link_title)
                if not link_tokens:
                    continue
                overlap = len(q_tokens & link_tokens)
                if overlap == 0:
                    continue
                # Simple normalized overlap score
                score = overlap / math.sqrt(len(q_tokens) * len(link_tokens))
                scored.append((score, link_title))

        if not scored:
            logger.debug("No bridge titles found from links for question: %s", question)
            return []

        # Sort by score descending and keep top unique titles
        scored.sort(key=lambda x: x[0], reverse=True)
        top_titles: List[str] = []
        for _, title in scored:
            if title not in top_titles:
                top_titles.append(title)
            if len(top_titles) >= limit:
                break
        logger.debug("Selected bridge titles: %s", top_titles)
        return top_titles

    def _fetch_passages_for_titles(self, titles: List[str]) -> List[Passage]:
        passages: List[Passage] = []
        for title in titles[:MAX_WIKI_TITLES_TOTAL]:
            extract = self._fetch_extract(title)
            if not extract:
                continue
            sentences = self._split_sentences(extract)
            passages.append(
                Passage(title=title, sentences=sentences, source="wikipedia")
            )
        return passages

    def retrieve(
        self,
        question: str,
        query: Optional[str] = None,
        example: Optional[Dict[str, Any]] = None,
    ) -> List[Passage]:
        seed_query = query or question
        logger.debug(
            "WikipediaRetriever: starting retrieval for question: %s", question
        )
        hop1_titles = self._search(seed_query, self.top_titles_hop1)
        logger.debug("WikipediaRetriever: hop1 titles: %s", hop1_titles)
        hop1_passages = self._fetch_passages_for_titles(hop1_titles)

        if self.hops <= 1 or not hop1_passages:
            logger.debug(
                "WikipediaRetriever: using hop1 only (hops=%d, hop1_passages=%d)",
                self.hops,
                len(hop1_passages),
            )
            # Return full hop1 passages without truncation.
            return hop1_passages

        # Hyperlink / link-graph driven bridging:
        # use outgoing links from hop-1 pages whose titles overlap with the question.
        bridge_titles = self._extract_bridge_titles(question, hop1_titles)
        logger.debug("WikipediaRetriever: bridge titles: %s", bridge_titles)
        hop2_titles = list(
            itertools.chain.from_iterable(
                self._search(f"{question} {title}", self.top_titles_hop2)
                for title in bridge_titles
            )
        )

        # Deduplicate titles while keeping order
        seen = set()
        merged_titles = []
        for t in hop1_titles + hop2_titles:
            if t not in seen:
                seen.add(t)
                merged_titles.append(t)

        passages = self._fetch_passages_for_titles(merged_titles)
        logger.debug("WikipediaRetriever: returning %d passages total", len(passages))
        # Return full passages; do not truncate to top sentences here.
        return passages

    def selector_collapse(
        self, question: str, query: Optional[str], passages: List[Passage]
    ) -> List[Passage]:
        """Apply sentence selection and return single-sentence passages.

        This helper is retained for backward compatibility but is no longer
        used in the main retrieval path now that we pass full context through.
        """
        selected = self.selector.select(
            question=question,
            query=query,
            passages=passages,
            total_limit=self.top_sentences,
            per_title_limit=self.max_per_title,
        )
        return [
            Passage(title=item.title, sentences=[item.sentence], source="wikipedia")
            for item in selected
        ]
