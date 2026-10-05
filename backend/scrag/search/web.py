"""Web Search Fallback Client: DuckDuckGo (Free, zero API key) and Tavily."""

import logging
from typing import List, Optional

from scrag.config import SCRAGSettings, get_settings
from scrag.retrieval.loaders import DocumentChunk

logger = logging.getLogger("SCRAG.WebSearch")


class WebSearchClient:
    """Performs live web searches using DuckDuckGo (free) with Tavily support."""

    def __init__(self, settings: Optional[SCRAGSettings] = None):
        self.settings = settings or get_settings()
        self.tavily_client = None

        if self.settings.search_provider == "tavily" and self.settings.tavily_api_key:
            try:
                from tavily import TavilyClient

                self.tavily_client = TavilyClient(api_key=self.settings.tavily_api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Tavily client: {e}")

    def search(self, query: str, max_results: int = 4) -> List[DocumentChunk]:
        """Search the web and return document chunks."""
        # 1. Use Tavily if explicitly configured
        if self.tavily_client is not None:
            try:
                response = self.tavily_client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="basic",
                )
                results = response.get("results", [])
                chunks = []
                for idx, r in enumerate(results):
                    content = r.get("content") or r.get("snippet", "")
                    url = r.get("url", f"web_result_{idx}")
                    title = r.get("title", "")
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"web_tavily_{idx}",
                            content=f"{title}: {content}".strip(),
                            metadata={
                                "source": url,
                                "title": title,
                                "is_web": True,
                                "provider": "tavily",
                            },
                        )
                    )
                if chunks:
                    return chunks
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to DuckDuckGo.")

        # 2. Free DuckDuckGo search (zero API key)
        return self._search_duckduckgo(query, max_results=max_results)

    def _search_duckduckgo(
        self, query: str, max_results: int = 4
    ) -> List[DocumentChunk]:
        """Query DuckDuckGo for public web search snippets."""
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from duckduckgo_search import DDGS

            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ddgs = DDGS()
                raw_results = list(ddgs.text(query, max_results=max_results))

            chunks = []
            for idx, r in enumerate(raw_results):
                body = r.get("body", "")
                url = r.get("href", f"ddg_result_{idx}")
                title = r.get("title", "")
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"web_ddg_{idx}",
                        content=f"{title}: {body}".strip(),
                        metadata={
                            "source": url,
                            "title": title,
                            "is_web": True,
                            "provider": "duckduckgo",
                        },
                    )
                )
            return chunks
        except Exception as e:
            logger.error(f"DuckDuckGo search fallback failed: {e}")
            return []
