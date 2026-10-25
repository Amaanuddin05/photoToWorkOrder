"""
Async httpx client for Text Embeddings Inference (TEI) services.
Communicates with TEI embedder (/embed) and TEI reranker (/rerank).
Includes timeouts, retries on connection failures, and custom exceptions.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger("siglip.tei_client")


class TEIError(Exception):
    """Base exception for all TEI client errors."""
    pass


class TEIConnectionError(TEIError):
    """Raised when connection to TEI container fails or times out after retries."""
    pass


class TEIResponseError(TEIError):
    """Raised when TEI returns an unexpected HTTP status code or payload."""
    pass


class TEIClient:
    """Async client interfacing with TEI embedder and reranker containers."""

    def __init__(
        self,
        embed_url: str = "http://tei-embed:80",
        rerank_url: str = "http://tei-rerank:80",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 2,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.embed_url = embed_url.rstrip("/")
        self.rerank_url = rerank_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                transport=self.transport,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _post_with_retry(self, url: str, json_body: Dict[str, Any]) -> Any:
        client = self._get_client()
        headers = self._get_headers()
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = await client.post(url, json=json_body, headers=headers)
                if resp.status_code != 200:
                    raise TEIResponseError(
                        f"TEI request to {url} returned status {resp.status_code}: {resp.text}"
                    )
                return resp.json()
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    backoff = 0.2 * (2 ** attempt)
                    logger.warning(
                        "Connection error calling TEI at %s (attempt %d/%d). Retrying in %.2fs: %s",
                        url,
                        attempt + 1,
                        self.max_retries,
                        backoff,
                        exc,
                    )
                    await asyncio.sleep(backoff)
                else:
                    break
            except TEIResponseError:
                raise
            except Exception as exc:
                raise TEIResponseError(f"Unexpected error calling TEI at {url}: {exc}") from exc

        raise TEIConnectionError(
            f"Failed to connect to TEI service at {url} after {self.max_retries + 1} attempts: {last_exc}"
        ) from last_exc

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Calls TEI embed endpoint: POST {embed_url}/embed
        Input format: {"inputs": ["text1", "text2"]}
        Output format: List of vectors [[float, ...], ...]
        """
        if not texts:
            return []
        url = f"{self.embed_url}/embed"
        data = await self._post_with_retry(url, {"inputs": texts})
        
        # TEI returns a list of float arrays
        if not isinstance(data, list):
            raise TEIResponseError(f"Expected list response from TEI /embed, got {type(data)}")
        
        # In case a single vector [float, ...] was returned when passing 1 element:
        if data and not isinstance(data[0], list) and isinstance(data[0], (int, float)):
            return [data]  # type: ignore

        return data

    async def rerank(self, query: str, texts: List[str], raw_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Calls TEI rerank endpoint: POST {rerank_url}/rerank
        Input format: {"query": query, "texts": texts, "raw_scores": raw_scores}
        Output format: List of dicts, e.g. [{"index": int, "score": float}, ...]
        """
        if not texts:
            return []
        url = f"{self.rerank_url}/rerank"
        payload: Dict[str, Any] = {"query": query, "texts": texts}
        if raw_scores:
            payload["raw_scores"] = True
        data = await self._post_with_retry(url, payload)
        if not isinstance(data, list):
            raise TEIResponseError(f"Expected list response from TEI /rerank, got {type(data)}")
        return data

    async def get_info(self, service: str = "embed") -> Dict[str, Any]:
        """Fetches /info from embed or rerank TEI container."""
        base_url = self.embed_url if service == "embed" else self.rerank_url
        client = self._get_client()
        headers = self._get_headers()
        url = f"{base_url}/info"
        resp = await client.get(url, headers=headers)
        if resp.status_code != 200:
            raise TEIResponseError(f"TEI /info from {url} returned status {resp.status_code}: {resp.text}")
        return resp.json()

    async def check_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Checks health of both TEI embed and rerank endpoints.
        Never throws; returns dictionary of reachability and latencies.
        """
        client = self._get_client()
        headers = self._get_headers()
        results: Dict[str, Dict[str, Any]] = {}

        for name, base_url in [("tei_embed", self.embed_url), ("tei_rerank", self.rerank_url)]:
            url = f"{base_url}/health"
            t0 = time.perf_counter()
            try:
                resp = await client.get(url, headers=headers, timeout=2.0)
                latency = round((time.perf_counter() - t0) * 1000.0, 2)
                reachable = (resp.status_code == 200)
                results[name] = {
                    "url": base_url,
                    "reachable": reachable,
                    "status_code": resp.status_code,
                    "latency_ms": latency,
                    "error": None if reachable else f"Status code {resp.status_code}",
                }
            except Exception as exc:
                latency = round((time.perf_counter() - t0) * 1000.0, 2)
                results[name] = {
                    "url": base_url,
                    "reachable": False,
                    "status_code": None,
                    "latency_ms": latency,
                    "error": str(exc),
                }

        return results

