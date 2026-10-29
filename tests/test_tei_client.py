"""
Offline pytest suite for TEIClient using httpx.MockTransport.
"""

import asyncio
import httpx
import pytest

from app.tei_client import TEIClient, TEIConnectionError, TEIError, TEIResponseError


def test_tei_embed_success():
    """TEI /embed returns list of embedding vectors."""
    async def _run():
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/embed"
            assert request.headers.get("authorization") == "Bearer secret-tei-key"
            return httpx.Response(200, json=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        transport = httpx.MockTransport(handler)
        client = TEIClient(
            embed_url="http://mock-tei:80",
            api_key="secret-tei-key",
            transport=transport,
        )
        result = await client.embed(["text1", "text2"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        await client.close()

    asyncio.run(_run())


def test_tei_rerank_success():
    """TEI /rerank returns ranked candidate scores."""
    async def _run():
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/rerank"
            return httpx.Response(200, json=[
                {"index": 1, "score": 0.88},
                {"index": 0, "score": 0.12},
            ])

        transport = httpx.MockTransport(handler)
        client = TEIClient(
            rerank_url="http://mock-tei:80",
            transport=transport,
        )
        result = await client.rerank("leaking pipe", ["broken chair", "sink leak"])
        assert len(result) == 2
        assert result[0]["index"] == 1
        assert result[0]["score"] == 0.88
        await client.close()

    asyncio.run(_run())


def test_tei_retry_on_connection_error():
    """TEI client retries on connection error before succeeding."""
    async def _run():
        attempts = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise httpx.ConnectError("Connection refused by mock container")
            return httpx.Response(200, json=[[0.9, 0.8]])

        transport = httpx.MockTransport(handler)
        client = TEIClient(
            embed_url="http://mock-tei:80",
            max_retries=2,
            transport=transport,
        )
        result = await client.embed(["test retry"])
        assert attempts == 2
        assert result == [[0.9, 0.8]]
        await client.close()

    asyncio.run(_run())


def test_tei_connection_error_exhausted():
    """TEI client raises TEIConnectionError when all retries fail."""
    async def _run():
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Always fail")

        transport = httpx.MockTransport(handler)
        client = TEIClient(
            embed_url="http://mock-tei:80",
            max_retries=1,
            transport=transport,
        )
        with pytest.raises(TEIConnectionError):
            await client.embed(["test fail"])
        await client.close()

    asyncio.run(_run())


def test_tei_response_error_non_200():
    """TEI client raises TEIResponseError when container returns non-200 status."""
    async def _run():
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal TEI Error")

        transport = httpx.MockTransport(handler)
        client = TEIClient(
            embed_url="http://mock-tei:80",
            transport=transport,
        )
        with pytest.raises(TEIResponseError) as exc_info:
            await client.embed(["bad request"])
        assert "status 500" in str(exc_info.value)
        await client.close()

    asyncio.run(_run())


def test_tei_check_health():
    """check_health() inspects dependencies and never throws."""
    async def _run():
        def handler(request: httpx.Request) -> httpx.Response:
            if "embed" in str(request.url):
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(503, text="Service Unavailable")

        transport = httpx.MockTransport(handler)
        client = TEIClient(
            embed_url="http://tei-embed:80",
            rerank_url="http://tei-rerank:80",
            transport=transport,
        )
        health_res = await client.check_health()
        assert "tei_embed" in health_res
        assert health_res["tei_embed"]["reachable"] is True
        assert health_res["tei_embed"]["status_code"] == 200

        assert "tei_rerank" in health_res
        assert health_res["tei_rerank"]["reachable"] is False
        assert health_res["tei_rerank"]["status_code"] == 503
        await client.close()

    asyncio.run(_run())
