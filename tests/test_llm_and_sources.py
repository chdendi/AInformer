from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.config import LLMConfig
from src.llm.client import chat_json
from src.search import tavily
from src.search.rss import RSS_FEEDS, _fetch


def _completion(content: str, *, finish_reason: str = "stop", completion_tokens: int = 100):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=""),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=completion_tokens),
    )


def _client_with(*responses):
    create = AsyncMock(side_effect=responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), create


class ChatJsonTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cfg = LLMConfig(api_key="test", base_url="https://example.test/v1", model="deepseek-chat")

    async def test_retries_truncated_json_with_larger_budget(self):
        client, create = _client_with(
            _completion('{"items": [', finish_reason="length", completion_tokens=3000),
            _completion('{"items": []}'),
        )

        result = await chat_json(client, self.cfg, "system", "user", max_tokens=3000)

        self.assertEqual(result, {"items": []})
        self.assertEqual(create.await_count, 2)
        self.assertEqual(create.await_args_list[0].kwargs["max_tokens"], 3000)
        self.assertEqual(create.await_args_list[1].kwargs["max_tokens"], 6000)
        self.assertEqual(create.await_args_list[1].kwargs["response_format"], {"type": "json_object"})

    async def test_retries_without_json_mode_when_output_is_not_truncated(self):
        client, create = _client_with(
            _completion("not json"),
            _completion('{"items": []}'),
        )

        result = await chat_json(client, self.cfg, "system", "user", max_tokens=2000)

        self.assertEqual(result, {"items": []})
        self.assertEqual(create.await_count, 2)
        self.assertNotIn("response_format", create.await_args_list[1].kwargs)


class RssFetchTests(unittest.IsolatedAsyncioTestCase):
    def test_uses_verified_rss_endpoints(self):
        self.assertEqual(RSS_FEEDS["wired_ai"]["url"], "https://www.wired.com/feed/tag/ai/latest/rss")
        self.assertEqual(RSS_FEEDS["geekpark"]["url"], "https://www.geekpark.net/rss")
        self.assertNotIn("microsoft_ai", RSS_FEEDS)
        self.assertNotIn("36kr_newsflash", RSS_FEEDS)

    async def test_rejects_html_that_has_no_feed_entries(self):
        response = SimpleNamespace(
            status_code=200,
            text="<html><body>not a feed</body></html>",
            headers={"content-type": "text/html"},
        )
        client = SimpleNamespace(get=AsyncMock(return_value=response))

        _, parsed = await _fetch(client, "example", {"url": "https://example.test/feed"})

        self.assertIsNone(parsed)

    async def test_keeps_a_valid_rss_feed(self):
        response = SimpleNamespace(
            status_code=200,
            text=(
                "<rss version=\"2.0\"><channel><title>Example</title><item>"
                "<title>Item</title><link>https://example.test/item</link>"
                "</item></channel></rss>"
            ),
            headers={"content-type": "application/rss+xml"},
        )
        client = SimpleNamespace(get=AsyncMock(return_value=response))

        _, parsed = await _fetch(client, "example", {"url": "https://example.test/feed"})

        self.assertIsNotNone(parsed)
        self.assertEqual(len(parsed.entries), 1)


class _TavilyResponse:
    status_code = 432
    text = '{"detail":"credits exhausted"}'

    def raise_for_status(self):
        raise AssertionError("permanent Tavily errors should not be retried")


class _TavilyHttpClient:
    def __init__(self):
        self.post = AsyncMock(return_value=_TavilyResponse())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class TavilyTests(unittest.IsolatedAsyncioTestCase):
    async def test_disables_search_for_run_after_permanent_failure(self):
        tavily._DISABLED_REASON = ""
        tavily._LAST_REQUEST_AT = 0.0
        client = _TavilyHttpClient()

        with patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}, clear=False), patch.object(
            tavily.httpx, "AsyncClient", return_value=client
        ):
            first = await tavily.tavily_search("first")
            second = await tavily.tavily_search("second")

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(client.post.await_count, 1)
        self.assertIn("HTTP 432", tavily._DISABLED_REASON)
        tavily._DISABLED_REASON = ""
