from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.config import LLMConfig
from src.llm.client import chat_json
from src.search.rss import RSS_FEEDS, _fetch
from src.search.web_search import unified_search


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
        self.cfg = LLMConfig(api_key="test", base_url="https://example.test/v1", model="deepseek-v4-flash")

    async def test_retries_truncated_json_with_larger_budget(self):
        client, create = _client_with(
            _completion('{"items": [', finish_reason="length", completion_tokens=3000),
            _completion('{"items": []}'),
        )

        result = await chat_json(client, self.cfg, "system", "user", max_tokens=3000)

        self.assertEqual(result, {"items": []})
        self.assertEqual(create.await_count, 2)
        self.assertEqual(create.await_args_list[0].kwargs["max_tokens"], 3000)
        self.assertEqual(
            create.await_args_list[0].kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
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


class SearchFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_ddg_results_without_contacting_direct_sources(self):
        ddg_items = [{"title": "DDG result", "url": "https://example.test/ddg"}]
        direct = AsyncMock()
        with patch("src.search.web_search.ddg_batch_search", new=AsyncMock(return_value=ddg_items)) as ddg, patch(
            "src.search.web_search.fetch_direct_sources", new=direct
        ):
            result = await unified_search(["AI news"], max_results=3)

        self.assertEqual(result, ddg_items)
        ddg.assert_awaited_once_with(["AI news"], max_results=3)
        direct.assert_not_awaited()

    async def test_uses_direct_sources_when_ddg_is_empty(self):
        direct_items = [{"title": "Direct result", "url": "https://example.test/direct"}]
        with patch("src.search.web_search.ddg_batch_search", new=AsyncMock(return_value=[])), patch(
            "src.search.web_search.fetch_direct_sources", new=AsyncMock(return_value=direct_items)
        ) as direct:
            result = await unified_search(["AI news"])

        self.assertEqual(result, direct_items)
        direct.assert_awaited_once_with()
