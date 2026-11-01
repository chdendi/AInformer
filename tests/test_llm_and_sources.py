from __future__ import annotations

import asyncio
import unittest
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.config import LLMConfig
from src import main_daily
from src.llm.client import chat_json, chat_text, get_usage_summary, reset_usage
from src.search.rss import RSS_FEEDS, _fetch
from src.search.web_search import unified_search
from src.summarize import monthly, yearly


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


class _LlmTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await reset_usage()
        self.addAsyncCleanup(reset_usage)
        self.cfg = LLMConfig(api_key="test", base_url="https://example.test/v1", model="deepseek-v4-flash")


class ChatJsonTests(_LlmTestCase):
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
        self.assertEqual(
            get_usage_summary(),
            {"calls": 2, "prompt_tokens": 20, "completion_tokens": 3100, "total_tokens": 3120},
        )

    async def test_retries_without_json_mode_when_output_is_not_truncated(self):
        client, create = _client_with(
            _completion("not json"),
            _completion('{"items": []}'),
        )

        result = await chat_json(client, self.cfg, "system", "user", max_tokens=2000)

        self.assertEqual(result, {"items": []})
        self.assertEqual(create.await_count, 2)
        self.assertNotIn("response_format", create.await_args_list[1].kwargs)


class UsageIsolationTests(_LlmTestCase):
    async def test_accumulates_parallel_json_and_text_requests(self):
        client, create = _client_with(
            _completion('{"items": []}', completion_tokens=20),
            _completion("Summary", completion_tokens=30),
        )

        await asyncio.gather(
            chat_json(client, self.cfg, "system", "user"),
            chat_text(client, self.cfg, "system", "user"),
        )

        self.assertEqual(create.await_count, 2)
        self.assertEqual(
            get_usage_summary(),
            {"calls": 2, "prompt_tokens": 20, "completion_tokens": 50, "total_tokens": 70},
        )

    async def test_reset_clears_usage_and_preserves_previous_snapshot(self):
        client, _ = _client_with(_completion("Summary", completion_tokens=20))
        await chat_text(client, self.cfg, "system", "user")
        previous = get_usage_summary()

        await reset_usage()
        await reset_usage()

        self.assertEqual(
            get_usage_summary(),
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        self.assertEqual(
            previous,
            {"calls": 1, "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )

    async def test_repeated_monthly_and_yearly_batches_have_independent_usage(self):
        reports = [{"date": "2026-10-01", "sections": {"industry": [{"title": "Example"}]}}]
        for module, build, args in (
            (monthly, monthly.build_monthly, (2026, 10)),
            (yearly, yearly.build_yearly, (2026,)),
        ):
            with self.subTest(report=module.__name__), ExitStack() as stack:
                client, create = _client_with(
                    _completion('{"overview": "First report"}', completion_tokens=20),
                    _completion('{"overview": "Second report"}', completion_tokens=30),
                )
                stack.enter_context(patch.object(module, "AsyncOpenAI", return_value=client))
                stack.enter_context(patch.object(module, "load_daily_in_range", return_value=reports))
                if module is yearly:
                    stack.enter_context(patch.object(module, "load_monthly_in_year", return_value=[]))

                for tokens in (20, 30):
                    await build(*args, self.cfg)
                    self.assertEqual(
                        get_usage_summary(),
                        {"calls": 1, "prompt_tokens": 10, "completion_tokens": tokens, "total_tokens": 10 + tokens},
                    )
                self.assertEqual(create.await_count, 2)

    async def test_daily_dry_run_does_not_inherit_previous_batch_usage(self):
        output = Path("/tmp/ainformer-test/docs/data/daily/20261001.json")
        with ExitStack() as stack:
            stack.enter_context(patch.object(main_daily, "ensure_dirs"))
            stack.enter_context(patch.object(main_daily.LLMConfig, "from_env", return_value=self.cfg))
            stack.enter_context(patch.object(main_daily, "make_client"))
            stack.enter_context(patch.object(main_daily, "load_recent_reports", return_value=[]))
            stack.enter_context(patch.object(main_daily, "collect_rss", new=AsyncMock(return_value=[])))
            stack.enter_context(patch.object(main_daily, "write_daily_json", return_value=output))
            stack.enter_context(patch.object(main_daily, "write_daily_html", return_value=output.with_suffix(".html")))
            stack.enter_context(patch.object(main_daily, "write_index"))
            stack.enter_context(patch.object(main_daily, "build_api"))

            for _ in range(2):
                client, _ = _client_with(_completion("Previous batch", completion_tokens=20))
                await chat_text(client, self.cfg, "system", "user")
                await main_daily.main_async(date(2026, 10, 1), dedupe_days=7, dry=True)
                self.assertEqual(
                    get_usage_summary(),
                    {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                )


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
