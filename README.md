# AInformer

**🌐 站点**：https://chdendi.github.io/AInformer

## API

站点提供 JSON API，便于 AI / 爬虫获取结构化数据：

| 端点 | 说明 |
|---|---|
| `/api/v1/manifest.json` | API 元信息、统计、端点列表 |
| `/api/v1/daily/index.json` | 所有日报日期列表 |
| `/api/v1/daily/latest.json` | 最新日报摘要（含 data_url 指向完整数据） |
| `/api/v1/monthly/index.json` | 所有月报列表 |
| `/api/v1/yearly/index.json` | 所有年报列表 |
| `/data/daily/YYYYMMDD.json` | 指定日报完整 JSON |

所有 JSON 响应均附带 `Access-Control-Allow-Origin: *` 头。

## 栏目健康检查

日报 JSON 会在 `section_meta` 中记录每个栏目的素材量、LLM 筛选量、fallback 状态和空栏原因。可用下面的命令检查最近 N 期栏目覆盖情况：

```bash
python -m src.report_health --days 7
```

## 环境变量

日报、月报、年报调用 DeepSeek，在 GitHub Secrets 里配置 `DEEPSEEK_API_KEY`。

检索不需要搜索 API Key。素材来自三条免费链路：

- RSS（OpenAI、The Verge、36kr、arXiv 等）
- DuckDuckGo
- DuckDuckGo 没有结果时，直连新闻源页面

可选：`DEEPSEEK_MODEL`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_THINKING_MODE`、`TZ`、`SITE_BASE_URL`。见 `.env.example`。

## 发布

生成的 HTML / JSON 写到 `gh-pages` 分支。`daily.yml`、`monthly.yml`、`yearly.yml` 会先把该分支上已有的日报、月报、年报和 `data/` 拉回 `docs/`，再生成新报告并推回 `gh-pages`，因此历史页面和最近 7 天的去重数据会保留。`pages.yml` 从 `gh-pages` 部署 GitHub Pages。

`main` 只保留站点配置 `docs/.nojekyll` 和 `docs/_headers`。本地要接着已有档案生成时：

```bash
bash scripts/publish_gh_pages.sh restore
```

## 目录结构

```
AInformer/
├── .github/workflows/
│   ├── ci.yml             push main / pull_request 时运行 pytest
│   ├── daily.yml          每日 09:00 (UTC+8) 生成日报并发布到 gh-pages
│   ├── monthly.yml        每月 1 号 10:00 生成上月月报并发布到 gh-pages
│   ├── yearly.yml         每年 1 月 2 号 11:00 生成上年年报并发布到 gh-pages
│   └── pages.yml          从 gh-pages 部署 GitHub Pages
├── scripts/
│   └── publish_gh_pages.sh  在 gh-pages 上保留历史并发布 docs/
├── src/
│   ├── main_daily.py      日报入口 (python -m src.main_daily)
│   ├── main_monthly.py    月报入口
│   ├── main_yearly.py     年报入口
│   ├── build_index.py     仅重建首页索引
│   ├── config.py          路径 / 环境变量 / 时区
│   ├── llm/client.py      DeepSeek (OpenAI 兼容) 客户端
│   ├── search/
│   │   ├── web_search.py  DuckDuckGo，空结果时直连新闻源
│   │   ├── duckduckgo.py
│   │   ├── news_direct.py
│   │   ├── rss.py         OpenAI/TheVerge/36kr/arXiv 等 RSS
│   │   └── github_trending.py
│   ├── agents/
│   │   ├── definitions.py 4 个 Agent 的 query 集和 RSS 过滤
│   │   └── runner.py      并行执行器
│   ├── dedupe.py          与最近 7 天对比的去重
│   ├── synthesize.py      生成 lede / headlines / daily-takeaway
│   ├── summarize/
│   │   ├── monthly.py
│   │   └── yearly.py
│   └── render/
│       ├── engine.py      Jinja2 + 极简 markdown
│       ├── daily.py
│       ├── summary_pages.py
│       ├── index_page.py
│       └── templates/     base.html.j2 / daily.html.j2 / ...
├── docs/                   ←  生成目录；线上站点在 gh-pages 分支根目录
│   ├── .nojekyll           ←  随 main 跟踪的 Pages 配置
│   └── _headers
├── requirements.txt
├── .env.example
└── README.md
```
