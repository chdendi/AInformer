# AInformer

由 DeepSeek 生成的中文 AI 日报、月报和年报。网页搜索采用 `DuckDuckGo → news_direct` 免费回退链路：DuckDuckGo 无结果时直接抓取新闻源，并结合 RSS 和 GitHub Trending 收集素材。搜索无需 API Key，内容生成需要 DeepSeek API Key。

**🌐 站点**：https://chdendi.github.io/AInformer

## 本地运行

使用 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，填写 `DEEPSEEK_API_KEY` 和实际的 `SITE_BASE_URL`。默认模型为 `deepseek-v4-flash`，`DEEPSEEK_THINKING_MODE=disabled`；可用值为 `disabled`、`enabled`。

程序只读取环境变量，不会自动加载 `.env`。文件采用 shell 赋值语法；编辑完成后在当前终端导出配置：

```bash
set -a
source .env
set +a
```

生成前恢复已有站点数据，供跨日去重、月报和年报聚合使用：

```bash
bash scripts/publish_gh_pages.sh restore
python -m src.main_daily
```

也可以指定日期或生成汇总：

```bash
python -m src.main_daily --date 2026-11-01
python -m src.main_monthly --year 2026 --month 10
python -m src.main_yearly --year 2025
```

结果写入本地 `docs/`。预览站点可运行 `python -m http.server 8000 --directory docs`，然后打开 http://localhost:8000。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

Push / PR 到 `main` 时，CI 使用 Python 3.11 自动运行测试。测试模拟外部服务，不需要真实 API Key。

## GitHub Actions 与发布

`main` 保存源码、工作流和 `site/` 中的静态配置，`gh-pages` 保存完整站点。生成的 HTML、JSON 和本地 `docs/` 不再提交到 `main`。

在仓库 Settings 中配置：

- Secrets：`DEEPSEEK_API_KEY`。
- Variables：`SITE_BASE_URL`；按需设置 `DEEPSEEK_MODEL`、`DEEPSEEK_THINKING_MODE`、`DEEPSEEK_BASE_URL`。
- Pages：部署来源选择 **GitHub Actions**。
- Actions：允许工作流写入仓库，以便更新 `gh-pages`。

日报每天 09:00、月报每月 1 日 10:00、年报每年 1 月 2 日 11:00 自动运行（均为 UTC+8），也可在 Actions 页面手动触发并指定日期。

三个生成工作流共用 `publish.yml`：先将历史站点恢复到 `docs/`，再生成报告、更新首页与 API，最后提交回 `gh-pages` 并在同一次运行中部署 Pages。发布共用并发组，避免多个任务从旧站点快照生成后覆盖彼此。`pages.yml` 可手动重新部署现有站点，不调用 LLM。

`site/legacy-ref` 记录迁移前最后一个含站点数据的源码提交。恢复脚本合并该提交中的历史 `docs/` 和现有 `gh-pages`，同名文件优先使用 `gh-pages` 版本，以保留已有发布结果及迁移前新增的报告。历史 HTML、JSON 和 API 索引都会保留；不要只迁移当天产物，否则会影响旧链接、跨日去重和汇总。此迁移停止向 `main` 追加站点文件，不重写已有 Git 历史。

本地也可显式发布已检查的完整站点（会推送到 `origin/gh-pages`）：

```bash
bash scripts/publish_gh_pages.sh publish docs 'Update generated site'
```

## API

站点提供静态 JSON API，便于 AI / 爬虫获取结构化数据。以下路径均相对于站点根地址：

| 端点 | 说明 |
|---|---|
| `/api/v1/manifest.json` | API 元信息、统计、端点列表 |
| `/api/v1/daily/index.json` | 所有日报日期列表 |
| `/api/v1/daily/latest.json` | 最新日报摘要（含 data_url 指向完整数据） |
| `/api/v1/monthly/index.json` | 所有月报列表 |
| `/api/v1/yearly/index.json` | 所有年报列表 |
| `/data/daily/YYYYMMDD.json` | 指定日报完整 JSON |

浏览器跨域访问取决于托管平台实际返回的响应头。`site/_headers` 为支持该文件格式的托管平台提供配置；GitHub Pages 不以该文件设置响应头，使用时应检查部署后的响应。

## 栏目健康检查

日报 JSON 会在 `section_meta` 中记录每个栏目的素材量、LLM 筛选量、fallback 状态和空栏原因。恢复历史数据后，可检查最近 N 期栏目覆盖情况：

```bash
python -m src.report_health --days 7
```

## 目录结构

```text
AInformer/
├── .github/workflows/
│   ├── ci.yml             Push / PR 测试
│   ├── daily.yml          日报调度
│   ├── monthly.yml        月报调度
│   ├── yearly.yml         年报调度
│   ├── publish.yml        共用生成、存档与部署流程
│   └── pages.yml          手动重新部署已有站点
├── scripts/
│   ├── publish_gh_pages.sh 恢复与发布命令入口
│   └── site_publish.py    历史恢复、归档校验与分支发布
├── src/
│   ├── main_daily.py      日报入口
│   ├── main_monthly.py    月报入口
│   ├── main_yearly.py     年报入口
│   ├── build_index.py     重建首页索引
│   ├── build_api.py       重建 JSON API 索引
│   ├── config.py          路径、环境变量、时区
│   ├── llm/client.py      DeepSeek 客户端与单次任务用量统计
│   ├── search/            DuckDuckGo、新闻源、RSS、GitHub Trending
│   ├── agents/            栏目筛选与并行执行
│   ├── dedupe.py          与最近日报对比去重
│   ├── synthesize.py      生成日报导语与重点标题
│   ├── summarize/         月报、年报聚合
│   └── render/            Jinja2 模板与 HTML / JSON 渲染
├── site/                  纳入版本管理的静态站点配置
│   ├── legacy-ref         迁移用历史数据提交
│   ├── .nojekyll
│   └── _headers
├── docs/                  本地生成目录（忽略，不提交）
│   ├── index.html
│   ├── daily/             YYYYMMDD.html
│   ├── monthly/           YYYYMM.html
│   ├── yearly/            YYYY.html
│   ├── api/v1/            JSON API 索引
│   └── data/              原始 JSON，供去重和汇总使用
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── LICENSE
└── README.md
```

## License

本仓库代码采用 [MIT License](LICENSE)。引用的第三方文章、图片和其他来源内容的权利归各自权利人所有，MIT 授权不改变这些内容的权利归属。
