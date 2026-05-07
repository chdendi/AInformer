# AInformer

**🌐 站点**：https://chenhao-dendi.github.io/AInformer

## 目录结构

```
AInformer/
├── .github/workflows/
│   ├── daily.yml          每日 09:00 (UTC+8) 跑，写日报
│   ├── monthly.yml        每月 1 号 10:00 跑，写上月月报
│   ├── yearly.yml         每年 1.2 号 11:00 跑，写上年年报
│   └── pages.yml          docs/ 变化时自动部署 GitHub Pages
├── src/
│   ├── main_daily.py      日报入口 (python -m src.main_daily)
│   ├── main_monthly.py    月报入口
│   ├── main_yearly.py     年报入口
│   ├── build_index.py     仅重建首页索引
│   ├── config.py          路径 / 环境变量 / 时区
│   ├── llm/client.py      DeepSeek (OpenAI 兼容) 客户端
│   ├── search/
│   │   ├── rss.py         OpenAI/Anthropic/TheVerge/36kr/ArXiv 等 RSS
│   │   └── tavily.py      Tavily Web Search
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
├── docs/                   ←  GitHub Pages 站点根目录
│   ├── .nojekyll
│   ├── index.html
│   ├── daily/   YYYYMMDD.html
│   ├── monthly/ YYYYMM.html
│   ├── yearly/  YYYY.html
│   └── data/               JSON 原始数据，用于去重和聚合
│       ├── daily/   YYYYMMDD.json
│       ├── monthly/ YYYYMM.json
│       └── yearly/  YYYY.json
├── requirements.txt
├── .env.example
└── README.md
```
