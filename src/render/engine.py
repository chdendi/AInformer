from __future__ import annotations

import re
from html import escape
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).parent / "templates"


_MD_CODE_BLOCK = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_MD_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_MD_LIST = re.compile(r"^(\s*)[-*]\s+(.*)$", re.MULTILINE)
_MD_NUM = re.compile(r"^(\s*)\d+\.\s+(.*)$", re.MULTILINE)


def md_to_html(text: str) -> str:
    if not text:
        return ""
    parts: list[str] = []
    last = 0
    for m in _MD_CODE_BLOCK.finditer(text):
        parts.append(_render_paragraphs(text[last:m.start()]))
        lang = m.group(1) or ""
        code = escape(m.group(2))
        parts.append(f'<pre class="code"><code class="lang-{escape(lang)}">{code}</code></pre>')
        last = m.end()
    parts.append(_render_paragraphs(text[last:]))
    return "".join(parts)


def _render_paragraphs(chunk: str) -> str:
    if not chunk.strip():
        return ""
    chunk = escape(chunk)
    chunk = _MD_BOLD.sub(r"<strong>\1</strong>", chunk)
    chunk = _MD_INLINE_CODE.sub(r"<code>\1</code>", chunk)

    lines = chunk.split("\n")
    out: list[str] = []
    in_list = False
    for line in lines:
        ml = _MD_LIST.match(line)
        mn = _MD_NUM.match(line)
        if ml or mn:
            if not in_list:
                out.append("<ul>")
                in_list = True
            content = (ml or mn).group(2)
            out.append(f"<li>{content}</li>")
        else:
            if in_list:
                out.append("</ul>")
                in_list = False
            stripped = line.strip()
            if stripped:
                out.append(f"<p>{stripped}</p>")
    if in_list:
        out.append("</ul>")
    return "".join(out)


def env() -> Environment:
    e = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    e.filters["md"] = md_to_html
    return e


def render(template_name: str, **ctx: Any) -> str:
    return env().get_template(template_name).render(**ctx)
