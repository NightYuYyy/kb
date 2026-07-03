"""
kb — 个人知识库 CLI
用法: kb add "内容" | kb ask "问题" | kb search "关键词" | kb list | kb serve
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
import httpx

from kb_core import load_kb

app = typer.Typer(
    name="kb",
    help="个人知识库 — 记录命令/工具/脚本用法，语义检索 + AI 问答",
    no_args_is_help=True,
)

_config_path = "data/config.yaml"
_remote_url: str = ""
_remote_token: str = ""


class RemoteKB:
    """通过 HTTP API 访问远程知识库。"""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"X-API-Key": token, "Content-Type": "application/json"}
        self._client = httpx.Client(timeout=60)

    def _post(self, path: str, data: dict = None) -> dict:
        r = self._client.post(f"{self.base_url}{path}", json=data or {}, headers=self.headers)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self._client.get(f"{self.base_url}{path}", headers=self.headers)
        r.raise_for_status()
        return r.json()

    def ask(self, question: str, k: int = 5) -> str:
        return self._post("/api/ask", {"query": question, "k": k})["answer"]

    def add(self, content: str, title: str = "", tags: str = "",
            source: str = "cli", auto_meta: bool = True, format_md: bool = True) -> dict:
        return self._post("/api/add", {"content": content, "title": title, "tags": tags,
                                        "format_md": format_md, "auto_meta": auto_meta})

    def search(self, query: str, k: int = 5) -> list[dict]:
        return self._post("/api/search", {"query": query, "k": k})

    def get_entry(self, entry_id: int) -> dict:
        return self._get(f"/api/entry/{entry_id}")

    def list_entries(self, tag: str = "", limit: int = 50, offset: int = 0) -> list[dict]:
        return self._get(f"/api/list?tag={tag}&limit={limit}&offset={offset}")["entries"]

    def count(self, tag: str = "") -> int:
        return self._get(f"/api/list?tag={tag}&limit=1")["total"]

    def delete_entry(self, entry_id: int) -> bool:
        self._client.delete(f"{self.base_url}/api/entry/{entry_id}", headers=self.headers)
        return True

    def reformat_entry(self, entry_id: int) -> dict:
        return self._post(f"/api/entry/{entry_id}/reformat")


def _get_kb():
    if _remote_url:
        return RemoteKB(_remote_url, _remote_token)
    config_path = _config_path
    if not Path(config_path).exists() and not Path(config_path).is_absolute():
        alt = Path(__file__).parent / config_path
        if alt.exists():
            config_path = str(alt)
    return load_kb(config_path)


@app.command()
def add(
    content: str = typer.Argument(..., help="要记录的知识内容（自由文本）"),
    title: str = typer.Option("", "--title", "-t", help="手动指定标题"),
    tags: str = typer.Option("", "--tags", "-g", help="手动指定标签（逗号分隔）"),
    no_auto: bool = typer.Option(False, "--no-auto", help="禁用 LLM 自动提取标题/标签"),
):
    """录入一条知识。最少只需提供内容文本，系统自动提取标题和标签。"""
    kb = _get_kb()
    result = kb.add(content, title, tags, auto_meta=not no_auto)
    typer.echo(f"✓ 已录入 (id={result['id']})")
    if result.get("title"):
        typer.echo(f"  标题: {result['title']}")
    if result.get("tags"):
        typer.echo(f"  标签: {result['tags']}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="要查询的问题"),
    k: int = typer.Option(5, "--k", "-k", help="检索条目数量"),
):
    """语义检索 + AI 回答问题。"""
    kb = _get_kb()
    typer.echo("⏳ 检索中...", err=True)
    try:
        answer = kb.ask(question, k=k)
        typer.echo(answer)
    except Exception as e:
        typer.echo(f"✗ 错误: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def search(
    query: str = typer.Argument(..., help="搜索关键词或自然语言描述"),
    k: int = typer.Option(5, "--k", "-k", help="返回条目数量"),
):
    """语义检索，列出相关知识库条目。"""
    kb = _get_kb()
    typer.echo("⏳ 检索中...", err=True)
    try:
        results = kb.search(query, k=k)
    except Exception as e:
        typer.echo(f"✗ 错误: {e}", err=True)
        raise typer.Exit(code=1)
    if not results:
        typer.echo("（未找到相关条目）")
        return
    for i, entry in enumerate(results, 1):
        title = entry.get("title") or f"条目 {entry['id']}"
        score = entry.get("score", 0)
        typer.echo(f"\n{'─' * 60}")
        typer.echo(f"[{i}] {title}  (相似度: {score:.2f})")
        typer.echo(f"    ID: {entry['id']}  标签: {entry.get('tags', '')}")
        content = entry.get("content", "")
        typer.echo(f"    {content[:200]}{'...' if len(content) > 200 else ''}")


@app.command("list")
def list_entries(
    tag: str = typer.Option("", "--tag", "-t", help="按标签筛选"),
    limit: int = typer.Option(50, "--limit", "-n", help="最大条目数"),
):
    """列出知识库中的所有条目。"""
    kb = _get_kb()
    entries = kb.list_entries(tag=tag, limit=limit)
    total = kb.count(tag=tag)
    typer.echo(f"共 {total} 条" + (f"（筛选: {tag}）" if tag else ""))
    for e in entries:
        title = e.get("title") or f"条目 {e['id']}"
        typer.echo(f"  [{e['id']}] {title}")
        typer.echo(f"      标签: {e.get('tags', '')}")


@app.command()
def show(
    entry_id: int = typer.Argument(..., help="条目 ID"),
):
    """查看某条知识的完整内容。"""
    kb = _get_kb()
    entry = kb.get_entry(entry_id)
    if not entry:
        typer.echo(f"✗ 条目 {entry_id} 不存在", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"ID: {entry['id']}")
    typer.echo(f"标题: {entry.get('title', '')}")
    typer.echo(f"标签: {entry.get('tags', '')}")
    typer.echo(f"来源: {entry.get('source', '')}")
    typer.echo(f"内容:\n{entry['content']}")


@app.command()
def delete(
    entry_id: int = typer.Argument(..., help="要删除的条目 ID"),
):
    """删除一条知识。"""
    kb = _get_kb()
    ok = kb.delete_entry(entry_id)
    if ok:
        typer.echo(f"✓ 已删除条目 {entry_id}")
    else:
        typer.echo(f"✗ 条目 {entry_id} 不存在", err=True)
        raise typer.Exit(code=1)


@app.command()
def reformat(
    entry_id: int = typer.Argument(..., help="要重新格式化的条目 ID"),
):
    """用 AI 将已有条目重新格式化为 Markdown。"""
    kb = _get_kb()
    typer.echo("⏳ AI 格式化中...", err=True)
    try:
        entry = kb.reformat_entry(entry_id)
        if not entry:
            typer.echo(f"✗ 条目 {entry_id} 不存在", err=True)
            raise typer.Exit(code=1)
        typer.echo(f"✓ 已格式化条目 {entry_id} — {entry.get('title', '')}")
    except Exception as e:
        typer.echo(f"✗ 格式化失败: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("", "--host", "-h", help="监听地址"),
    port: int = typer.Option(0, "--port", "-p", help="监听端口"),
):
    """启动 Web UI 和 REST API 服务。"""
    if _remote_url:
        typer.echo("✗ serve 只能在本地使用，不能连接远程服务", err=True)
        raise typer.Exit(code=1)

    try:
        from kb_web import create_app
        kb = _get_kb()
        app_web = create_app(kb)
        import uvicorn
        cfg = kb.config.server
        h = host or cfg.host
        p = port or cfg.port
        typer.echo(f"🚀 知识库 Web 服务启动: http://{h}:{p}")
        uvicorn.run(app_web, host=h, port=p, log_level="info")
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"✗ 启动失败: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def stats():
    """显示知识库统计信息。"""
    kb = _get_kb()
    total = kb.count()
    typer.echo(f"总条目: {total}")
    if total > 0:
        entries = kb.list_entries(limit=1)
        if entries:
            latest = entries[0]
            ts = latest.get("updated_at", 0)
            if ts:
                typer.echo(f"最近更新: {time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))}")


@app.callback()
def main(
    config: str = typer.Option("data/config.yaml", "--config", "-c", help="配置文件路径"),
    remote: str = typer.Option("", "--remote", "-r", help="远程 KB 服务地址"),
    token: str = typer.Option("", "--token", "-k", help="远程 API token"),
):
    global _config_path, _remote_url, _remote_token
    _config_path = config
    _remote_url = remote or os.environ.get("KB_REMOTE_URL", "")
    _remote_token = token or os.environ.get("KB_TOKEN", "")


if __name__ == "__main__":
    app()
