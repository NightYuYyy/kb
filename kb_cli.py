"""
kb — 个人知识库 CLI
用法: kb add "内容" | kb ask "问题" | kb search "关键词" | kb list | kb serve
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from kb_core import load_kb

app = typer.Typer(
    name="kb",
    help="个人知识库 — 记录命令/工具/脚本用法，语义检索 + AI 问答",
    no_args_is_help=True,
)

# 全局配置路径，允许命令行覆盖
_config_path = "data/config.yaml"


def _get_kb():
    return load_kb(_config_path)


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
    if result["title"]:
        typer.echo(f"  标题: {result['title']}")
    if result["tags"]:
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
        title = entry["title"] or f"条目 {entry['id']}"
        score = entry.get("score", 0)
        typer.echo(f"\n{'─' * 60}")
        typer.echo(f"[{i}] {title}  (相似度: {score:.2f})")
        typer.echo(f"    ID: {entry['id']}  标签: {entry['tags']}")
        typer.echo(f"    {entry['content'][:200]}{'...' if len(entry['content']) > 200 else ''}")


@app.command()
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
        title = e["title"] or f"条目 {e['id']}"
        typer.echo(f"  [{e['id']}] {title}")
        typer.echo(f"      标签: {e['tags']}")


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
    typer.echo(f"标题: {entry['title']}")
    typer.echo(f"标签: {entry['tags']}")
    typer.echo(f"来源: {entry['source']}")
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
        typer.echo(f"✓ 已格式化条目 {entry_id} — {entry['title']}")
    except Exception as e:
        typer.echo(f"✗ 格式化失败: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("", "--host", "-h", help="监听地址"),
    port: int = typer.Option(0, "--port", "-p", help="监听端口"),
):
    """启动 Web UI 和 REST API 服务。"""
    from kb_web import create_app

    kb = _get_kb()
    app_web = create_app(kb)
    import uvicorn

    cfg = kb.config.server
    h = host or cfg.host
    p = port or cfg.port
    typer.echo(f"🚀 知识库 Web 服务启动: http://{h}:{p}")
    uvicorn.run(app_web, host=h, port=p, log_level="info")


@app.command()
def stats():
    """显示知识库统计信息。"""
    kb = _get_kb()
    total = kb.count()
    typer.echo(f"总条目: {total}")
    if total > 0:
        entries = kb.list_entries(limit=1)
        if entries:
            import time
            latest = entries[0]
            typer.echo(f"最近更新: {time.strftime('%Y-%m-%d %H:%M', time.localtime(latest['updated_at']))}")


@app.callback()
def main(
    config: str = typer.Option("data/config.yaml", "--config", "-c", help="配置文件路径"),
):
    global _config_path
    _config_path = config


if __name__ == "__main__":
    app()
