"""
kb_web.py — FastAPI REST API + 内嵌 Web UI
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from kb_core import DuplicateEntryError, KnowledgeBase


# ── Models ───────────────────────────────────────────────────────────────────

class AddRequest(BaseModel):
    content: str
    title: str = ""
    tags: str = ""
    auto_meta: bool = True
    format_md: bool = True
    force: bool = False


class AddResponse(BaseModel):
    id: int
    title: str
    tags: str
    content: str = ""


class UpdateRequest(BaseModel):
    content: str = ""
    title: str = ""
    tags: str = ""
    format_md: bool = False
    auto_meta: bool = False


class AskRequest(BaseModel):
    query: str
    k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class SearchRequest(BaseModel):
    query: str
    k: int = 5


# ── Agent Prompt（单一来源：CLI 文档和 Web UI 复制按钮都从这里取）──────────────

AGENT_PROMPT_TEMPLATE = """你收到了个人知识库（KB）的接入配置。依次完成两步，然后向用户确认结果。

## 第一步：安装 skill（只做一次）
把本消息末尾「=== KB 技能定义 ===」之后的全部内容保存为名为 kb 的 skill，重开会话后仍然生效：
- Claude Code：`~/.claude/skills/kb/SKILL.md`
- Codex：`~/.codex/skills/kb/SKILL.md`
- 其他 agent：写入你的 skills 目录下 `kb/SKILL.md`
目录不存在则创建；已存在同名 skill 则整体覆盖为本版本。无文件系统权限（网页对话 AI）时跳过此步，在本会话内直接遵守该定义。

## 第二步：验证连接
```bash
curl -s "{KB_URL}/api/list?limit=1" -H "X-API-Key: {TOKEN}"
```
返回 JSON（含 total 字段）即接入成功，向用户报告 skill 安装位置和知识库条目总数；失败则报告具体错误。

=== KB 技能定义（从下一行起完整保存）===
---
name: kb
description: 个人知识库（远程 API）。查询已记录的任何信息时使用——技术类如命令、脚本、服务器、配置，也包括事实、决定、偏好、笔记等非技术类信息；仅在用户明确要求时记录。用户说"查知识库/记到知识库/kb 查"或遇到不确定的信息时触发。
---

# kb — 个人知识库

- 服务地址：{KB_URL}
- 认证：所有请求带请求头 `X-API-Key: {TOKEN}`

## 查询（自动）
遇到可能之前记录过的问题，先查知识库再回答，不凭猜测——技术类如命令用法、脚本、服务器/部署信息、工具配置，也包括事实、决定、偏好、笔记等非技术类信息：
```bash
curl -s -X POST {KB_URL}/api/ask \\
  -H "Content-Type: application/json" -H "X-API-Key: {TOKEN}" \\
  -d '{"query": "问题描述"}'
```
返回 {"answer": "...", "sources": [...]}，直接引用 answer。只需要相关条目列表、不需要 AI 总结时，改用 /api/search（更快，返回条目含 id 和 score）。

## 记录（仅在用户明确要求时）
只有用户明确说"记录/记到知识库/存一下"这类指令时才写入；其他任何时候都不要主动记录，最多提醒一句"要记入知识库吗"。
写入前必须先用 /api/search 查重，按结果二选一：

1. 没有同主题条目 → 新增：
```bash
curl -s -X POST {KB_URL}/api/add \\
  -H "Content-Type: application/json" -H "X-API-Key: {TOKEN}" \\
  -d '{"content": "要记录的内容（Markdown）", "format_md": true, "auto_meta": true}'
```
服务端也会自动查重：若返回 HTTP 409，说明已存在相似条目，响应体 detail.similar 列出候选条目（含 id、title、score、content 片段）。此时改走第 2 步，GET 拿到最相似条目后 PUT /api/entry/<id> 合并，不要重复新增；只有确认是完全不同的主题时，才在请求体加 "force": true 重新提交以强制新增。

2. 已有同主题条目 → 更新而不是新增。先 GET {KB_URL}/api/entry/<id> 拿到原文，把新信息合并进去（保留仍然有效的旧内容，改写过时的部分），整体替换：
```bash
curl -s -X PUT {KB_URL}/api/entry/<id> \\
  -H "Content-Type: application/json" -H "X-API-Key: {TOKEN}" \\
  -d '{"content": "合并后的完整 Markdown"}'
```
更新时可加 "auto_meta": true 重新生成标题标签（主题变化大时用）。

## 不要记录
- 密码、API key、私钥等明文凭据——只记引用位置（如"密码在密码管理器 xx 条目"）
- 一次性临时信息、项目代码库里已有的内容

## 内容规范
- Markdown 格式，代码块标注语言；换行用真实换行，不要写字面的 \\n
- 一次记全：用途、命令、参数说明、路径、示例
- 条目要自足：脱离当前对话也能独立读懂。不写「与 X 无关」这类只在当前对话里才有意义的澄清；不写可由条目内容自然推出的信息（如路径已表明所在机器，就不必再写平台限制）；实现细节不写进使用说明
- Windows/PowerShell 下用 curl 时注意 JSON 引号转义，推荐把 JSON 写入临时文件后 -d @文件名

## 其他操作
- 列出条目: curl -s "{KB_URL}/api/list?limit=50" -H "X-API-Key: {TOKEN}"
- 查看单条: GET {KB_URL}/api/entry/<id>
- 删除单条: DELETE {KB_URL}/api/entry/<id>（仅在用户明确要求删除时）
- 本机装有 kb CLI 时，可 `kb connect {KB_URL} --token {TOKEN}` 后改用 kb ask/add/update 命令"""


def build_agent_prompt(kb_url: str, token: str) -> str:
    return (AGENT_PROMPT_TEMPLATE
            .replace("{KB_URL}", kb_url or "{KB_URL}")
            .replace("{TOKEN}", token or "{TOKEN}"))


# ── App Factory ──────────────────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(kb: KnowledgeBase) -> FastAPI:
    app = FastAPI(title="KB - 个人知识库", version="1.0.0")
    auth_token = kb.config.server.auth_token

    # ── Auth dependency ──
    async def verify_auth(x_api_key: str = Header(None, alias="X-API-Key")):
        if auth_token and x_api_key != auth_token:
            raise HTTPException(status_code=401, detail="无效的 API Key")

    # ── Web UI ───────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = TEMPLATE_DIR / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return HTMLResponse("<h1>KB 知识库</h1><p>Web UI 模板未找到</p>")

    # ── REST API ─────────────────────────────────────────────────────────────

    @app.post("/api/add", response_model=AddResponse, dependencies=[Depends(verify_auth)])
    async def api_add(req: AddRequest):
        try:
            result = kb.add(
                content=req.content, title=req.title, tags=req.tags,
                source="web", auto_meta=req.auto_meta, format_md=req.format_md,
                force=req.force,
            )
        except DuplicateEntryError as e:
            raise HTTPException(status_code=409, detail={
                "error": "duplicate",
                "message": "存在相似条目：请先 GET 对应条目，合并后用 PUT /api/entry/<id> 更新；确属不同主题时才用 force=true 强制新增",
                "similar": [
                    {
                        "id": s["id"],
                        "title": s.get("title", ""),
                        "score": round(float(s.get("score", 0)), 2),
                        "content": s.get("content", "")[:300],
                    }
                    for s in e.similar
                ],
            })
        return AddResponse(id=result["id"], title=result["title"], tags=result["tags"], content=result.get("content", ""))

    @app.get("/api/prompt", dependencies=[Depends(verify_auth)])
    async def api_prompt(base: str = Query("", description="知识库对外地址，如 https://kb.example.com")):
        # 配置了 public_url 时优先使用，避免从 localhost 打开 UI 复制出只能本机用的提示词
        kb_url = kb.config.server.public_url or base
        return JSONResponse({"prompt": build_agent_prompt(kb_url, auth_token)})

    @app.post("/api/ask", response_model=AskResponse, dependencies=[Depends(verify_auth)])
    async def api_ask(req: AskRequest):
        if not kb.ai:
            raise HTTPException(status_code=503, detail="服务端未配置 api.api_key，问答功能不可用")
        entries = [
            e for e in kb.search(req.query, k=req.k)
            if e.get("vec_score", 0.0) >= kb.config.search.min_score or e.get("fts_hit", False)
        ]
        sources = [
            {"id": e["id"], "title": e["title"], "content": e["content"][:500], "score": e.get("score", 0)}
            for e in entries
        ]
        try:
            answer = kb.ai.ask(req.query, entries)
        except Exception as e:
            answer = f"回答生成失败: {e}"
        return AskResponse(answer=answer, sources=sources)

    @app.post("/api/search", dependencies=[Depends(verify_auth)])
    async def api_search(req: SearchRequest):
        results = kb.search(req.query, k=req.k)
        return JSONResponse([
            {"id": e["id"], "title": e["title"], "content": e["content"][:500],
             "tags": e["tags"], "score": e.get("score", 0),
             "vec_score": round(float(e.get("vec_score", 0.0)), 4),
             "fts_hit": bool(e.get("fts_hit", False))}
            for e in results
        ])

    @app.get("/api/list", dependencies=[Depends(verify_auth)])
    async def api_list(tag: str = Query(""), limit: int = Query(50), offset: int = Query(0)):
        entries = kb.list_entries(tag=tag, limit=limit, offset=offset)
        total = kb.count(tag=tag)
        return JSONResponse({"total": total, "entries": entries})

    @app.get("/api/entry/{entry_id}", dependencies=[Depends(verify_auth)])
    async def api_get(entry_id: int):
        entry = kb.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="条目不存在")
        return JSONResponse(entry)

    @app.put("/api/entry/{entry_id}", dependencies=[Depends(verify_auth)])
    async def api_update(entry_id: int, req: UpdateRequest):
        if not (req.content or req.title or req.tags):
            raise HTTPException(status_code=422, detail="content / title / tags 至少提供一项")
        try:
            entry = kb.update_entry(
                entry_id, content=req.content or None, title=req.title or None,
                tags=req.tags or None, format_md=req.format_md, auto_meta=req.auto_meta,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not entry:
            raise HTTPException(status_code=404, detail="条目不存在")
        return JSONResponse(entry)

    @app.delete("/api/entry/{entry_id}", dependencies=[Depends(verify_auth)])
    async def api_delete(entry_id: int):
        ok = kb.delete_entry(entry_id)
        if not ok:
            raise HTTPException(status_code=404, detail="条目不存在")
        return JSONResponse({"deleted": entry_id})

    @app.post("/api/entry/{entry_id}/reformat", dependencies=[Depends(verify_auth)])
    async def api_reformat(entry_id: int):
        try:
            entry = kb.reformat_entry(entry_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        if not entry:
            raise HTTPException(status_code=404, detail="条目不存在")
        return JSONResponse(entry)

    return app


# ── Standalone entry ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from kb_core import load_kb

    kb = load_kb()
    cfg = kb.config.server
    app = create_app(kb)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
