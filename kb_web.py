"""
kb_web.py — FastAPI REST API + 内嵌 Web UI
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kb_core import KnowledgeBase


# ── Request/Response Models ──────────────────────────────────────────────────

class AddRequest(BaseModel):
    content: str
    title: str = ""
    tags: str = ""
    auto_meta: bool = True


class AddResponse(BaseModel):
    id: int
    title: str
    tags: str


class AskRequest(BaseModel):
    query: str
    k: int = 5


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class SearchRequest(BaseModel):
    query: str
    k: int = 5


# ── App Factory ──────────────────────────────────────────────────────────────

TEMPLATE_DIR = Path(__file__).parent / "templates"


def create_app(kb: KnowledgeBase) -> FastAPI:
    app = FastAPI(title="KB - 个人知识库", version="1.0.0")

    # ── Web UI ───────────────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = TEMPLATE_DIR / "index.html"
        if html_path.exists():
            return html_path.read_text(encoding="utf-8")
        return "<h1>KB 知识库</h1><p>Web UI 模板未找到</p>"

    # ── REST API ─────────────────────────────────────────────────────────────

    @app.post("/api/add", response_model=AddResponse)
    async def api_add(req: AddRequest):
        result = kb.add(
            content=req.content,
            title=req.title,
            tags=req.tags,
            source="web",
            auto_meta=req.auto_meta,
        )
        return AddResponse(id=result["id"], title=result["title"], tags=result["tags"])

    @app.post("/api/ask", response_model=AskResponse)
    async def api_ask(req: AskRequest):
        entries = kb.search(req.query, k=req.k)
        sources = [
            {"id": e["id"], "title": e["title"], "content": e["content"][:500], "score": e.get("score", 0)}
            for e in entries
        ]
        try:
            answer = kb.ai.ask(req.query, entries)
        except Exception as e:
            answer = f"回答生成失败: {e}"
        return AskResponse(answer=answer, sources=sources)

    @app.post("/api/search")
    async def api_search(req: SearchRequest):
        results = kb.search(req.query, k=req.k)
        return JSONResponse([
            {"id": e["id"], "title": e["title"], "content": e["content"][:500],
             "tags": e["tags"], "score": e.get("score", 0)}
            for e in results
        ])

    @app.get("/api/list")
    async def api_list(
        tag: str = Query(""),
        limit: int = Query(50),
        offset: int = Query(0),
    ):
        entries = kb.list_entries(tag=tag, limit=limit, offset=offset)
        total = kb.count(tag=tag)
        return JSONResponse({"total": total, "entries": entries})

    @app.get("/api/entry/{entry_id}")
    async def api_get(entry_id: int):
        entry = kb.get_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="条目不存在")
        return JSONResponse(entry)

    @app.delete("/api/entry/{entry_id}")
    async def api_delete(entry_id: int):
        ok = kb.delete_entry(entry_id)
        if not ok:
            raise HTTPException(status_code=404, detail="条目不存在")
        return JSONResponse({"deleted": entry_id})

    return app


# ── Standalone entry ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from kb_core import load_kb

    kb = load_kb()
    cfg = kb.config.server
    app = create_app(kb)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")
