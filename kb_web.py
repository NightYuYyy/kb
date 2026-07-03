"""
kb_web.py — FastAPI REST API + 内嵌 Web UI
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from kb_core import KnowledgeBase


# ── Models ───────────────────────────────────────────────────────────────────

class AddRequest(BaseModel):
    content: str
    title: str = ""
    tags: str = ""
    auto_meta: bool = True
    format_md: bool = True


class AddResponse(BaseModel):
    id: int
    title: str
    tags: str
    content: str = ""


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
        result = kb.add(
            content=req.content, title=req.title, tags=req.tags,
            source="web", auto_meta=req.auto_meta, format_md=req.format_md,
        )
        return AddResponse(id=result["id"], title=result["title"], tags=result["tags"], content=result.get("content", ""))

    @app.post("/api/ask", response_model=AskResponse, dependencies=[Depends(verify_auth)])
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

    @app.post("/api/search", dependencies=[Depends(verify_auth)])
    async def api_search(req: SearchRequest):
        results = kb.search(req.query, k=req.k)
        return JSONResponse([
            {"id": e["id"], "title": e["title"], "content": e["content"][:500],
             "tags": e["tags"], "score": e.get("score", 0)}
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
