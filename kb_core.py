"""
kb_core.py — 知识库核心引擎
录入 / 检索 / LLM 回答 / 管理
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from openai import OpenAI


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class ApiConfig:
    base_url: str = "https://api.siliconflow.cn/v1"
    api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "deepseek-ai/DeepSeek-V3"


@dataclass
class StorageConfig:
    db_path: str = "data/kb.db"


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    auth_token: str = ""


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    @classmethod
    def load(cls, path: str = "data/config.yaml") -> "Config":
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}
        api_raw = raw.get("api", {})
        storage_raw = raw.get("storage", {})
        server_raw = raw.get("server", {})
        return cls(
            api=ApiConfig(
                base_url=api_raw.get("base_url", "https://api.siliconflow.cn/v1"),
                api_key=api_raw.get("api_key", ""),
                embedding_model=api_raw.get("embedding_model", "BAAI/bge-m3"),
                llm_model=api_raw.get("llm_model", "deepseek-ai/DeepSeek-V3"),
            ),
            storage=StorageConfig(db_path=storage_raw.get("db_path", "data/kb.db")),
            server=ServerConfig(
                host=server_raw.get("host", "0.0.0.0"),
                port=server_raw.get("port", 8765),
                auth_token=server_raw.get("auth_token", ""),
            ),
        )


# ── Database ─────────────────────────────────────────────────────────────────

class KBStore:
    """SQLite-backed knowledge store with numpy vector similarity."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                content     TEXT    NOT NULL,
                title       TEXT    DEFAULT '',
                tags        TEXT    DEFAULT '',
                source      TEXT    DEFAULT 'cli',
                embedding   BLOB,
                created_at  REAL    NOT NULL,
                updated_at  REAL    NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_entries_tags ON entries(tags)")
        self._conn.commit()

    def add(self, content: str, title: str = "", tags: str = "",
            source: str = "cli", embedding: Optional[np.ndarray] = None) -> int:
        now = time.time()
        emb_bytes = embedding.tobytes() if embedding is not None else None
        cur = self._conn.execute(
            "INSERT INTO entries (content, title, tags, source, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (content, title, tags, source, emb_bytes, now, now),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_title_tags(self, entry_id: int, title: str, tags: str):
        now = time.time()
        self._conn.execute(
            "UPDATE entries SET title = ?, tags = ?, updated_at = ? WHERE id = ?",
            (title, tags, now, entry_id),
        )
        self._conn.commit()

    def update_content(self, entry_id: int, content: str, embedding: Optional[np.ndarray] = None):
        now = time.time()
        if embedding is not None:
            self._conn.execute(
                "UPDATE entries SET content = ?, embedding = ?, updated_at = ? WHERE id = ?",
                (content, embedding.tobytes(), now, entry_id),
            )
        else:
            self._conn.execute(
                "UPDATE entries SET content = ?, updated_at = ? WHERE id = ?",
                (content, now, entry_id),
            )
        self._conn.commit()

    def get(self, entry_id: int) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT id, content, title, tags, source, embedding, created_at, updated_at "
            "FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_all(self, tag: str = "", limit: int = 50, offset: int = 0) -> list[dict]:
        if tag:
            rows = self._conn.execute(
                "SELECT id, content, title, tags, source, embedding, created_at, updated_at "
                "FROM entries WHERE tags LIKE ? ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (f"%{tag}%", limit, offset),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, content, title, tags, source, embedding, created_at, updated_at "
                "FROM entries ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count(self, tag: str = "") -> int:
        if tag:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM entries WHERE tags LIKE ?", (f"%{tag}%",)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        return row[0] if row else 0

    def delete(self, entry_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def get_all_embeddings(self) -> list[tuple[int, np.ndarray]]:
        rows = self._conn.execute(
            "SELECT id, embedding FROM entries WHERE embedding IS NOT NULL"
        ).fetchall()
        result = []
        for rid, emb in rows:
            if emb:
                result.append((rid, np.frombuffer(emb, dtype=np.float32)))
        return result

    @staticmethod
    def _row_to_dict(row) -> Optional[dict]:
        if row is None:
            return None
        return {
            "id": row[0], "content": row[1], "title": row[2],
            "tags": row[3], "source": row[4],
            "created_at": row[6], "updated_at": row[7],
        }


# ── Vector Search ────────────────────────────────────────────────────────────

def cosine_similarity(vecs: np.ndarray, query: np.ndarray) -> np.ndarray:
    vecs_norm = np.linalg.norm(vecs, axis=1)
    query_norm = np.linalg.norm(query)
    denom = vecs_norm * query_norm
    denom[denom == 0] = 1e-10
    return np.dot(vecs, query) / denom


def search_similar(store: KBStore, query_vec: np.ndarray, k: int = 5) -> list[dict]:
    pairs = store.get_all_embeddings()
    if not pairs:
        return []
    ids = [p[0] for p in pairs]
    vecs = np.stack([p[1] for p in pairs])
    scores = cosine_similarity(vecs, query_vec)
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        entry = store.get(ids[idx])
        if entry:
            entry["score"] = float(scores[idx])
            results.append(entry)
    return results


# ── AI Client ─────────────────────────────────────────────────────────────────

class AIClient:
    """封装硅基流动 API：embedding + LLM 问答 + Markdown 格式化。"""

    def __init__(self, config: ApiConfig):
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url, api_key=config.api_key,
            timeout=60.0, max_retries=1,
        )

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(
            model=self.config.embedding_model, input=text,
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def extract_meta(self, content: str) -> tuple[str, str]:
        prompt = (
            "从以下文本提取标题（简短摘要）和标签（逗号分隔关键词）。只输出 JSON。\n"
            "格式：{\"title\": \"...\", \"tags\": \"...\"}\n\n"
            f"{content}\n\nJSON:"
        )
        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip optional ``` fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        try:
            data = json.loads(raw.strip())
            return data.get("title", ""), data.get("tags", "")
        except (json.JSONDecodeError, IndexError):
            return "", ""

    def format_content(self, raw_text: str) -> str:
        """用 LLM 将自由文本格式化为结构化 Markdown，自动 strip 外层代码块。"""
        prompt = (
            "将以下文本整理成结构化的 Markdown。规则：\n"
            "- 用 ## 标题概括主题\n"
            "- 命令/代码用 ``` 代码块，标注语言\n"
            "- 参数说明用列表或小标题\n"
            "- 保留所有信息，不删减\n"
            "- 直接输出 Markdown，禁止用 ```markdown 包裹\n\n"
            f"{raw_text}\n\nMarkdown:"
        )
        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=2000,
        )
        result = resp.choices[0].message.content.strip()
        # Strip outer ``` fences if present
        if result.startswith("```"):
            first_nl = result.find("\n")
            if first_nl != -1:
                result = result[first_nl + 1:]
            if result.rstrip().endswith("```"):
                result = result.rstrip()[:-3].strip()
        return result

    def ask(self, query: str, context_entries: list[dict]) -> str:
        if not context_entries:
            context_text = "（知识库中没有找到相关内容）"
        else:
            parts = []
            for e in context_entries:
                parts.append(
                    f"--- 条目 {e['id']} (标题: {e['title']}) ---\n"
                    f"内容: {e['content']}\n标签: {e['tags']}"
                )
            context_text = "\n\n".join(parts)
        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": (
                    "你是一个个人知识库助手。根据以下知识库条目内容回答用户问题。"
                    "如果知识库中有相关信息，请准确引用并给出详细说明。"
                    "如果知识库中没有相关信息，请如实告知，不要编造。"
                    "回答要简洁、准确、直接给出命令或步骤。"
                )},
                {"role": "user", "content": (
                    f"知识库内容：\n{context_text}\n\n"
                    f"用户问题：{query}\n\n请根据知识库内容回答："
                )},
            ],
            temperature=0.3, max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()


# ── High-Level API ────────────────────────────────────────────────────────────

class KnowledgeBase:
    """顶层 API，组合 Store 和 AI Client。"""

    def __init__(self, config: Config):
        self.config = config
        self.store = KBStore(config.storage.db_path)
        self.ai = AIClient(config.api) if config.api.api_key else None

    def add(self, content: str, title: str = "", tags: str = "",
            source: str = "cli", auto_meta: bool = True, format_md: bool = True) -> dict:
        """添加一条知识。auto_meta 提取元数据，format_md 格式化为 Markdown。"""
        if format_md and self.ai:
            try:
                content = self.ai.format_content(content)
            except Exception:
                pass

        if auto_meta and self.ai and (not title or not tags):
            try:
                auto_title, auto_tags = self.ai.extract_meta(content)
                title = title or auto_title
                tags = tags or auto_tags
            except Exception:
                pass

        embedding = None
        if self.ai:
            try:
                embedding = self.ai.embed(content)
            except Exception:
                pass

        entry_id = self.store.add(content, title, tags, source, embedding)
        return {"id": entry_id, "title": title, "tags": tags, "content": content}

    def reformat_entry(self, entry_id: int) -> Optional[dict]:
        """用 LLM 重新格式化已有条目为 Markdown。"""
        entry = self.store.get(entry_id)
        if not entry:
            return None
        if not self.ai:
            return entry
        new_content = self.ai.format_content(entry["content"])
        embedding = self.ai.embed(new_content)
        self.store.update_content(entry_id, new_content, embedding)
        return self.store.get(entry_id)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if not self.ai:
            return []
        query_vec = self.ai.embed(query)
        return search_similar(self.store, query_vec, k)

    def ask(self, query: str, k: int = 5) -> str:
        if not self.ai:
            return "错误：未配置 API key，无法使用问答功能。"
        entries = self.search(query, k)
        return self.ai.ask(query, entries)

    def list_entries(self, tag: str = "", limit: int = 50, offset: int = 0) -> list[dict]:
        return self.store.list_all(tag, limit, offset)

    def count(self, tag: str = "") -> int:
        return self.store.count(tag)

    def get_entry(self, entry_id: int) -> Optional[dict]:
        return self.store.get(entry_id)

    def delete_entry(self, entry_id: int) -> bool:
        return self.store.delete(entry_id)

    def set_meta(self, entry_id: int, title: str, tags: str):
        self.store.update_title_tags(entry_id, title, tags)


def load_kb(config_path: str = "data/config.yaml") -> KnowledgeBase:
    config = Config.load(config_path)
    return KnowledgeBase(config)
