"""
kb_core.py — 知识库核心引擎
录入 / 检索 / LLM 回答 / 管理
"""

from __future__ import annotations

import os
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
            storage=StorageConfig(
                db_path=storage_raw.get("db_path", "data/kb.db"),
            ),
            server=ServerConfig(
                host=server_raw.get("host", "0.0.0.0"),
                port=server_raw.get("port", 8765),
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
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entries_tags ON entries(tags)
        """)
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
        """返回 [(id, embedding_array), ...]，仅包含有 embedding 的条目。"""
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
            "id": row[0],
            "content": row[1],
            "title": row[2],
            "tags": row[3],
            "source": row[4],
            "created_at": row[6],
            "updated_at": row[7],
        }


# ── Vector Search ────────────────────────────────────────────────────────────

def cosine_similarity(vecs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """批量计算余弦相似度。vecs: (N, D), query: (D,) → (N,)"""
    vecs_norm = np.linalg.norm(vecs, axis=1)
    query_norm = np.linalg.norm(query)
    # 避免除零
    denom = vecs_norm * query_norm
    denom[denom == 0] = 1e-10
    return np.dot(vecs, query) / denom


def search_similar(store: KBStore, query_vec: np.ndarray, k: int = 5) -> list[dict]:
    """在 store 中搜索与 query_vec 最相似的 k 条记录。"""
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
    """封装硅基流动 API：embedding + LLM 问答。"""

    def __init__(self, config: ApiConfig):
        self.config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key, timeout=30.0, max_retries=1)

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(
            model=self.config.embedding_model,
            input=text,
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        return vec

    def extract_meta(self, content: str) -> tuple[str, str]:
        """用 LLM 从内容中提取 title 和 tags。"""
        prompt = (
            "你是一个知识库助手。从以下用户输入的文本中提取标题（简短摘要）和标签（逗号分隔的英文/中文关键词）。\n"
            "只输出 JSON 格式：{\"title\": \"...\", \"tags\": \"...\"}\n\n"
            f"用户输入：\n{content}\n\n"
            "直接输出 JSON："
        )
        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content.strip()
        # 容错：尝试从响应中提取 JSON
        import json
        try:
            # 可能带 markdown 代码块标记
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                if raw.endswith("```"):
                    raw = raw[:-3]
            data = json.loads(raw.strip())
            return data.get("title", ""), data.get("tags", "")
        except (json.JSONDecodeError, IndexError):
            return "", ""

    def ask(self, query: str, context_entries: list[dict]) -> str:
        """基于检索到的知识库条目，用 LLM 回答问题。"""
        if not context_entries:
            context_text = "（知识库中没有找到相关内容）"
        else:
            parts = []
            for e in context_entries:
                parts.append(
                    f"--- 条目 {e['id']} (标题: {e['title']}) ---\n"
                    f"内容: {e['content']}\n"
                    f"标签: {e['tags']}"
                )
            context_text = "\n\n".join(parts)

        system_prompt = (
            "你是一个个人知识库助手。根据以下知识库条目内容回答用户问题。\n"
            "如果知识库中有相关信息，请准确引用并给出详细说明。\n"
            "如果知识库中没有相关信息，请如实告知，不要编造。\n"
            "回答要简洁、准确、直接给出命令或步骤。"
        )
        user_prompt = (
            f"知识库内容：\n{context_text}\n\n"
            f"用户问题：{query}\n\n"
            "请根据知识库内容回答："
        )

        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
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
            source: str = "cli", auto_meta: bool = True) -> dict:
        """添加一条知识。auto_meta=True 时自动用 LLM 提取标题和标签。"""
        # 自动提取元数据
        if auto_meta and self.ai and (not title or not tags):
            try:
                auto_title, auto_tags = self.ai.extract_meta(content)
                title = title or auto_title
                tags = tags or auto_tags
            except Exception:
                pass  # LLM 提取失败不阻塞录入

        # 生成 embedding
        embedding = None
        if self.ai:
            try:
                embedding = self.ai.embed(content)
            except Exception:
                pass

        entry_id = self.store.add(content, title, tags, source, embedding)
        return {"id": entry_id, "title": title, "tags": tags}

    def search(self, query: str, k: int = 5) -> list[dict]:
        """语义检索。"""
        if not self.ai:
            return []
        try:
            query_vec = self.ai.embed(query)
            return search_similar(self.store, query_vec, k)
        except Exception as e:
            raise RuntimeError(f"检索失败: {e}")

    def ask(self, query: str, k: int = 5) -> str:
        """检索 + LLM 回答。"""
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
        # 重新生成 embedding（因为内容可能未变，但标题变了不改变 embedding 也合理）
        # 这里只更新元数据，不重新 embed


def load_kb(config_path: str = "data/config.yaml") -> KnowledgeBase:
    """从配置文件加载 KnowledgeBase 实例。"""
    config = Config.load(config_path)
    return KnowledgeBase(config)
