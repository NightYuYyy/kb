"""
kb_core.py — 知识库核心引擎
录入 / 检索 / LLM 回答 / 管理
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from openai import OpenAI

logger = logging.getLogger("kb")


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
class SearchConfig:
    min_score: float = 0.35        # ask 检索纳入上下文的最低相似度
    dedup_threshold: float = 0.85  # add 查重阈值,相似度 ≥ 此值拒绝写入


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    auth_token: str = ""
    public_url: str = ""  # 对外访问地址（如 https://kb.example.com），用于生成 Agent 提示词


@dataclass
class Config:
    api: ApiConfig = field(default_factory=ApiConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

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
        search_raw = raw.get("search", {})
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
                # KB_AUTH_TOKEN 环境变量优先于配置文件（docker-compose 部署依赖此覆盖）
                auth_token=os.environ.get("KB_AUTH_TOKEN", "") or server_raw.get("auth_token", ""),
                public_url=str(server_raw.get("public_url", "") or "").rstrip("/"),
            ),
            search=SearchConfig(
                min_score=float(search_raw.get("min_score", 0.35)),
                dedup_threshold=float(search_raw.get("dedup_threshold", 0.85)),
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
        self._init_fts()

    def _init_fts(self):
        """建立 FTS5 全文索引（trigram 分词，外部内容表）及三个同步触发器。

        - 建表前先查 sqlite_master 判断是否已存在；仅首次创建时 rebuild，
          让升级前已存在的旧行进入索引。
        - trigram 分词支持子串匹配（工具名、错误码等精确 token），但要求 ≥3 字符。
        - 环境不支持 FTS5/trigram 时降级：fts_enabled=False，检索退回纯向量路径。
        """
        try:
            exists = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='entries_fts'"
            ).fetchone()
            if exists is None:
                self._conn.execute("""
                    CREATE VIRTUAL TABLE entries_fts USING fts5(
                        content, title, tags,
                        content='entries', content_rowid='id', tokenize='trigram'
                    )
                """)
                self._conn.execute("""
                    CREATE TRIGGER entries_ai AFTER INSERT ON entries BEGIN
                        INSERT INTO entries_fts(rowid, content, title, tags)
                        VALUES (new.id, new.content, new.title, new.tags);
                    END
                """)
                self._conn.execute("""
                    CREATE TRIGGER entries_ad AFTER DELETE ON entries BEGIN
                        INSERT INTO entries_fts(entries_fts, rowid, content, title, tags)
                        VALUES ('delete', old.id, old.content, old.title, old.tags);
                    END
                """)
                self._conn.execute("""
                    CREATE TRIGGER entries_au AFTER UPDATE ON entries BEGIN
                        INSERT INTO entries_fts(entries_fts, rowid, content, title, tags)
                        VALUES ('delete', old.id, old.content, old.title, old.tags);
                        INSERT INTO entries_fts(rowid, content, title, tags)
                        VALUES (new.id, new.content, new.title, new.tags);
                    END
                """)
                # 首次创建：把已存在的旧行灌入索引
                self._conn.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
                self._conn.commit()
            self.fts_enabled = True
        except sqlite3.OperationalError as e:
            self.fts_enabled = False
            logger.warning(
                "FTS5 全文索引不可用（SQLite 缺少 trigram 分词支持），检索退回纯向量: %s", e
            )

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

    def fts_search(self, query: str, k: int = 5) -> list[int]:
        """FTS5 关键词检索，返回按 bm25 相关度（ORDER BY rank）排序的条目 id 列表。

        查询构建：按空白切词，每个 token 内部双引号翻倍后用双引号包裹，用 OR 连接。
        任何 FTS 查询错误一律返回 []（绝不抛出）。注意 trigram 需 ≥3 字符才能命中。
        """
        if not self.fts_enabled:
            return []
        tokens = query.split()
        if not tokens:
            return []
        match_expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        try:
            rows = self._conn.execute(
                "SELECT rowid FROM entries_fts WHERE entries_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (match_expr, k),
            ).fetchall()
        except sqlite3.Error:
            return []
        return [r[0] for r in rows]

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


def rrf_fuse(vec_ids: list[int], fts_ids: list[int],
             k: int = 5, rrf_k: int = 60) -> list[tuple[int, float]]:
    """倒数排名融合（Reciprocal Rank Fusion）。

    对两个已排序的 id 列表，按 score = Σ 1/(rrf_k + rank) 累加（rank 从 1 起），
    同时出现在两个列表中的条目得分更高；按得分降序取前 k，
    返回 (entry_id, rrf_score) 列表。
    """
    scores: dict[int, float] = {}
    for ranked in (vec_ids, fts_ids):
        for rank, eid in enumerate(ranked, start=1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ordered[:k]


# ── AI Client ─────────────────────────────────────────────────────────────────

def normalize_llm_text(text: str) -> str:
    """修复 LLM 把整段输出写成字面 \\n 转义的缺陷。

    仅当字面 \\n 很多而真实换行几乎没有时才替换，
    避免破坏正文里合法的转义（如 printf "a\\nb"、sed 表达式）。
    """
    if text.count("\\n") >= 3 and text.count("\n") <= 1:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return text


def strip_outer_fences(text: str) -> str:
    """去掉包裹整个输出的 ``` 围栏（含 ```json / ```markdown 语言标注）。"""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


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
            "从 <content> 中的文本提取元数据，输出一个 JSON 对象：\n"
            '{"title": "简短标题", "tags": "标签1,标签2,标签3"}\n'
            "要求：\n"
            "- title：10~25 字，概括主题，不带标点结尾\n"
            "- tags：3~6 个，逗号分隔；用适中粒度的领域词/关键词，技术或非技术均可"
            "（如 docker、ffmpeg、运维、健康、旅行、决策），"
            "英文一律小写，与正文语言一致，不要重复 title\n"
            "- 只输出 JSON，不要解释\n\n"
            f"<content>\n{content[:4000]}\n</content>"
        )
        messages = [{"role": "user", "content": prompt}]
        try:
            resp = self._client.chat.completions.create(
                model=self.config.llm_model, messages=messages,
                temperature=0.2, max_tokens=300,
                response_format={"type": "json_object"},
            )
        except Exception:
            # 模型/网关不支持 JSON mode 时退回普通模式
            resp = self._client.chat.completions.create(
                model=self.config.llm_model, messages=messages,
                temperature=0.2, max_tokens=300,
            )
        raw = strip_outer_fences(resp.choices[0].message.content)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("extract_meta: JSON 解析失败，原始输出: %.100s", raw)
            return "", ""
        title = normalize_llm_text(str(data.get("title", ""))).strip()
        tags = normalize_llm_text(str(data.get("tags", ""))).strip()
        tags = ",".join(t.strip().lower() if t.strip().isascii() else t.strip()
                        for t in tags.split(",") if t.strip())
        return title, tags

    def format_content(self, raw_text: str) -> str:
        """用 LLM 将自由文本格式化为结构化 Markdown，输出做围栏/换行归一化。"""
        prompt = (
            "把 <raw> 中的文本整理成一篇结构化的 Markdown 笔记。规则：\n"
            "- 用 ## 小标题概括主题，内容多时用 ### 分节\n"
            "- 命令/代码放入 ``` 围栏代码块并标注语言\n"
            "- 参数、选项用列表说明\n"
            "- 保留原文全部信息，不删减、不编造、不评论\n"
            "- 换行必须是真实换行符，严禁输出字面的 \\n 转义序列\n"
            "- 直接输出 Markdown 正文，不要用 ```markdown 包裹，不要任何解释\n\n"
            f"<raw>\n{raw_text}\n</raw>"
        )
        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=4000,
        )
        return normalize_llm_text(strip_outer_fences(resp.choices[0].message.content))

    def ask(self, query: str, context_entries: list[dict]) -> str:
        if not context_entries:
            context_text = "（知识库中没有找到相关内容）"
        else:
            parts = []
            for e in context_entries:
                parts.append(
                    f"--- 条目 #{e['id']} (标题: {e['title']}) ---\n"
                    f"内容: {e['content']}\n标签: {e['tags']}"
                )
            context_text = "\n\n".join(parts)
        resp = self._client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": (
                    "你是一个个人知识库助手，只根据提供的知识库条目回答问题。\n"
                    "- 回答准确、可直接使用（命令、步骤、事实、结论均可）\n"
                    "- 引用了哪些条目，在对应内容后标注来源，如 [#5]\n"
                    "- 条目只部分覆盖问题时，说明哪部分来自知识库、哪部分缺失\n"
                    "- 知识库没有相关信息就如实说明，严禁编造"
                )},
                {"role": "user", "content": (
                    f"知识库条目：\n{context_text}\n\n"
                    f"用户问题：{query}"
                )},
            ],
            temperature=0.3, max_tokens=2000,
        )
        return normalize_llm_text(resp.choices[0].message.content.strip())


# ── High-Level API ────────────────────────────────────────────────────────────

class DuplicateEntryError(Exception):
    """add() 查重命中阈值时抛出，携带相似条目列表（每项为完整条目 + score）。"""

    def __init__(self, similar: list[dict]):
        self.similar = similar
        super().__init__(f"发现 {len(similar)} 条相似条目，相似度已达查重阈值")


class KnowledgeBase:
    """顶层 API，组合 Store 和 AI Client。"""

    def __init__(self, config: Config):
        self.config = config
        self.store = KBStore(config.storage.db_path)
        self.ai = AIClient(config.api) if config.api.api_key else None

    def add(self, content: str, title: str = "", tags: str = "",
            source: str = "cli", auto_meta: bool = True, format_md: bool = True,
            force: bool = False) -> dict:
        """添加一条知识。auto_meta 提取元数据，format_md 格式化为 Markdown。

        force=False（默认）时，若新内容与已有条目的嵌入余弦相似度达到
        `config.search.dedup_threshold`，抛出 DuplicateEntryError 并拒绝写入；
        force=True 或未配置 AI / 嵌入生成失败时跳过查重。
        """
        if format_md and self.ai:
            try:
                content = self.ai.format_content(content)
            except Exception as e:
                logger.warning("Markdown 格式化失败，使用原文: %s", e)

        if auto_meta and self.ai and (not title or not tags):
            try:
                auto_title, auto_tags = self.ai.extract_meta(content)
                title = title or auto_title
                tags = tags or auto_tags
            except Exception as e:
                logger.warning("元数据提取失败: %s", e)

        embedding = None
        if self.ai:
            try:
                embedding = self.ai.embed(content)
            except Exception as e:
                logger.warning("嵌入生成失败，该条目将无法被语义检索: %s", e)

        if not force and self.ai is not None and embedding is not None:
            candidates = search_similar(self.store, embedding, k=3)
            dupes = [e for e in candidates if e.get("score", 0) >= self.config.search.dedup_threshold]
            if dupes:
                raise DuplicateEntryError(dupes)

        entry_id = self.store.add(content, title, tags, source, embedding)
        return {"id": entry_id, "title": title, "tags": tags, "content": content}

    def update_entry(self, entry_id: int, content: Optional[str] = None,
                     title: Optional[str] = None, tags: Optional[str] = None,
                     format_md: bool = False, auto_meta: bool = False) -> Optional[dict]:
        """更新条目。content 为整体替换（重新生成嵌入），title/tags 单独可改。"""
        entry = self.store.get(entry_id)
        if not entry:
            return None
        if content:
            if format_md and self.ai:
                try:
                    content = self.ai.format_content(content)
                except Exception as e:
                    logger.warning("Markdown 格式化失败，使用原始内容: %s", e)
            embedding = None
            if self.ai:
                try:
                    embedding = self.ai.embed(content)
                except Exception as e:
                    logger.warning("嵌入生成失败，该条目将无法被语义检索: %s", e)
            self.store.update_content(entry_id, content, embedding)
            if auto_meta and self.ai and not (title or tags):
                try:
                    title, tags = self.ai.extract_meta(content)
                except Exception as e:
                    logger.warning("元数据提取失败: %s", e)
        if title or tags:
            cur = self.store.get(entry_id)
            self.store.update_title_tags(entry_id, title or cur["title"], tags or cur["tags"])
        return self.store.get(entry_id)

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
        """混合检索：向量语义 + FTS5 关键词，RRF 融合排序。

        - 有 API key（self.ai）时走向量路径；FTS 索引可用时走关键词路径。
        - 两路结果用 RRF 融合，返回条目带 score(RRF 融合分)、vec_score(余弦相似度,
          不在向量结果中时为 0.0)、fts_hit(bool)。
        - 无 API key 时退化为纯 FTS（无需 key 也能检索）；两者皆不可用时返回 []。
        """
        vec_score_map: dict[int, float] = {}
        vec_ids: list[int] = []
        if self.ai:
            try:
                query_vec = self.ai.embed(query)
                for e in search_similar(self.store, query_vec, k):
                    vec_score_map[e["id"]] = e["score"]
                    vec_ids.append(e["id"])
            except Exception as e:
                logger.warning("向量检索失败，本次退回纯关键词检索: %s", e)

        fts_ids = self.store.fts_search(query, k)
        fts_id_set = set(fts_ids)

        if not vec_ids and not fts_ids:
            return []

        results: list[dict] = []
        for eid, rrf_score in rrf_fuse(vec_ids, fts_ids, k):
            entry = self.store.get(eid)
            if entry is None:
                continue
            entry["score"] = rrf_score
            entry["vec_score"] = vec_score_map.get(eid, 0.0)
            entry["fts_hit"] = eid in fts_id_set
            results.append(entry)
        return results

    def ask(self, query: str, k: int = 5) -> str:
        if not self.ai:
            return "错误：未配置 API key，无法使用问答功能。"
        min_score = self.config.search.min_score
        entries = [
            e for e in self.search(query, k)
            if e.get("vec_score", 0.0) >= min_score or e.get("fts_hit", False)
        ]
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
