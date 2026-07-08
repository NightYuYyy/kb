"""tests/test_hybrid_search.py — 混合检索（FTS5 + 向量 + RRF）单元测试。

约束：无网络、无真实 LLM/embedding；用确定性的 FakeAI 注入 kb.ai。
构造 Config() 于代码内，绝不加载真实 data/config.yaml。
"""

from __future__ import annotations

import os
import sqlite3
import sys
import time

import numpy as np

# 让位于 tests/ 下的用例能 import 根目录的 kb_core（python -m pytest 已把 cwd 入栈，
# 此处再补一次以兼容直接 pytest 调用）。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kb_core import Config, KBStore, KnowledgeBase, rrf_fuse  # noqa: E402


# ── 测试替身 ──────────────────────────────────────────────────────────────────

class FakeAI:
    """确定性 AI 替身：仅实现 embed（search 只用到它）。

    vectors 为 {文本: np.ndarray} 精确映射，命中即返回对应向量；
    否则回退到基于内容的确定性向量（不涉及网络）。
    """

    def __init__(self, vectors: dict[str, np.ndarray] | None = None):
        self.vectors = vectors or {}

    def embed(self, text: str) -> np.ndarray:
        if text in self.vectors:
            return self.vectors[text].astype(np.float32)
        # 确定性回退：按字符和生成两维向量
        acc = sum(ord(c) for c in text) or 1
        return np.array([acc % 7, acc % 5], dtype=np.float32)


def _fresh_kb(tmp_path, ai=None) -> KnowledgeBase:
    config = Config()
    config.storage.db_path = str(tmp_path / "kb.db")
    kb = KnowledgeBase(config)  # 无 api_key → kb.ai is None
    kb.ai = ai
    return kb


# ── FTS 建表 / 触发器 / rebuild ───────────────────────────────────────────────

def test_fts_created_on_fresh_db(tmp_path):
    store = KBStore(str(tmp_path / "fresh.db"))
    assert store.fts_enabled is True

    tables = {
        r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "entries_fts" in tables

    triggers = {
        r[0] for r in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        ).fetchall()
    }
    assert {"entries_ai", "entries_ad", "entries_au"} <= triggers


def test_fts_rebuild_on_preexisting_db_without_index(tmp_path):
    """打开一个缺 FTS 索引的旧库时应创建索引，并 rebuild 把旧行灌入。"""
    db = str(tmp_path / "legacy.db")
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE entries (
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
    now = time.time()
    con.execute(
        "INSERT INTO entries (content, title, tags, source, embedding, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("旧数据：安装 omni-upgrade.nu 到 nushell", "老条目", "nushell", "cli", None, now, now),
    )
    con.commit()
    con.close()

    store = KBStore(db)  # 触发 _init_fts：检测无索引 → 建表 + rebuild
    assert store.fts_enabled is True
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='entries_fts'"
    ).fetchone()
    assert row is not None
    # rebuild 应使旧行可被检索
    assert store.fts_search("omni", 5) != []


# ── fts_search 查询构建与同步 ─────────────────────────────────────────────────

def test_fts_sync_on_add_update_delete(tmp_path):
    store = KBStore(str(tmp_path / "sync.db"))
    # 注意 tags 不含 docker：update_content 只改 content，tags 列若含 docker 仍会命中
    eid = store.add("docker compose 部署说明", "标题", "容器")
    assert store.fts_search("docker", 5) == [eid]

    # 更新内容后旧 token 失效、新 token 命中
    store.update_content(eid, "改写为 kubernetes helm 部署")
    assert store.fts_search("docker", 5) == []
    assert store.fts_search("kubernetes", 5) == [eid]

    # 删除后不再命中
    store.delete(eid)
    assert store.fts_search("kubernetes", 5) == []


def test_fts_search_quotes_are_escaped(tmp_path):
    """含双引号的 token 不应破坏 MATCH 语法（内部双引号翻倍）。"""
    store = KBStore(str(tmp_path / "quote.db"))
    eid = store.add('配置项 say"hello"world 示例', "标题", "")
    # 查询含双引号，构建时应转义，不抛异常且能命中
    assert store.fts_search('say"hello"world', 5) == [eid]


def test_fts_search_disabled_returns_empty(tmp_path):
    store = KBStore(str(tmp_path / "off.db"))
    store.add("docker 说明", "标题", "docker")
    store.fts_enabled = False
    assert store.fts_search("docker", 5) == []


# ── RRF 融合 ──────────────────────────────────────────────────────────────────

def test_rrf_ranks_common_entry_above_single_list(tmp_path):
    vec_ids = [1, 2, 3]
    fts_ids = [2, 4, 5]
    fused = rrf_fuse(vec_ids, fts_ids, k=5)
    ranked = [eid for eid, _ in fused]
    scores = dict(fused)

    # id 2 同时出现在两个列表 → 得分最高、排第一
    assert ranked[0] == 2
    assert scores[2] > scores[1]  # 1 只在向量列表
    assert scores[2] > scores[4]  # 4 只在 FTS 列表
    # 得分公式核对：2 -> 1/(60+2)+1/(60+1)
    assert abs(scores[2] - (1 / 62 + 1 / 61)) < 1e-12


# ── 无 AI 的纯 FTS 检索 ───────────────────────────────────────────────────────

def test_exact_token_hit_without_ai(tmp_path):
    kb = _fresh_kb(tmp_path, ai=None)
    assert kb.ai is None
    kb.add("安装 omni-upgrade.nu 到 nushell 配置目录", format_md=False, auto_meta=False)

    results = kb.search("omni", k=5)
    assert results, "无 API key 时也应能通过 FTS 命中"
    assert any("omni-upgrade.nu" in r["content"] for r in results)

    r = results[0]
    assert r["fts_hit"] is True
    assert r["vec_score"] == 0.0
    assert r["score"] > 0


def test_search_neither_available_returns_empty(tmp_path):
    kb = _fresh_kb(tmp_path, ai=None)
    kb.add("omni-upgrade.nu 笔记", format_md=False, auto_meta=False)
    assert kb.search("omni", 5)  # FTS 可用时能命中
    kb.store.fts_enabled = False  # 关闭 FTS，且无 AI → 两者皆不可用
    assert kb.search("omni", 5) == []


# ── search 结果契约字段 / 类型 + ask 过滤语义 ─────────────────────────────────

def test_search_result_fields_and_types_with_ai(tmp_path):
    qv = np.array([1.0, 0.0], dtype=np.float32)
    kb = _fresh_kb(tmp_path, ai=FakeAI({"docker": qv}))

    # A：向量高相似（余弦≈1）且 FTS 命中 docker
    id_a = kb.store.add("docker compose 用法", "A", "docker", "cli",
                        np.array([1.0, 0.0], dtype=np.float32))
    # B：向量正交（余弦 0），且不含 docker → 仅出现在向量 top-k
    id_b = kb.store.add("完全无关的内容 xyz", "B", "", "cli",
                        np.array([0.0, 1.0], dtype=np.float32))

    results = kb.search("docker", k=5)
    assert results
    for r in results:
        assert isinstance(r["score"], float)
        assert isinstance(r["vec_score"], float)
        assert isinstance(r["fts_hit"], bool)

    by_id = {r["id"]: r for r in results}
    # A：既是向量命中又是 FTS 命中
    assert by_id[id_a]["vec_score"] > 0.99
    assert by_id[id_a]["fts_hit"] is True
    # B：向量结果里出现但非 FTS 命中，vec_score 为 0.0
    assert by_id[id_b]["fts_hit"] is False
    assert by_id[id_b]["vec_score"] == 0.0
    # 同时命中双路的 A 应排在只命中单路的 B 之前
    assert results[0]["id"] == id_a


def test_ask_filter_semantics_at_search_level(tmp_path):
    """ask 过滤规则 vec_score >= min_score or fts_hit 在 search 结果层面成立。"""
    qv = np.array([1.0, 0.0], dtype=np.float32)
    kb = _fresh_kb(tmp_path, ai=FakeAI({"docker": qv}))
    min_score = kb.config.search.min_score  # 默认 0.35

    # 高相似向量 + 无 FTS token：应因 vec_score 通过
    id_vec = kb.store.add("语义相近但无关键词", "V", "", "cli",
                          np.array([1.0, 0.0], dtype=np.float32))
    # 低相似向量 + FTS 命中 docker：应因 fts_hit 通过
    id_fts = kb.store.add("docker 命令备忘", "F", "docker", "cli",
                          np.array([0.0, 1.0], dtype=np.float32))

    results = kb.search("docker", k=5)
    kept = [
        e for e in results
        if e.get("vec_score", 0.0) >= min_score or e.get("fts_hit", False)
    ]
    kept_ids = {e["id"] for e in kept}
    assert id_vec in kept_ids   # 靠 vec_score 保留
    assert id_fts in kept_ids   # 靠 fts_hit 保留

    vec_entry = next(e for e in kept if e["id"] == id_vec)
    fts_entry = next(e for e in kept if e["id"] == id_fts)
    assert vec_entry["vec_score"] >= min_score
    assert fts_entry["fts_hit"] is True
