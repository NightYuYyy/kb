"""
tests/test_dedup.py — Task 2: 服务端强查重（add 时）

覆盖：
- KnowledgeBase.add 层：命中阈值抛 DuplicateEntryError / force=True 跳过 / 低于阈值正常写入 / 无 AI 时跳过
- kb_web API 层：/api/add 命中查重返回 409（body 完全符合 Global Constraints 契约）、
  force=true 返回 200、鉴权仍然生效

不发起任何网络请求：AI 客户端全部用确定性 fake 对象替代。
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from kb_core import Config, DuplicateEntryError, KnowledgeBase, cosine_similarity
from kb_web import create_app


# ── Fakes / helpers ─────────────────────────────────────────────────────────

class FakeAIClient:
    """确定性 fake AI 客户端：按输入文本返回预设向量，不发起任何网络请求。"""

    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors

    def embed(self, text: str) -> np.ndarray:
        return self.vectors[text]

    def extract_meta(self, content: str):
        return "", ""

    def format_content(self, raw_text: str) -> str:
        return raw_text


def make_kb(tmp_path, ai=None) -> KnowledgeBase:
    config = Config()
    config.storage.db_path = str(tmp_path / "kb.db")
    kb = KnowledgeBase(config)
    kb.ai = ai
    return kb


DUPLICATE_MESSAGE = (
    "存在相似条目：请先 GET 对应条目，合并后用 PUT /api/entry/<id> 更新；"
    "确属不同主题时才用 force=true 强制新增"
)


# ── KnowledgeBase.add 层 ─────────────────────────────────────────────────────

def test_duplicate_detected_on_second_add(tmp_path):
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ai = FakeAIClient({"first entry": vec, "near duplicate entry": vec})
    kb = make_kb(tmp_path, ai)

    first = kb.add("first entry", auto_meta=False, format_md=False)

    with pytest.raises(DuplicateEntryError) as exc_info:
        kb.add("near duplicate entry", auto_meta=False, format_md=False)

    similar_ids = [s["id"] for s in exc_info.value.similar]
    assert first["id"] in similar_ids
    # 未写入
    assert kb.count() == 1


def test_force_true_inserts_despite_duplicate(tmp_path):
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ai = FakeAIClient({"first entry": vec, "near duplicate entry": vec})
    kb = make_kb(tmp_path, ai)

    kb.add("first entry", auto_meta=False, format_md=False)
    result = kb.add("near duplicate entry", auto_meta=False, format_md=False, force=True)

    assert result["id"]
    assert kb.count() == 2


def test_below_threshold_inserts_normally(tmp_path):
    vec_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vec_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # 正交 -> 相似度 0
    ai = FakeAIClient({"first entry": vec_a, "different topic entry": vec_b})
    kb = make_kb(tmp_path, ai)

    kb.add("first entry", auto_meta=False, format_md=False)
    result = kb.add("different topic entry", auto_meta=False, format_md=False)

    assert result["id"]
    assert kb.count() == 2


def test_no_ai_client_skips_dedup(tmp_path):
    kb = make_kb(tmp_path, ai=None)
    kb.add("first entry", auto_meta=False, format_md=False)
    result = kb.add("first entry", auto_meta=False, format_md=False)

    assert result["id"]
    assert kb.count() == 2


# ── API 层（TestClient）─────────────────────────────────────────────────────

@pytest.fixture
def sqlite_thread_patch(monkeypatch):
    """FastAPI TestClient 在工作线程中运行 app，KBStore 使用单一 sqlite 连接，
    需强制 check_same_thread=False 才能跨线程访问（仅测试用，不改生产连接代码）。"""
    orig_connect = sqlite3.connect

    def patched_connect(*args, **kwargs):
        kwargs["check_same_thread"] = False
        return orig_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", patched_connect)


def make_app(tmp_path, ai=None, auth_token: str = ""):
    config = Config()
    config.storage.db_path = str(tmp_path / "kb.db")
    config.server.auth_token = auth_token
    kb = KnowledgeBase(config)
    kb.ai = ai
    return create_app(kb), kb


def test_api_add_duplicate_returns_409_with_exact_contract(tmp_path, sqlite_thread_patch):
    long_content = "first entry " + ("x" * 400)
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ai = FakeAIClient({long_content: vec, "near duplicate entry": vec})
    app, kb = make_app(tmp_path, ai=ai)
    client = TestClient(app)

    r1 = client.post("/api/add", json={"content": long_content, "auto_meta": False, "format_md": False})
    assert r1.status_code == 200
    first_id = r1.json()["id"]

    r2 = client.post("/api/add", json={"content": "near duplicate entry", "auto_meta": False, "format_md": False})
    assert r2.status_code == 409

    detail = r2.json()["detail"]
    assert detail["error"] == "duplicate"
    assert detail["message"] == DUPLICATE_MESSAGE
    assert isinstance(detail["similar"], list) and len(detail["similar"]) >= 1

    item = detail["similar"][0]
    assert set(item.keys()) == {"id", "title", "score", "content"}
    assert item["id"] == first_id
    assert item["content"] == long_content[:300]
    assert len(item["content"]) == 300
    assert isinstance(item["score"], float)


def test_api_add_duplicate_score_rounded_to_2_decimals(tmp_path, sqlite_thread_patch):
    vec_a = np.array([1.0, 0.5, 0.2], dtype=np.float32)
    vec_b = np.array([0.9, 0.6, 0.1], dtype=np.float32)  # 与 vec_a 高度相似但不完全相同
    ai = FakeAIClient({"first entry": vec_a, "near duplicate entry": vec_b})
    app, kb = make_app(tmp_path, ai=ai)
    client = TestClient(app)

    client.post("/api/add", json={"content": "first entry", "auto_meta": False, "format_md": False})
    r2 = client.post("/api/add", json={"content": "near duplicate entry", "auto_meta": False, "format_md": False})
    assert r2.status_code == 409

    raw_score = float(cosine_similarity(vec_a.reshape(1, -1), vec_b)[0])
    expected = round(raw_score, 2)
    assert 0.85 <= raw_score < 1.0  # 确认测试向量确实命中默认阈值 0.85 但非完全相同
    assert r2.json()["detail"]["similar"][0]["score"] == expected


def test_api_add_force_true_returns_200(tmp_path, sqlite_thread_patch):
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ai = FakeAIClient({"first entry": vec, "near duplicate entry": vec})
    app, kb = make_app(tmp_path, ai=ai)
    client = TestClient(app)

    client.post("/api/add", json={"content": "first entry", "auto_meta": False, "format_md": False})
    r2 = client.post(
        "/api/add",
        json={"content": "near duplicate entry", "auto_meta": False, "format_md": False, "force": True},
    )
    assert r2.status_code == 200
    assert kb.count() == 2


def test_api_add_auth_still_enforced(tmp_path, sqlite_thread_patch):
    app, kb = make_app(tmp_path, ai=None, auth_token="secret-token")
    client = TestClient(app)

    r = client.post("/api/add", json={"content": "x"})
    assert r.status_code == 401

    r_ok = client.post(
        "/api/add", json={"content": "x"}, headers={"X-API-Key": "secret-token"}
    )
    assert r_ok.status_code == 200
