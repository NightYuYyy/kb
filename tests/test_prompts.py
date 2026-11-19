"""
tests/test_prompts.py — Task 3：提示词从运维专用放宽为通用知识库，并加入 409 查重指引。

不涉及外部网络 / 真实 LLM 调用：使用 TestClient 和本地字符串断言。
"""

import inspect
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from kb_core import AIClient, Config, ServerConfig
from kb_web import AGENT_PROMPT_TEMPLATE, build_agent_prompt, create_app


# ── build_agent_prompt：占位符替换 ────────────────────────────────────────────

def test_build_agent_prompt_substitutes_placeholders():
    prompt = build_agent_prompt("https://x", "t")
    assert "{KB_URL}" not in prompt
    assert "{TOKEN}" not in prompt
    assert "https://x/api/list" in prompt
    assert "X-API-Key: t" in prompt


# ── /api/prompt：以当前访问入口为准 ─────────────────────────────────────────

@pytest.fixture
def prompt_api():
    config = Config(server=ServerConfig(
        auth_token="test-key", public_url="http://192.0.2.10:8765",
    ))
    app = create_app(SimpleNamespace(config=config))
    with TestClient(app, base_url="http://internal.example.test:8765") as client:
        yield client


@pytest.mark.parametrize("origin", [
    "https://kb.example.test",
    "https://kb.example.test:8443",
    "http://198.51.100.25:8765",
    "http://[2001:db8::25]:8765",
])
def test_browser_origin_overrides_stale_config_and_internal_address(prompt_api, origin):
    response = prompt_api.get("/api/prompt", params={"base": origin},
                              headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    prompt = response.json()["prompt"]
    assert f"服务地址：{origin}\n" in prompt
    for endpoint in ("/api/list?limit=1", "/api/ask", "/api/add", "/api/entry/<id>"):
        assert origin + endpoint in prompt
    assert f"kb connect {origin} " in prompt
    assert "192.0.2.10" not in prompt
    assert "internal.example.test" not in prompt
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("origin", ["https://kb.example.test", "http://198.51.100.25:8765"])
def test_direct_api_uses_request_origin_when_base_is_omitted(prompt_api, origin):
    response = prompt_api.get(origin + "/api/prompt", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert f"服务地址：{origin}\n" in response.json()["prompt"]
    assert "192.0.2.10" not in response.json()["prompt"]


def test_explicit_origin_accepts_a_trailing_slash(prompt_api):
    response = prompt_api.get("/api/prompt", params={"base": "https://kb.example.test/"},
                              headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert "https://kb.example.test/api/list" in response.json()["prompt"]


@pytest.mark.parametrize("base", [
    "kb.example.test", "ftp://kb.example.test", "https://kb.example.test:65536",
    "https://kb.example.test/path", "https://kb.example.test?query=1",
])
def test_prompt_rejects_base_that_is_not_an_http_origin(prompt_api, base):
    response = prompt_api.get("/api/prompt", params={"base": base},
                              headers={"X-API-Key": "test-key"})
    assert response.status_code == 422
    assert "prompt" not in response.json()


def test_prompt_still_requires_authentication(prompt_api):
    response = prompt_api.get("/api/prompt")
    assert response.status_code == 401
    assert "prompt" not in response.json()


# ── AGENT_PROMPT_TEMPLATE：范围放宽 ───────────────────────────────────────────

def test_agent_prompt_no_longer_scoped_to_ops_only():
    assert "命令/脚本/服务器/运维" not in AGENT_PROMPT_TEMPLATE


def test_agent_prompt_mentions_non_technical_examples():
    for kw in ("事实", "决定", "偏好", "笔记"):
        assert kw in AGENT_PROMPT_TEMPLATE


def test_agent_prompt_keeps_record_only_on_explicit_request():
    assert "仅在用户明确要求时记录" in AGENT_PROMPT_TEMPLATE


# ── AGENT_PROMPT_TEMPLATE：409 查重契约 ───────────────────────────────────────

def test_agent_prompt_documents_409_duplicate_contract():
    assert "409" in AGENT_PROMPT_TEMPLATE
    assert "similar" in AGENT_PROMPT_TEMPLATE
    assert '"force": true' in AGENT_PROMPT_TEMPLATE
    assert "PUT" in AGENT_PROMPT_TEMPLATE
    assert "/api/entry/<id>" in AGENT_PROMPT_TEMPLATE
    assert "合并" in AGENT_PROMPT_TEMPLATE


# ── kb_core.AIClient 提示词字面量 ─────────────────────────────────────────────

def test_extract_meta_prompt_broadens_tag_examples():
    src = inspect.getsource(AIClient.extract_meta)
    # 新增的非技术领域示例
    for kw in ("健康", "旅行", "决策"):
        assert kw in src
    # 原有技术示例保留
    assert "docker" in src
    assert "ffmpeg" in src


def test_ask_prompt_drops_command_centric_phrase():
    src = inspect.getsource(AIClient.ask)
    assert "直接给出命令或步骤" not in src
    assert "命令" in src
    assert "步骤" in src
    assert "事实" in src
    assert "结论" in src
    # 来源标注与禁止编造规则保持不变
    assert "[#5]" in src
    assert "严禁编造" in src
