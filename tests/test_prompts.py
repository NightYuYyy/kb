"""
tests/test_prompts.py — Task 3：提示词从运维专用放宽为通用知识库，并加入 409 查重指引。

不涉及网络 / 真实 LLM 调用：只检查 prompt 源字符串，通过 inspect 获取方法源码后做子串断言。
"""

import inspect

from kb_core import AIClient
from kb_web import AGENT_PROMPT_TEMPLATE, build_agent_prompt


# ── build_agent_prompt：占位符替换 ────────────────────────────────────────────

def test_build_agent_prompt_substitutes_placeholders():
    prompt = build_agent_prompt("https://x", "t")
    assert "{KB_URL}" not in prompt
    assert "{TOKEN}" not in prompt
    assert "https://x/api/list" in prompt
    assert "X-API-Key: t" in prompt


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
