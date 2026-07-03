# KB 知识库 — Agent 接入提示词

把提示词发给任意新环境的 agent（Claude Code、Codex、ChatGPT 等），它会**把知识库安装为自己的 skill**（`~/.claude/skills/kb/SKILL.md`、`~/.codex/skills/kb/SKILL.md` 等）并验证连接——粘贴一次，重开会话依然生效；无文件权限的网页 AI 则在当前会话内遵守。

提示词内容由后端统一生成（单一来源：`kb_web.py` 中的 `AGENT_PROMPT_TEMPLATE`），两种获取方式：

## 方式一：Web UI 一键复制（推荐）

登录 Web UI 后点右上角「📋 复制提示词」，得到已填好真实地址和 token 的完整版本。

## 方式二：API 获取

```bash
curl -s "https://your-kb-host/api/prompt?base=https://your-kb-host" \
  -H "X-API-Key: <auth_token>"
```

返回 `{"prompt": "..."}`，`base` 参数是知识库对外地址（省略时输出 `{KB_URL}` 占位符）。token 取自 `data/config.yaml` 的 `server.auth_token`。

## 提示词涵盖的行为约定

- 查询自动：遇到命令/脚本/服务器/配置类问题先查库（`/api/ask`），不凭猜测
- **记录仅在用户明确要求时**（"记录/记到知识库"）——AI 不主动写库，最多提醒一句
- 写入前先 `/api/search` 查重：已有同主题条目 → `PUT /api/entry/<id>` 合并更新，而不是重复新增
- 不记录明文凭据（密码、API key、私钥）——只记引用位置
- 内容用 Markdown、代码块标注语言、一次记全用途/参数/路径/示例

修改提示词内容时只需编辑 `kb_web.py` 的 `AGENT_PROMPT_TEMPLATE`，Web UI 复制按钮和本文档引用的接口会同步生效。
