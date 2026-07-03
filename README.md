# kb — 个人知识库

记录命令、脚本、工具用法和运维信息,支持语义检索 + AI 问答。CLI + Web UI + REST API,可让任意 AI(ChatGPT / Claude / Codex...)通过提示词直接读写。

- 存储:SQLite(单文件 `data/kb.db`),嵌入向量 + numpy 余弦相似度检索
- AI:硅基流动 API(bge-m3 嵌入 + DeepSeek 问答/格式化),未配 key 时纯存储功能可用

## 部署(第一台机器 / 服务器)

```bash
git clone https://github.com/NightYuYyy/kb.git && cd kb
pip install -e .
cp data/config.yaml.example data/config.yaml
# 编辑 data/config.yaml:填 api_key;生成并填 auth_token;
# 要给其他设备用的话,填 public_url 为对外地址
kb serve            # http://<host>:8765
```

Docker 方式:`docker compose up -d`(配置同样读挂载的 `data/config.yaml`)。push 到 master 会由 GitHub Actions 自动构建镜像,也可直接拉取:`docker pull ghcr.io/nightyuyyy/kb:latest`。

**安全提醒**:`auth_token` 必须改成随机值(`python -c "import secrets; print(secrets.token_urlsafe(32))"`),不要用示例值;服务暴露公网时建议套 HTTPS 反代。

## 新设备接入

前提:知识库已部署在新设备可达的地址(公网/内网/Tailscale),`localhost` 提示词只在部署机本机有效。

**方式一:Agent 一键接入(推荐,无需手动安装)**

打开 Web UI → 登录 → 点「📋 复制提示词」→ 发给新环境里的 agent(Claude Code / Codex 等)。这是一个**接入引导提示词**:agent 会把内嵌真实地址和 token 的「KB 技能定义」安装为自己的 skill(`~/.claude/skills/kb/SKILL.md`、`~/.codex/skills/kb/SKILL.md` 等),再验证连接并汇报——重开会话依然生效。网页对话 AI 无文件权限则在当前会话内直接遵守。也可 `GET /api/prompt?base=<对外地址>` 获取。

**方式二:CLI 接入**

```bash
pip install git+https://github.com/NightYuYyy/kb.git
kb connect https://kb.example.com --token <auth_token>   # 验证并保存到 ~/.kb/remote.json
kb ask "docker 怎么清理"                                  # 之后所有命令自动走远程
kb disconnect                                             # 恢复本地模式
```

临时覆盖:`kb --remote <url> --token <t> <命令>` 或环境变量 `KB_REMOTE_URL` / `KB_TOKEN`(优先级高于保存的连接)。

## CLI 命令

| 命令 | 说明 |
|------|------|
| `kb add "内容"` | 录入(AI 自动格式化 Markdown + 提标题标签,`--no-auto` 关闭) |
| `kb ask "问题"` | 语义检索 + AI 回答 |
| `kb search "关键词"` | 语义检索列出条目 |
| `kb list` / `kb show <id>` | 列出 / 查看全文(`--tag` 筛选) |
| `kb update <id>` | 更新内容或标题标签(`--content` / `--title` / `--tags`) |
| `kb delete <id>` / `kb reformat <id>` | 删除 / AI 重新格式化 |
| `kb connect <url>` / `kb disconnect` | 保存 / 删除远程连接 |
| `kb serve` | 启动 Web UI + API(仅本地模式) |

## REST API

所有接口需请求头 `X-API-Key: <auth_token>`。

| 接口 | 说明 |
|------|------|
| `POST /api/add` | 新增 `{content, title?, tags?, format_md, auto_meta}` |
| `POST /api/ask` | 检索 + AI 回答 `{query, k?}` |
| `POST /api/search` | 语义检索 `{query, k?}` |
| `GET /api/list?tag=&limit=&offset=` | 列出条目 |
| `GET /api/entry/{id}` | 查看单条 |
| `PUT /api/entry/{id}` | 更新 `{content?, title?, tags?, format_md?, auto_meta?}` |
| `DELETE /api/entry/{id}` | 删除 |
| `POST /api/entry/{id}/reformat` | AI 重新格式化 |
| `GET /api/prompt?base=` | 获取 Agent 提示词(单一来源,见 `prompt-snippet.md`) |

## Agent 行为约定(提示词内置)

- 查询自动:遇到不确定的命令/配置先查库,不凭猜测
- **记录仅在用户明确要求时**:AI 不主动写库,最多提醒
- 写入前查重:已有同主题条目时走 `PUT` 合并更新,不重复新增
- 不记录明文凭据,只记引用位置
