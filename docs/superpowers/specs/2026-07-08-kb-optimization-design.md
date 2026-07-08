# kb 知识库优化设计

日期:2026-07-08
状态:已确认(需求访谈 + 设计均经用户逐项确认)

## 背景与目标

kb 定位确认为 **AI agent 外挂记忆为主**:各设备上的 Claude Code / Codex 等 agent 通过
REST API 自动查询、按用户指令记录;人工 CLI / Web UI 为辅。内容范围从「命令/运维」
**放宽为通用知识**(生活信息、决策记录、阅读笔记等均可)。

需求访谈确认的六项结论:

| # | 维度 | 结论 |
|---|------|------|
| 1 | 定位 | Agent 外挂记忆为主 |
| 2 | 内容范围 | 通用知识,不限技术 |
| 3 | 检索 | 关键词 + 向量混合检索 |
| 4 | 写入查重 | 服务端硬查重,超阈值拒绝 |
| 5 | 部署 | VPS 公网 + HTTPS |
| 6 | 备份 | 服务端自动定时备份 |

推进方式:**方案 B(功能先行)**——第一批本地开发验证,第二批部署上线,第三批并发与周边。

## 第一批:检索与写入质量

### 1.1 混合检索(FTS5 + 向量 + RRF)

- 新增 FTS5 外部内容虚表 `entries_fts`(索引 `title`、`content`、`tags`,
  `content=entries`,`tokenize='trigram'`)。trigram 分词对中文和英文精确子串
  (工具名、错误码、IP、文件名)均可命中,无需外部分词库。
- 用 SQLite 触发器(INSERT/UPDATE/DELETE)保持 `entries_fts` 与 `entries` 同步。
- 迁移:`KBStore._init_schema` 中 `CREATE VIRTUAL TABLE IF NOT EXISTS`;首次创建时
  执行 `INSERT INTO entries_fts(entries_fts) VALUES('rebuild')` 重建索引,老库无痛升级。
- 检索流程 `KnowledgeBase.search(query, k)`:
  1. 向量检索(现有逻辑)→ 排名列表 A;
  2. FTS5 检索(bm25 排序,查询词做 FTS 语法转义)→ 排名列表 B;
  3. RRF 融合:`score = Σ 1/(60 + rank_i)`,取 top-k;结果保留各路来源分数便于调试。
- 未配置 API key 时退化为纯关键词检索(`search` 从「不可用」变为「可用」)。
- 无新增第三方依赖;要求 SQLite ≥ 3.34(Python 3.12+ 内置版本满足)。

### 1.2 服务端硬查重

- `/api/add` 写入前:对新内容生成嵌入,与全库做相似度检查;最高相似度 ≥
  `search.dedup_threshold`(配置项,默认 0.85)时返回 **HTTP 409**,响应体含
  `similar: [{id, title, score, content 前 300 字}]` 与指引文案(改用
  `PUT /api/entry/<id>` 合并更新)。
- `AddRequest` 增加 `force: bool = False`,为 true 时跳过查重强制新增。
- CLI `kb add` 收到 409 时打印相似条目列表,提示 `--force` 或 `kb update`;
  `RemoteKB.add` 透传 force 与 409 语义。
- 未配置 API key(无法生成嵌入)时跳过查重,行为与现状一致。

### 1.3 提示词放宽为通用知识

三处改写,单一来源仍是 `kb_web.py` 的 `AGENT_PROMPT_TEMPLATE`:

- **Agent 接入模板**:查询触发范围从「命令/脚本/服务器/运维」放宽为「任何可能
  记录过的事实、决定、配置、笔记」;新增 409 查重语义说明(收到 409 → 按返回的
  相似条目改走更新);记录仍然仅限用户明确要求。
- **`extract_meta` 提示词**:标签示例从纯工具名扩展为「工具名/领域词/主题词」。
- **`ask` 系统提示词**:去掉「直接给出命令或步骤」的命令中心措辞,改为「给出
  准确、可执行的信息」。
- 本机 `~/.claude/skills/kb/SKILL.md` 以新模板为源重新生成覆盖。

## 第二批:上线

### 2.1 VPS 部署与安全

- 新增 `docker-compose.prod.yml`:GHCR 镜像 + Caddy 反代(自动 HTTPS,域名走
  环境变量),`./data` 卷挂载,`KB_AUTH_TOKEN` 环境变量注入(已支持)。
- 安全强化:`create_app` / serve 启动时,若监听非回环地址且 `auth_token` 为空,
  **拒绝启动**并给出明确报错。
- 上线动作:轮换现有 auth_token(旧 token 已在本地明文出现过)、填写
  `server.public_url`、各设备 `kb connect`。
- VPS 实操可经 ssh-mcp 代执行。

### 2.2 自动备份与导出

- FastAPI lifespan 后台任务:每 `backup.interval_hours`(默认 24)执行
  `VACUUM INTO data/backups/kb-<时间戳>.db`,保留最近 `backup.keep`(默认 7)份,
  超出自动清理。
- 新增 `GET /api/export`(需认证):默认导出全库 JSON(含全部字段,可完整还原),
  `?format=md` 输出 Markdown 打包(人读);CLI 对应 `kb export [路径]`。
- 备份目录在数据卷内;异地容灾靠 `kb export` 拉到本地,VPS 层快照作为补充建议
  (不在应用层实现)。

## 第三批:并发与周边

### 3.1 并发改造

- Web 端点从 `async def` 改为 `def`(FastAPI 线程池执行),消除同步 LLM 调用
  (最长 60s)阻塞事件循环的问题。
- `KBStore` 改为按操作建立 SQLite 连接(WAL 模式下开销可忽略),消除单连接的
  跨线程限制;此改动同时使 pytest/TestClient 可用。

### 3.2 周边小修

- CLI `kb add` / `kb update` 增加 `--raw`:跳过 LLM Markdown 重写,原文入库
  (逐字引用、已格式化内容适用)。
- tags 筛选从 `LIKE %tag%` 改为逗号边界精确匹配(查 `git` 不再命中 `github`)。
- `min_score`(0.35)与 `dedup_threshold` 收进配置 `search` 段,消除硬编码重复。
- 基础 pytest 测试:存储 CRUD、混合检索(含纯关键词退化)、查重 409/force、
  API 认证、KB_AUTH_TOKEN 覆盖。

## 配置新增(均有默认值,老配置文件无需改动即兼容)

```yaml
search:
  min_score: 0.35        # ask 检索的最低相似度
  dedup_threshold: 0.85  # add 查重阈值,≥ 此值返回 409
backup:
  interval_hours: 24     # 自动备份间隔
  keep: 7                # 快照保留份数
```

## 明确不做(YAGNI)

- 长文自动分块——靠 Agent 提示词约束「一事一条」粒度;
- 引入向量数据库——几千条内全量余弦足够;
- Web UI 大改——agent 为主定位下人用界面保持现状;
- 应用层异地备份上传——`kb export` + VPS 快照覆盖。

## 错误处理原则

- 所有 AI 依赖能力(格式化/元数据/嵌入/查重)在未配 API key 或调用失败时降级,
  不阻断核心读写;
- 查重 409 是唯一的「拒绝写入」路径,且必须附带可操作的相似条目信息;
- 公网监听 + 空 token 是唯一的「拒绝启动」路径。

## 验收标准

1. 搜「omni」能命中含 `omni-upgrade.nu` 的条目(关键词路径);模糊语义查询命中率不低于现状;
2. 重复主题 add 返回 409 且相似条目正确;`force: true` 可写入;
3. 未配 API key:search 可用(纯关键词)、ask 返回 503、add 直接入库;
4. 公网监听 + 空 token 拒绝启动;
5. 备份目录按周期出现快照且数量不超过 keep;`kb export` 产物可还原全部条目;
6. 并发 2 个 `/api/ask` 期间,`/api/list` 响应不被阻塞;
7. pytest 全绿。
