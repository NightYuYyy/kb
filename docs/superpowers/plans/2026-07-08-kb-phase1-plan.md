# kb Phase 1 Implementation Plan — Hybrid Search, Server-Side Dedup, Prompt Broadening

Source spec: `docs/superpowers/specs/2026-07-08-kb-optimization-design.md` (user-approved).
Branch: `feat/phase1-agent-memory`. Tasks 1-3 run in PARALLEL in isolated worktrees and
are merged by the controller — each task MUST stay inside its declared ownership zones.

## Global Constraints

- Codebase: three flat modules — `kb_core.py` (engine), `kb_web.py` (FastAPI), `kb_cli.py`
  (Typer CLI). Follow existing style: Chinese docstrings/comments, section banner comments,
  stdlib + numpy + openai + fastapi + typer only. NO new third-party dependencies.
- `SearchConfig` already exists in `kb_core.py` (committed before dispatch):
  `config.search.min_score` (default 0.35) and `config.search.dedup_threshold` (default 0.85).
  Do NOT re-add it; consume it.
- User-facing strings (CLI output, API messages, prompt copy) are Chinese, matching existing tone.
- Tests: pytest 9.1.1 is installed. Put tests under `tests/`. NO network calls and NO real
  LLM/embedding API usage — use a fake AI client object with deterministic `embed()` vectors
  (e.g. fixed numpy float32 arrays per input) injected as `kb.ai`. `data/config.yaml` is
  gitignored and absent in your worktree; construct `Config()` in code, never load real config.
- Known constraint: `KBStore` holds a single SQLite connection with default thread affinity.
  FastAPI `TestClient` runs the app in a worker thread, so API-level tests must monkeypatch
  `sqlite3.connect` to pass `check_same_thread=False` in a test fixture (tests only — do NOT
  change production connection code; that refactor is a later phase).
- Run `python -m pytest tests/ -q` (full suite) once before committing; `ruff check <changed files>`
  must pass. Commit messages: conventional commits, concise subject.
- 409 duplicate contract (binds Task 2 producer and Task 3 documentation):
  HTTP 409, body `detail` object:
  `{"error": "duplicate", "message": "存在相似条目：请先 GET 对应条目，合并后用 PUT /api/entry/<id> 更新；确属不同主题时才用 force=true 强制新增", "similar": [{"id": <int>, "title": <str>, "score": <float>, "content": <str, first 300 chars>}]}`
- Hybrid search result contract (binds Task 1; Task 2's dedup does NOT use it):
  each entry dict from `KnowledgeBase.search()` carries `score` (float, RRF fused score),
  `vec_score` (float cosine similarity, 0.0 when not in vector results), `fts_hit` (bool).
  Relevance filter for ask flows: `vec_score >= min_score or fts_hit`.

## Task 1: Hybrid search — FTS5 + vector + RRF fusion

**Ownership zones:** `kb_core.py` ONLY in: `KBStore._init_schema`, new `KBStore` FTS methods,
module-level search functions (`search_similar` area), `KnowledgeBase.search` and
`KnowledgeBase.ask`; `kb_web.py` ONLY the one filter line inside `api_ask`
(`entries = [e for e in kb.search(...) if e.get("score", 0) >= 0.35]`). Do NOT touch
`KnowledgeBase.add`, `api_add`, `AddRequest`, `AGENT_PROMPT_TEMPLATE`, `AIClient` prompt
strings, or `kb_cli.py`.

Requirements:

1. In `KBStore._init_schema`, create an external-content FTS5 index when missing:
   ```sql
   CREATE VIRTUAL TABLE entries_fts USING fts5(
     content, title, tags,
     content='entries', content_rowid='id', tokenize='trigram'
   );
   ```
   plus the standard three sync triggers (AFTER INSERT / AFTER DELETE / AFTER UPDATE on
   `entries`, using the `entries_fts(entries_fts, rowid, ...) VALUES('delete', ...)` pattern).
   Detect prior existence via `sqlite_master` BEFORE creating; on first creation run
   `INSERT INTO entries_fts(entries_fts) VALUES('rebuild')` so existing rows get indexed.
   Wrap FTS setup in try/except `sqlite3.OperationalError`: on failure set
   `self.fts_enabled = False` and `logger.warning(...)` (SQLite without trigram support);
   otherwise `self.fts_enabled = True`. All FTS query paths must respect `fts_enabled`.
2. New `KBStore.fts_search(query: str, k: int) -> list[int]` returning entry ids ranked by
   bm25 (`ORDER BY rank`). Query building: split the raw query on whitespace, escape each
   token by doubling internal double-quotes and wrapping in double quotes, join with ` OR `.
   Any FTS query error → return `[]` (never raise). Note trigram matches need ≥3 chars;
   that limitation is acceptable, do not work around it.
3. RRF fusion in `kb_core.py`: `score = Σ 1/(60 + rank)` over both ranked lists (rank starts
   at 1), sort desc, take top-k. Populate the Hybrid search result contract fields
   (`score` = RRF, `vec_score`, `fts_hit`) on every returned entry.
4. `KnowledgeBase.search(query, k)`: vector ranking (existing cosine path, unchanged math)
   requires `self.ai`; FTS ranking always available when `fts_enabled`. With no AI configured,
   return pure FTS results (search becomes useful without an API key — previously it returned
   `[]`). With neither available, return `[]`.
5. Relevance filtering: `KnowledgeBase.ask` currently filters `score >= min_score`; change to
   `vec_score >= self.config.search.min_score or fts_hit`, and update the duplicated filter
   line in `kb_web.api_ask` to the same rule (keep its response shape unchanged, `sources`
   still report each entry's `score`).
6. Tests (`tests/test_hybrid_search.py`): FTS table + triggers created on a fresh DB and after
   opening a pre-existing DB missing the index (rebuild indexes old rows); exact-token hit —
   store entry containing `omni-upgrade.nu`, query `omni` finds it via FTS with no AI client;
   RRF fusion ranks an entry present in both lists above single-list entries; add/update/delete
   keep FTS in sync; `ask`-filter semantics unit-tested at the `search` result level
   (vec_score/fts_hit fields present and correct types).

## Task 2: Server-side hard dedup on add

**Ownership zones:** `kb_core.py` ONLY in: new exception class + `KnowledgeBase.add`;
`kb_web.py` ONLY `AddRequest`, `AddResponse` (if needed), `api_add`; `kb_cli.py` ONLY the
`add` command, `RemoteKB.add`, and any new 409-printing helper. Do NOT touch
`KBStore._init_schema`, search functions, `api_ask`, `AGENT_PROMPT_TEMPLATE`, or `AIClient`
prompt strings.

Requirements:

1. New exception in `kb_core.py`: `class DuplicateEntryError(Exception)` carrying
   `similar: list[dict]` (each dict: full entry + `score`).
2. `KnowledgeBase.add(..., force: bool = False)`: after content formatting/meta/embedding are
   computed (existing flow unchanged), and only when `not force and self.ai is not None and
   embedding is not None`: run cosine similarity of the new embedding against stored embeddings
   (reuse the existing `search_similar`-style comparison against `self.store`, k=3). If the top
   score `>= self.config.search.dedup_threshold`, raise `DuplicateEntryError` with all matches
   `>= threshold` (do NOT insert). No AI / no embedding → skip dedup silently (current behavior).
3. `kb_web.py`: `AddRequest` gains `force: bool = False`; `api_add` passes it through and maps
   `DuplicateEntryError` to the exact 409 duplicate contract from Global Constraints
   (`content` truncated to 300 chars, `score` rounded to 2 decimals).
4. `kb_cli.py`: `kb add` gains `--force`; local mode catches `DuplicateEntryError`, remote mode
   catches `httpx.HTTPStatusError` with status 409 and parses the contract body. Both print the
   same Chinese guidance: list similar entries (id, title, score) and hint
   `kb update <id> --content ...` 合并 or `--force` 强制新增, then exit code 1.
   `RemoteKB.add` passes `force` in the JSON body.
5. Tests (`tests/test_dedup.py`): with a fake AI client whose `embed()` returns identical
   vectors for near-duplicate texts — add → second add raises `DuplicateEntryError` with the
   original entry in `similar`; `force=True` inserts; below-threshold vectors insert normally;
   no AI client → dedup skipped; API level (TestClient + the sqlite monkeypatch from Global
   Constraints): 409 body matches the contract exactly, `force: true` returns 200, auth still
   enforced on `/api/add`.

## Task 3: Broaden prompts from ops-only to general knowledge

**Ownership zones:** `kb_web.py` ONLY `AGENT_PROMPT_TEMPLATE` (and `build_agent_prompt` if
needed); `kb_core.py` ONLY the prompt string literals inside `AIClient.extract_meta` and
`AIClient.ask`. Do NOT touch schema/search/add code, endpoints, or `kb_cli.py`.

Requirements:

1. `AGENT_PROMPT_TEMPLATE` rewrite (keep overall structure, install steps, auth header, curl
   examples, and the "仅在用户明确要求时记录" rule intact):
   - skill frontmatter `description` and the 查询 section: broaden triggers from
     命令/脚本/服务器/运维 to 通用个人知识库 — any fact, decision, configuration, note,
     preference, or record the user may have stored before (technical AND non-technical);
     phrasing stays concise Chinese consistent with existing copy.
   - 记录 section: keep "search before write" guidance, and ADD the server-enforced dedup
     semantics per the 409 duplicate contract in Global Constraints: on HTTP 409 the response
     `detail.similar` lists close entries — merge into the closest via GET + PUT
     `/api/entry/<id>`; only resend with `"force": true` when it is genuinely a different topic.
   - Do not document endpoints that do not exist; do not change the `{KB_URL}`/`{TOKEN}`
     placeholder mechanism.
2. `AIClient.extract_meta` prompt: broaden tag examples from tool names only to mixed domains,
   e.g. `(如 docker、ffmpeg、运维、健康、旅行、决策)`; other rules unchanged.
3. `AIClient.ask` system prompt: replace the command-centric line
   `- 回答简洁、准确,直接给出命令或步骤` with copy that asks for accurate, directly usable
   information (命令、步骤、事实、结论均可); the source-citation and no-fabrication rules stay.
4. Do NOT modify `~/.claude/skills/kb/SKILL.md` (controller regenerates it after merge).
5. Tests (`tests/test_prompts.py`): `build_agent_prompt("https://x", "t")` substitutes both
   placeholders (no literal `{KB_URL}`/`{TOKEN}` remain); template mentions `409`, `force` and
   PUT-merge guidance; `extract_meta`/`ask` prompt literals no longer contain the removed
   command-centric phrases (assert on the source strings via importable constants or function
   inspection — keep it simple and non-brittle).

## Out of Scope (all tasks)

Backup/export, deployment files, async/threading refactor, `--raw` CLI flag, tags exact-match
filtering, Web UI changes, long-document chunking.
