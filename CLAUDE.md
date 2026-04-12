# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Hermes Context Manager (HMC) is a plugin for the Hermes agent gateway that automatically optimizes conversation context. It uses a **silent-first** architecture: most of the pipeline runs with no main-model involvement (pattern matching, code-aware compression, deduplication, truncation, error purging). Background compression via an auxiliary model handles the rest. The main model never sees compression tools or nudges.

HMC runs **in tandem** with Hermes's built-in `ContextCompressor` — it's a layered per-tool-call / per-turn silent compressor, not a replacement. Both run. The `ContextEngine` abstract base class in `agent.context_engine` is the wrong interface for this plugin: that's a mutually-exclusive single-select slot for engines that do one big summarization pass on token pressure. HMC's compression happens continuously at four hook points and cannot be expressed as a single `compress(messages)` callback. If a future contributor is tempted to migrate HMC to `ContextEngine`, don't.

## Commands

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run a single test file
python -m unittest tests/test_engine.py -v

# Run a single test case
python -m unittest tests.test_engine.TestMaterialize.test_dedup_identical_reads -v

# Install locally for development
hermes plugins install file:///absolute/path/to/hermes-context-manager
hermes gateway restart
```

No linter or formatter is configured. No Makefile.

## Architecture

### Two-tier pipeline

The plugin hooks into Hermes's lifecycle via four hooks (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `on_session_end`). Compression runs at three of the four hook points:

- **`post_tool_call`** — single-message strategies (short_circuit, code_filter, truncation) run in place on the new tool output as it lands, so agent loops between user turns stay compressed continuously. Publishes a `tool` SSE event.
- **`pre_llm_call`** — full `materialize_view` pipeline: all single-message strategies plus dedup and error_purge (which need the whole conversation). Triggers background compression if the context threshold is hit. Publishes a `turn` SSE event.
- **`on_session_end`** — runs a final materialize pass to catch tail-end tool outputs that never saw a `pre_llm_call` sweep, then restores the raw conversation so Hermes saves the unmutated session file. Writes the analytics row and publishes a `session_end` SSE event.

**Layer 0 — Silent strategies** (engine.py, zero model involvement). The pipeline in `materialize_view`:

1. **Short-circuit pattern matching** (`short_circuits.py`) — compresses known output shapes (JSON success, test results, git output) to one-liners. Errors are never short-circuited.
2. **Code-aware compression** (`code_filter.py`) — strips function/class bodies from tool outputs containing source code, preserving signatures, imports, and docstrings. Supports Python (indentation-aware), Rust, Go, JS, TS. String-aware brace counting handles `"hello {world}"` literals. JSX bailout for React safety.
3. **Head/tail truncation** (`truncation.py`) — keeps first/last lines of long outputs with a gap marker.
4. **Deduplication** — fingerprints tool inputs (`tool_name::sorted_args`), suppresses repeated reads of same content. `write_file` and `patch` are always protected. Second pass hashes normalized content.
5. **Error purging** — removes tool error outputs older than N turns.
6. **Tool-output pruning** — mutates pruned content to placeholders and credits savings to the `dedup` or `error_purge` bucket.
7. **Message ID snapshot** — builds an in-memory `{m001: timestamp, ...}` mapping for internal addressing. Does NOT mutate message content.

`apply_strategies_to_tool_output` wraps strategies 1-3 for the `post_tool_call` path; `materialize_view` runs all seven for `pre_llm_call` and `on_session_end`.

**Layer 1 — Background Compression** (`background_compressor.py`):
- Triggers when context usage > 80% (`max_context_percent`)
- Uses Hermes auxiliary model (not main model) for summarization
- Three-phase lock pattern: locked prepare → unlocked LLM call → locked commit
- Builds searchable index entries at `~/.hermes/hmc_state/{session}_index.jsonl`
- Deletes compressed messages from context

**Layer 2 — Passive Context** (`prompts.py`):
- One-liner system context injected via `pre_llm_call` return value
- Lists indexed phase topics when the session has any
- Does NOT advertise any tool the plugin doesn't register

### Persistent Analytics (`analytics.py`)

Cross-session, cross-project SQLite store at `~/.hermes/hmc_state/analytics.db`. One row per `(session, strategy)` is written at `on_session_end`. WAL mode + `synchronous=NORMAL` + 5s busy_timeout + on-write 90-day TTL. Queryable via the `hmc_control analytics` action with `scope`, `period`, and `limit` args.

### Opt-In Web Dashboard (`dashboard.py`)

Stdlib-only embedded HTTP+SSE server (no Flask, no FastAPI). Launch via `hmc_control dashboard action=start`. Binds to `127.0.0.1` only. Publishes live `turn` and `session_end` events from the plugin hooks to connected SSE clients. Default port 48800 with rotation to 48801+ on conflict. Single-page HTML dashboard with inline CSS/JS — no bundler, no external CDN.

### Key Data Flow

```
pre_tool_call → record input fingerprint in SessionState
post_tool_call → finalize ToolRecord → apply_strategies_to_tool_output on
                 the new tool message → update context stats → publish "tool"
                 event (heartbeat for agent-loop visibility)
pre_llm_call → restore previous mutations → materialize_view (all 6 strategies
                including dedup + error_purge) → maybe trigger background
                compression → append turn snapshot to ring buffer → publish
                "turn" event → persist state
on_session_end → restore previous mutations → final materialize_view pass
                 (catches tail-end tool outputs) → restore again so Hermes
                 saves raw content → write analytics row → publish
                 "session_end" event → save state to disk → cleanup
```

Single-message strategies (short_circuit, code_filter, truncation) run on BOTH `post_tool_call` and `pre_llm_call`. Full-list strategies (dedup, error_purge) require the whole conversation and run on `pre_llm_call` and `on_session_end`. All in-place mutations are tracked via `_active_mutations` and restored before the next `pre_llm_call` materialize pass and before `on_session_end`'s final save.

### Core Types

- **SessionState** (`state.py`) — per-session runtime: tool records, pruned IDs, token savings, turn counter, `counted_savings_ids: set[tuple[str, str]]` (per-(id, strategy) gate preventing double-count), `turn_history: deque` (bounded 50-turn observability ring buffer), `project_path`.
- **MaterializedView** (`engine.py`) — computed pruned conversation with message ID snapshot, total tokens, and a `content_backup` dict keyed by list-index for in-place mutation restoration.
- **ToolRecord** (`state.py`) — per-tool-call metadata: input fingerprint, error flag, turn index, token estimate.
- **HmcConfig** (`config.py`) — nested dataclasses loaded from `config.yaml` via a generic recursive `_hydrate_dataclass` helper. Blocks: `manual_mode`, `compress`, `strategies`, `truncation`, `short_circuits`, `background_compression`, `analytics`, `code_filter`. Adding a new config block requires only defining the dataclass and referencing it from `HmcConfig` — the hydrator walks `fields(cls)` + `get_type_hints(cls)` and fills everything automatically.
- **AnalyticsStore** (`analytics.py`) — SQLite wrapper with WAL mode, on-write retention, per-operation connections.
- **Dashboard** (`dashboard.py`) — HTTP/SSE lifecycle wrapper with port rotation.

### Plugin Lifecycle

`__init__.py` at the **repo root** provides the `register(ctx)` entry point for Hermes. It conditionally imports from either the Hermes package path or the local directory (for pytest). The global `_PLUGIN` instance of `HermesContextManagerPlugin` (`plugin.py`) registers the `hmc_control` tool and all four hooks. The package-level `hermes_context_manager/__init__.py` only exports the class for tests and standalone usage — it does NOT have a `register()` function.

### State & Persistence

- `persistence.py` — `JsonStateStore` saves/loads session state as JSON at `~/.hermes/hmc_state/`
- Atomic writes via `NamedTemporaryFile` + `os.replace`, with temp-file cleanup on exception
- `_active_mutations` dict tracks in-flight message content backups for restoration on session end
- All state mutations protected by `threading.RLock()` (reentrant — safe for nested hooks)

## Configuration

`config.yaml` lives in the **installed plugin directory** (not the repo root). The file is gitignored; on first load, the plugin copies `config.yaml.example` next to itself. Key settings documented in the README Configuration section.

## Conventions

- All hooks use broad `try/except` with `LOGGER.exception()` — plugin errors must never crash Hermes
- Token estimation: `estimate_message_tokens` JSON-serializes only the API-visible fields (`role`, `content`, `tool_calls`, `tool_call_id`, `name`) and divides by 4. HMC-internal metadata like `timestamp` never ships to the provider and is not counted.
- Message IDs are positional (`m001`, `m002`, ...) and stored in `state.message_id_snapshot` mapping to message timestamps. Never written into message content — the model would echo the format and break Hermes's stream display.
- Content normalization for dedup replaces timestamps, UUIDs, hex hashes with placeholders (`normalizer.py`)
- Per-(tool_call_id, strategy) savings gate — each strategy claims its incremental savings exactly once per call, ever
- Only stdlib + PyYAML as dependencies; Hermes agent APIs accessed via optional imports with graceful fallback
- Dashboard is stdlib-only (`http.server` + `queue` + `threading`); no third-party web framework
