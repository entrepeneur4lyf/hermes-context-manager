---
name: hmc
description: Hermes Context Manager — automatic context optimization
---

# Hermes Context Manager (HMC)

HMC automatically optimizes your conversation context in the background.
You don't need to manage context yourself.

## What HMC Does Automatically

- **Short-circuits** success patterns (test results, file writes, git
  output) into one-liners
- **Code-aware compression** strips function and class bodies from tool
  outputs containing source code, keeping signatures, imports, and
  docstrings. Supports Python, Rust, Go, JavaScript, and TypeScript.
- **Head/tail truncation** keeps the first and last N lines of long
  outputs with a gap marker
- **Deduplicates** repeated tool calls with identical or similar outputs
- **Purges** old error outputs after they're no longer relevant
- **Background compression** summarizes old completed work via an
  auxiliary model when context pressure is high (never the main model)

## Available Actions (via `hmc_control` tool)

| Action          | Description |
|-----------------|-------------|
| `status`        | Compact one-liner overview: mode, context %, savings, tool count |
| `context`       | Detailed context window stats (tokens, window, percent, counts) |
| `stats`         | Token savings breakdown by strategy (short_circuit, truncation, code_filter, dedup, error_purge, background_compression) |
| `index`         | List indexed completed-work entries for the current session |
| `analytics`     | Cumulative savings from the SQLite analytics store. Supports `scope=global\|project`, `period=all\|day\|month\|recent\|project`, and `limit` |
| `dashboard`     | Live web dashboard lifecycle. `sub_action=start` launches a localhost HTTP+SSE server and returns the URL; `stop` shuts it down; `status` (default) reports whether it's running |
| `sweep`         | Manually prune tool outputs. Optional `count` limits how many; respects the protected list |
| `manual_set`    | Toggle manual mode via `enabled: bool`. In manual mode, automatic strategies are gated by `manual_mode.automatic_strategies` |
| `manual_status` | Report whether the plugin is in manual mode |

## Live Dashboard

For watching savings roll in as the model works:

```
/hmc dashboard action=start
→ {"running": true, "url": "http://127.0.0.1:48800/", ...}
```

Open the URL in a browser. The page streams turn, tool-call, and
session-end events via Server-Sent Events and shows:

- Current session: saved tokens, context %, turn count
- Lifetime totals from the SQLite analytics store
- Per-strategy bar breakdown
- Recent sessions ("workhorses") table with short ID, time,
  context %, tool count, savings, and per-strategy breakdown
- Live event log with per-tool heartbeats during agent loops

The server binds to `127.0.0.1` only and has no auth — it is not
designed for remote exposure.

## Historical Analytics

Token savings are persisted across sessions in a SQLite store at
`~/.hermes/hmc_state/analytics.db`. Query the cumulative totals at any
time:

```
/hmc analytics                        → global all-time summary
/hmc analytics scope=project          → current project only
/hmc analytics period=day limit=30    → last 30 days daily breakdown
/hmc analytics period=month           → last 12 calendar months
/hmc analytics period=recent limit=5  → most recent 5 sessions
/hmc analytics period=project         → top projects by savings
```
