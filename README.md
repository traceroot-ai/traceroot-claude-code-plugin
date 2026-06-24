# traceroot-claude-code-plugin

Observability plugin for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that sends per-turn traces to [TraceRoot](https://traceroot.ai).

Each conversation turn becomes a trace in TraceRoot showing the LLM call, tool invocations, token usage, and any nested sub-agents — all grouped by session.

---

## Installation

### Plugin marketplace (recommended)

```bash
claude plugin marketplace add traceroot-ai/traceroot-claude-code-plugin
claude plugin install traceroot-observability@traceroot-claude-code-plugin
```

Or, from within Claude Code using the `/plugin` command:

```
/plugin marketplace add traceroot-ai/traceroot-claude-code-plugin
```

### Manual installation

Clone the repo and register the hook directory with Claude Code:

```bash
git clone https://github.com/traceroot-ai/traceroot-claude-code-plugin.git ~/.claude/plugins/traceroot
# then add the hooks path to your Claude Code config
```

---

## Configuration

Set the following environment variables (or add them to your shell profile):

| Variable | Required | Description |
|---|---|---|
| `TRACEROOT_API_KEY` | Yes | Your TraceRoot API key. Find it at [app.traceroot.ai](https://app.traceroot.ai) → Settings → API Keys. |
| `TRACEROOT_HOST_URL` | No | TraceRoot ingest endpoint. Defaults to the hosted service. Set this only when self-hosting. |

Example (add to `~/.zshrc` or `~/.bashrc`):

```bash
export TRACEROOT_API_KEY="tr-..."
# Optional: only needed for self-hosted deployments
# export TRACEROOT_HOST_URL="https://your-traceroot.example.com"
```

Restart Claude Code (or open a new terminal session) after setting the variables.

---

## How traces are structured

### Turn = one trace

Every time you press Enter in Claude Code, the plugin captures the full conversation turn as a single trace in TraceRoot:

```
AGENT  "Claude Code Turn"      ← root span; covers the whole turn
  LLM  claude-opus-4-8         ← one span per LLM API call (with token counts)
  TOOL Read                    ← one span per tool use (with input args + result)
  TOOL Bash                    ← another tool call in the same turn
```

Token counts (prompt, completion, cache read, cache creation) appear on every LLM span.

### Sub-agents are nested inside the turn

When Claude Code spawns a sub-agent (a `Task` or `Agent` tool call), the plugin captures the sub-agent's own LLM calls and nests them under the parent `Task` TOOL span:

```
AGENT  "Claude Code Turn"
  LLM  claude-opus-4-8         ← parent's planning call
  TOOL Task                    ← the sub-agent invocation
    LLM  claude-haiku-4-5      ← sub-agent's LLM call (nested)
    LLM  claude-haiku-4-5      ← another sub-agent call
```

All nested sub-agent spans carry the **parent session's** `session.id`, so the full turn is always a single unified trace.

### Session grouping

All turns from the same Claude Code session share the same `session.id` attribute. In the TraceRoot UI you can filter by `session.id` to see the full conversation history as a connected sequence of traces.

### Git context

When the working directory has a git repository, each trace is annotated with `traceroot.git.repo` and `traceroot.git.ref` (current branch or HEAD). This is resolved by reading `.git/config` and `.git/HEAD` directly — no `git` binary is required, so it works in containers.

---

## Privacy

The plugin sends the following data to TraceRoot:

- **User prompts** (the text you type into Claude Code), truncated to the configured `max_chars` limit (default 20,000 characters; middle-truncated if longer).
- **Assistant responses** (text output only; thinking blocks are not captured).
- **Tool inputs and outputs** (command arguments, file paths, tool results), also truncated.
- **Token counts** (prompt, completion, cache tokens) for each LLM call.
- **Timestamps** backdated from the transcript.
- **Git repo URL and branch** from the working directory (if present).

The plugin does **not** send:

- File contents unless they appear as a tool result (e.g., `Read` output).
- System prompts or internal tool injection metadata (`isMeta` rows are skipped).
- Any data from sessions where `TRACEROOT_API_KEY` is not set (fail-open: missing key = no-op).

All data is sent over HTTPS to the TraceRoot ingest endpoint.

---

## Development

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git clone https://github.com/traceroot-ai/traceroot-claude-code-plugin.git
cd traceroot-claude-code-plugin
uv sync
```

### Run the tests

```bash
uv run pytest -v
```

Expected output: all tests pass (31 tests as of the current release).


---

## License

Apache 2.0. See [LICENSE](LICENSE).
