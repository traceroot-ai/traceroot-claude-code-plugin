# traceroot-claude-code-plugin

Observability plugin for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that sends per-turn traces to [TraceRoot](https://traceroot.ai).

Each conversation turn becomes a trace in TraceRoot showing the LLM call, tool invocations, token usage, and any nested sub-agents — all grouped by session.

---

## Requirements

This plugin runs its hooks with [`uv`](https://docs.astral.sh/uv/), which fetches the tracing SDK on demand (no global install needed). You must have `uv` available on your `PATH`:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

`uv` provisions Python 3.11+ automatically, so no separate Python install is required.

> **Note:** Without `uv`, the plugin still installs but **silently captures nothing** (the hooks fail open with no traces and no error). If you've installed it and see no traces in TraceRoot, check that `uv` is on your `PATH` first.

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
