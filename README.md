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
