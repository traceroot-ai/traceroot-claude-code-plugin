# traceroot-claude-code-plugin

Observability plugin for [Claude Code](https://docs.anthropic.com/en/docs/claude-code) that sends per-turn traces to [TraceRoot](https://traceroot.ai).

Each conversation turn becomes a trace in TraceRoot showing the LLM call, tool invocations, token usage, and any nested sub-agents — all grouped by session.

---

## Installation

### Prerequisite: Claude Code

Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code) and confirm `claude --version` works before continuing.

**Windows (PowerShell):**

```powershell
irm https://claude.ai/install.ps1 | iex
```

If `claude` is not recognized afterwards, add `%USERPROFILE%\.local\bin` to your user `PATH`
(see [Troubleshooting](#troubleshooting)), then fully restart your terminal or IDE.

### Plugin marketplace (recommended)

```bash
claude plugin marketplace add traceroot-ai/traceroot-claude-code-plugin
claude plugin install traceroot-observability@traceroot-claude-code-plugin
```

Or, from within Claude Code using the `/plugin` command: