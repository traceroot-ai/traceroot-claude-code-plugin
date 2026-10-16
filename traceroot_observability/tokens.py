"""
Token accounting for Claude Code transcript assistant messages.

Provides:
  llm_calls(assistant_msgs) -> list[LlmCall]

Token-counting rules (mirrors Claude Code's per-requestId streaming behavior):
  - Input and cache tokens are IDENTICAL on every transcript line that shares a
    requestId (the same API call streams multiple content-block lines). Count
    them ONCE per requestId (first sighting only).
  - Output tokens are reported CUMULATIVELY: early lines hold partials, the
    final line holds the true total. Track a running MAX per requestId and add
    only the delta when a larger value appears, so the total converges on the
    final (largest) value without double-counting intermediate partials.
  - prompt = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
    (cache folded into prompt total, also exposed separately as cache_read /
    cache_creation for downstream cost attribution).
"""

from dataclasses import dataclass, field


@dataclass
class LlmCall:
    model: str = "claude"
    prompt: int = 0
    completion: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    text: str = ""
    thinking: str = ""
    tool_calls: list = field(default_factory=list)
    start_ts: str | None = None
    end_ts: str | None = None


def _text(content) -> str:
    """Extract concatenated text from a content field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            x.get("text", "")
            for x in content
            if isinstance(x, dict) and x.get("type") == "text"
        )
    return ""


def _thinking(content) -> str:
    """Extract concatenated thinking blocks from a content field."""
    if isinstance(content, list):
        return "\n".join(
            x.get("thinking", "")
            for x in content
            if isinstance(x, dict) and x.get("type") == "thinking"
        )
    return ""


def _tool_calls(content) -> list:
    """Extract tool_use blocks from a content list as normalized dicts."""
    out = []
    if isinstance(content, list):
        for x in content:
            if isinstance(x, dict) and x.get("type") == "tool_use":
                out.append({"id": x.get("id"), "name": x.get("name"), "input": x.get("input")})
    return out


def _usage_int(value) -> int:
    """Return a token count as int, defaulting malformed values to zero."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def llm_calls(assistant_msgs: list[dict]) -> list[LlmCall]:
    """
    Collapse a flat list of assistant transcript messages into per-requestId LlmCall
    records with correct token accounting.

    Each requestId = one logical LLM API call. Multiple transcript lines may share
    a requestId because Claude Code writes one line per content block. The function
    groups them and applies:
      - input / cache: counted on the FIRST sighting of each requestId only.
      - output: running MAX (add only deltas), so streaming partials don't inflate totals.

    Returns calls in order of first requestId appearance.
    """
    order: list[str] = []
    by_rid: dict[str, LlmCall] = {}
    seen_input: set[str] = set()   # requestIds whose input/cache have been counted
    out_max: dict[str, int] = {}   # requestId -> highest output_tokens seen so far

    for m in assistant_msgs:
        msg = m.get("message") or {}
        rid = m.get("requestId") or msg.get("id") or f"noid:{len(order)}"

        if rid not in by_rid:
            by_rid[rid] = LlmCall(
                model=msg.get("model") or "claude",
                start_ts=m.get("timestamp"),
            )
            order.append(rid)

        call = by_rid[rid]
        call.end_ts = m.get("timestamp")

        # Accumulate text, thinking, and tool_calls across all frames for this requestId.
        txt = _text(msg.get("content"))
        if txt:
            call.text = (call.text + "\n" + txt) if call.text else txt
        thk = _thinking(msg.get("content"))
        if thk:
            call.thinking = (call.thinking + "\n" + thk) if call.thinking else thk
        # Attach frame timestamp to each tool_call so spans.py can use per-tool start times
        frame_ts = m.get("timestamp")
        new_tcs = _tool_calls(msg.get("content"))
        for tc in new_tcs:
            tc["ts"] = frame_ts
        call.tool_calls += new_tcs

        # Token accounting.
        u = msg.get("usage") or {}
        if rid not in seen_input:
            # First sighting: count input and cache tokens exactly once.
            inp = _usage_int(u.get("input_tokens"))
            cr = _usage_int(u.get("cache_read_input_tokens"))
            cc = _usage_int(u.get("cache_creation_input_tokens"))
            call.prompt += inp + cr + cc
            call.cache_read += cr
            call.cache_creation += cc
            seen_input.add(rid)

        # Output: add only the delta over the running max for this requestId.
        out = _usage_int(u.get("output_tokens"))
        prior = out_max.get(rid, 0)
        if out > prior:
            call.completion += out - prior
            out_max[rid] = out

    return [by_rid[r] for r in order]
