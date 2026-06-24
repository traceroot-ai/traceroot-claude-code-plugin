"""
Stop hook orchestration: reads transcript, builds turns, groups into asks,
resolves git, emits one OTEL trace per new completed ask via traceroot-py.

Design constraints:
- No top-level `import traceroot` — traceroot is imported lazily inside
  _tracer() and _resolve_git() only. The smoke test runs under plain python3
  where traceroot may be absent; a top-level import would break it.
  The replay test (test_emit_replay.py) monkeypatches _tracer/_resolve_git
  so those lazy imports never execute during tests.
- Flush is capped at 5s on a daemon thread to avoid blocking Claude Code.
- Fail-open: missing api_key, session_id, or transcript_path → return quietly.
- emitted_ask_keys tracks which asks have been emitted by stable timestamp key;
  the whole transcript is re-parsed each Stop so cross-read-boundary splits
  never lose turns. Key-based tracking is out-of-order safe: asks complete
  in any order and each is emitted exactly once.

Ask grouping:
  A "task-notification" turn is one whose user text starts with <task-notification>.
  An "ask" = one human turn + any task-notification continuations.
  Notifications are id-matched to their launching ask via agent-<task-id>.meta.json
  → toolUseId → ask that issued that tool_use block. Falls back to positional
  attachment when subagents_dir is unavailable or the meta file is missing.
  Asks are emitted only when COMPLETE:
    - launched == 0 (no sub-agents → done at its Stop), OR
    - notifs >= launched (all launched sub-agents have reported), OR
    - is_session_end (flush everything)
"""

import threading
from .config import load_config
from . import state as _state
from . import transcript as _t
from . import spans as _s
from .debug import debug_log


# ---------------------------------------------------------------------------
# Lazy helpers (monkeypatched by tests)
# ---------------------------------------------------------------------------

def _tracer(cfg, git):
    """Initialise traceroot and return (tracer, flush_fn).

    Called at most once per handle_stop invocation.  The flush_fn runs
    traceroot.flush() + traceroot.shutdown() inside the capped thread.
    """
    import traceroot
    from opentelemetry import trace as _trace

    repo, ref = git
    traceroot.initialize(
        api_key=cfg.api_key,
        host_url=cfg.host_url,
        git_repo=repo,
        git_ref=ref,
    )
    tracer = _trace.get_tracer("traceroot.claude-code")

    def flush():
        try:
            traceroot.flush()
            traceroot.shutdown()
        except Exception:
            pass

    return tracer, flush


def _resolve_git(payload, st):
    """Return (repo, ref) for the current session.

    Priority:
    1. Values already on the SessionState (cached from a prior Stop).
    2. git_context_from_files(cwd) — reads .git/config + .git/HEAD directly,
       no git subprocess needed (works in containers without a git binary).

    Note on real API: git_context_from_files returns {"git_repo": ..., "git_ref": ...}
    (not "repo"/"ref" as the brief sketched). Keys are adapted here accordingly.
    """
    if st.git_repo or st.git_ref:
        return st.git_repo, st.git_ref

    from traceroot import git_context  # transport-only reuse; never the instrumentor
    try:
        cwd = payload.get("cwd") or "."
        ctx = git_context.git_context_from_files(cwd)
        st.git_repo = ctx.get("git_repo")
        st.git_ref = ctx.get("git_ref")
    except Exception:
        pass

    return st.git_repo, st.git_ref


# ---------------------------------------------------------------------------
# Flush cap
# ---------------------------------------------------------------------------

def _capped_flush(flush):
    """Run flush() on a daemon thread, joining with a 5s cap."""
    th = threading.Thread(target=flush, daemon=True)
    th.start()
    th.join(5.0)


# ---------------------------------------------------------------------------
# Ask completeness check
# ---------------------------------------------------------------------------

def _count_launched(ask_turns: list) -> int:
    """Count Task/Agent tool_use blocks across all turns in an ask."""
    count = 0
    for turn in ask_turns:
        for msg in turn.assistant_msgs:
            content = (msg.get("message") or {}).get("content") or []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("name") in ("Task", "Agent"):
                            count += 1
    return count


def _count_notifs(ask_turns: list) -> int:
    """Count task-notification turns in an ask."""
    return sum(1 for t in ask_turns if _t.is_task_notification(t))


def _ask_key(ask_turns: list) -> str:
    """
    Stable key for an ask: the first turn's user-message timestamp.

    Fallback (only if a record somehow lacks a timestamp) is a content hash —
    deterministic across re-parses, so the same ask keeps the same key and is
    never re-emitted. (Never id(), which differs each handle_stop.)
    """
    if not ask_turns:
        return "empty-ask"
    first = ask_turns[0]
    ts = first.user_msg.get("timestamp")
    if ts:
        return ts
    import hashlib
    import json
    content = (first.user_msg.get("message") or {}).get("content")
    digest = hashlib.sha1(
        json.dumps(content, default=str, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"ask-{digest}"


def _is_ask_complete(ask_turns: list, is_session_end: bool) -> bool:
    """Return True if ask is complete and safe to emit."""
    if is_session_end:
        return True
    launched = _count_launched(ask_turns)
    if launched == 0:
        return True  # no sub-agents → done at its Stop
    notifs = _count_notifs(ask_turns)
    return notifs >= launched


# ---------------------------------------------------------------------------
# Public hook handlers
# ---------------------------------------------------------------------------

def dispatch(payload: dict) -> None:
    event = payload.get("hook_event_name")
    if event == "Stop":
        handle_stop(payload)
    elif event == "SubagentStop":
        handle_subagent_stop(payload)
    elif event == "SessionEnd":
        handle_session_end(payload)


def handle_stop(payload: dict, is_session_end: bool = False) -> None:
    """Main Stop hook: emit one trace per new completed ask in the transcript."""
    cfg = load_config()
    if not cfg.api_key:
        return

    session_id = payload.get("session_id") or ""
    tpath = payload.get("transcript_path")
    if not session_id or not tpath:
        return

    # Derive subagents directory from transcript_path + session_id
    import os as _os
    subagents_dir = _os.path.join(_os.path.dirname(tpath), session_id, "subagents")

    with _state.session_lock(session_id):
        st = _state.load_state(session_id)
        records = _t.read_all_records(tpath)
        turns = _t.assemble_turns(records)
        asks = _t.group_turns_into_asks(turns, subagents_dir)
        git = _resolve_git(payload, st)

        # Determine which asks to emit using key-based tracking (out-of-order safe)
        emitted = set(st.emitted_ask_keys)
        to_emit = []
        newly_emitted_keys = []
        for ask in asks:
            key = _ask_key(ask)
            if key in emitted:
                continue
            if _is_ask_complete(ask, is_session_end):
                to_emit.append(ask)
                newly_emitted_keys.append(key)

        st.emitted_ask_keys = sorted(emitted | set(newly_emitted_keys))
        _state.save_state(session_id, st)

    debug_log(
        f"Stop: {len(records)} records total, {len(turns)} complete turns,"
        f" {len(asks)} asks, {len(to_emit)} new complete asks, session={session_id}"
    )

    if not to_emit:
        return

    tracer, flush = _tracer(cfg, git)

    from .subagents import make_emitter
    emitter = make_emitter(session_id, tpath)

    for ask_turns in to_emit:
        _s.build_ask_spans(
            tracer,
            ask_turns,
            session_id,
            git,
            ["claude-code"],
            cfg.max_chars,
            subagent_emitter=emitter,
        )

    _capped_flush(flush)
    debug_log(f"Stop: emitted {len(to_emit)} ask-traces, flush capped")


def handle_subagent_stop(payload: dict) -> None:
    """SubagentStop hook: snapshot the sub-agent transcript and record the mapping."""
    from .subagents import handle_subagent_stop as _h, _parent_tool_use_id
    sid = payload.get("session_id") or ""
    if not sid:
        return
    tool_use_id = _parent_tool_use_id(payload)
    agent_path = payload.get("agent_transcript_path")
    debug_log(
        f"SubagentStop: session={sid}"
        f" tool_use_id={tool_use_id}"
        f" agent_transcript_path={agent_path}"
    )
    with _state.session_lock(sid):
        st_before = _state.load_state(sid)
        snapshot_existed = tool_use_id in st_before.snapshots if tool_use_id else False
        _h(payload)
        st_after = _state.load_state(sid)
        snapshot_now = st_after.snapshots.get(tool_use_id) if tool_use_id else None
    debug_log(
        f"SubagentStop: snapshot_existed={snapshot_existed}"
        f" snapshot_path={snapshot_now}"
    )


def handle_session_end(payload: dict) -> None:
    """SessionEnd hook: flush any trailing ask not yet emitted by a Stop."""
    cfg = load_config()
    if not cfg.api_key:
        return
    handle_stop({**payload, "hook_event_name": "Stop"}, is_session_end=True)
