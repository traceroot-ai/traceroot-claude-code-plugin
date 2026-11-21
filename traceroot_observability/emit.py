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
- Stop waits up to _SETTLE_TIMEOUT_S for the turn's final assistant rows,
  so that long is added to the time the hook keeps Claude Code waiting.
- Fail-open: missing api_key, session_id, or transcript_path → return quietly.
- emitted_ask_keys tracks which asks have been exported, by stable timestamp
  key. The transcript is re-parsed each Stop so a split read never loses a
  turn. Keys are recorded after the export succeeds. A flush still running
  at the 5s cap is recorded too, without waiting for that upload to return,
  because the server may already have the batch. A flush that returns false
  or raises is retried. An ask whose token counts cannot be parsed is
  recorded, and no spans from it are exported. Any other error while
  building that ask is left unrecorded, so the next Stop can retry it.
  A crash after a successful export can send the ask again.
  One flush covers every ask in the Stop. If a later batch in that flush
  fails, the asks are rebuilt, so a batch the server already accepted is
  sent again with new trace ids.

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

import sys
import threading
import time
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
        return _flush_client(traceroot)

    return tracer, flush


def _export_result_failed(result) -> bool:
    """True when an OpenTelemetry exporter reports FAILURE.

    HTTP 401 and other export errors come back as that result. They do not
    raise, and BatchSpanProcessor.force_flush still returns True after logging
    them, so the result itself is what says the batch was not accepted.
    """
    return getattr(result, "name", None) == "FAILURE"


def _flush_client(traceroot) -> bool:
    """Flush the live client and report whether the export was accepted."""
    client = traceroot.get_client()
    processor = getattr(client, "span_processor", None) if client is not None else None
    exporter = getattr(processor, "span_exporter", None) if processor is not None else None
    original = getattr(exporter, "export", None) if exporter is not None else None
    failed = {"value": False}

    def watching(spans):
        try:
            result = original(spans)
        except Exception:
            # The batch worker catches this and force_flush still returns True.
            failed["value"] = True
            raise
        if _export_result_failed(result):
            failed["value"] = True
        return result

    try:
        if exporter is not None and original is not None:
            exporter.export = watching
        if processor is not None:
            flushed = processor.force_flush()
        else:
            traceroot.flush()
            flushed = None
        traceroot.shutdown()
    except Exception:
        return False
    finally:
        if exporter is not None and original is not None:
            exporter.export = original
    if failed["value"] or flushed is False:
        return False
    return True


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

def _capped_flush(flush) -> bool:
    """Run flush() on a daemon thread, joining with a 5s cap.

    False from flush, or an exception, is a failed export and is retried.
    A thread still running at the cap is treated as sent. The SDK's HTTP
    timeout is longer than this cap, so the server may already be storing
    the batch, and retrying it writes a second trace. None and True do:
    the tests' fakes return None.
    """
    outcome = {"ok": True}

    def run():
        try:
            result = flush()
        except Exception:
            outcome["ok"] = False
            return
        if result is False:
            outcome["ok"] = False

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(5.0)
    if th.is_alive():
        # Still uploading at the cap: the server may already have it.
        # Retrying here duplicates traces.
        return True
    return outcome["ok"]


def _note_export_failure(session_id: str) -> None:
    """The ask stays unrecorded, so the next Stop tries the export again."""
    debug_log(f"Stop: export failed, session={session_id}; asks left unrecorded")
    try:
        print(
            "traceroot: export failed; this turn will be sent again on the next Stop",
            file=sys.stderr,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Transcript read
# ---------------------------------------------------------------------------

def _final_text_on_disk(records: list, want: str) -> bool:
    """Return True if the current turn's last assistant text row ends `want`."""
    for r in reversed(records):
        if r.get("type") == "user" and not r.get("isMeta") and not _t._is_tool_result(r):
            return False  # reached this turn's prompt: no assistant text on disk yet
        if r.get("type") == "assistant":
            text = _s._text(_t._content(r))
            if text.strip():
                # last_assistant_message is the final message's text blocks joined
                # with "\n", and each block is its own row: the last row is the
                # whole message or its last block. A bare suffix match would also
                # accept an earlier row that merely ends the same way.
                return text.strip() == want or want.endswith("\n" + text.rstrip())
    return False


_SETTLE_TIMEOUT_S = 2.0
_SETTLE_POLL_S = 0.1


def _read_settled(tpath: str, payload: dict) -> list:
    """Read the transcript, waiting briefly for the turn's final assistant rows.

    Claude Code can fire Stop before those rows are on disk. Emitting then drops
    the final answer for good, because the ask is marked emitted. The payload's
    last_assistant_message says what the last assistant text row must contain;
    without it (older Claude Code, SessionEnd) there is nothing to wait for.

    Call this with the session lock held. A SessionEnd that arrives during the
    wait must queue behind it; otherwise it reads the half-written transcript,
    emits the ask truncated and marks it emitted. The cost is that a concurrent
    SubagentStop can be delayed by up to _SETTLE_TIMEOUT_S. It loses nothing:
    the sub-agent transcript it copies stays on disk until its hook returns.
    """
    want = str(payload.get("last_assistant_message") or "").strip()
    deadline = time.monotonic() + _SETTLE_TIMEOUT_S
    while True:
        records = _t.read_all_records(tpath)
        if not want or _final_text_on_disk(records, want):
            return records
        if time.monotonic() >= deadline:
            # Give up: emit what is on disk, which may lack the final answer.
            debug_log(
                f"Stop: final assistant rows still missing after {_SETTLE_TIMEOUT_S:g}s, "
                f"emitting what is on disk: {tpath}"
            )
            return records
        time.sleep(_SETTLE_POLL_S)


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
    deterministic across re-parses, so the same ask keeps the same key.
    (Never id(), which differs each handle_stop.)
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
        json.dumps(content, default=str, sort_keys=True).encode("utf-8"),
        usedforsecurity=False,
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
        records = _read_settled(tpath, payload)
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

        # Nothing new to export: keep the git cache and the keys already sent.
        # New keys wait until the export is known to have succeeded. Saving them
        # first made a 401 or a raised tracer look like a finished send.
        if not newly_emitted_keys:
            _state.save_state(session_id, st)

        # The lock stays held through the flush. Releasing it first let a
        # second Stop select the same asks and export them twice.
        debug_log(
            f"Stop: {len(records)} records total, {len(turns)} complete turns,"
            f" {len(asks)} asks, {len(to_emit)} new complete asks, session={session_id}"
        )

        if not to_emit:
            return

        try:
            tracer, flush = _tracer(cfg, git)
            from .subagents import make_emitter
            emitter = make_emitter(session_id, tpath)
        except Exception:
            # Starting the tracer can fail for a reason that goes away, such as
            # a rejected key. Leave the asks unrecorded so the next Stop retries.
            _note_export_failure(session_id)
            return

        built_keys = []
        unbuildable = []
        for ask_turns, key in zip(to_emit, newly_emitted_keys):
            try:
                _s.build_ask_spans(
                    tracer,
                    ask_turns,
                    session_id,
                    git,
                    ["claude-code"],
                    cfg.max_chars,
                    subagent_emitter=emitter,
                )
            except (ValueError, TypeError):
                # A token count that is not a number will still be wrong on
                # the next Stop. Record the ask so it does not block the rest.
                debug_log(f"Stop: skip ask {key}; token usage cannot be parsed, session={session_id}")
                unbuildable.append(key)
                continue
            except Exception:
                # A plugin bug can be fixed. Leave the key out so a later Stop
                # rebuilds the ask.
                debug_log(f"Stop: ask {key} will be retried; span build failed, session={session_id}")
                try:
                    print(
                        "traceroot: span build failed; this turn will be tried again on the next Stop",
                        file=sys.stderr,
                    )
                except Exception:
                    pass
                continue
            built_keys.append(key)

        if unbuildable:
            st.emitted_ask_keys = sorted(set(st.emitted_ask_keys) | set(unbuildable))
            _state.save_state(session_id, st)

        if not built_keys:
            return

        try:
            exported = _capped_flush(flush)
        except Exception:
            _note_export_failure(session_id)
            return

        if not exported:
            _note_export_failure(session_id)
            return

        st.emitted_ask_keys = sorted(set(st.emitted_ask_keys) | set(newly_emitted_keys))
        repo, ref = git
        if repo:
            st.git_repo = repo
        if ref:
            st.git_ref = ref
        _state.save_state(session_id, st)
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
