"""
Subagent snapshot + nested span emitter.

When Claude Code spawns a sub-agent (Task/Agent tool), it fires a SubagentStop
hook carrying the sub-agent's own transcript path.  That file is transient —
Claude Code may GC it once the hook returns — so we copy it into the parent
session's state directory immediately (handle_subagent_stop).  A mapping of
  parent_tool_use_id -> snapshot_path
is stored on SessionState.snapshots so that the parent-turn emitter can later
walk the sub-agent's LLM calls and emit them as nested spans under the Task/Agent
tool span, using the PARENT session.id (not the sub-agent's own session id).
This is the key invariant: all sub-agent LLM spans appear inside the parent
turn's trace, never as orphan traces.

Correlation design note
-----------------------
handle_subagent_stop extracts the parent tool_use_id via _parent_tool_use_id().
Real SubagentStop payloads may omit "tool_use_id" entirely.  When that field is
absent but "agent_transcript_path" is present, we fall back to reading the
sibling <transcript>.meta.json sidecar written by Claude Code, which contains
{"toolUseId": "toolu_..."}.  This sidecar is always present at SubagentStop time
because Claude Code writes it before firing the hook.

Disk resolution at Stop time (primary path)
-------------------------------------------
resolve_subagent_transcript() first checks state.snapshots (populated by
handle_subagent_stop when tool_use_id WAS present).  If absent, it derives
the subagents directory from the main transcript path and globs agent-*.meta.json
files to find the one whose toolUseId matches.  This is the production-reliable
path because the sidecar files are always on disk at Stop time.
"""

import glob
import json
import os
import shutil

from opentelemetry import trace

from . import state as _state
from . import spans as _s
from . import transcript


# ---------------------------------------------------------------------------
# Correlation helper (isolated for easy swap)
# ---------------------------------------------------------------------------

def _parent_tool_use_id(payload: dict) -> str | None:
    """
    Extract the parent Task/Agent tool_use_id from a SubagentStop payload.

    Primary: payload.get("tool_use_id") — present when Claude Code includes it.
    Fallback: read the sibling <agent_transcript_path>.meta.json (replace .jsonl
    with .meta.json) and return its "toolUseId" field.  Fail-open to None on
    any error (missing file, bad JSON, missing key).
    """
    tid = payload.get("tool_use_id") or None
    if tid:
        return tid

    src = payload.get("agent_transcript_path")
    if not src:
        return None

    meta_path = src.replace(".jsonl", ".meta.json")
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
        return meta.get("toolUseId") or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Disk-based transcript resolver
# ---------------------------------------------------------------------------

def resolve_subagent_transcript(
    session_id: str,
    tool_use_id: str,
    transcript_path: str,
) -> str | None:
    """
    Return the file path of the sub-agent transcript for the given tool_use_id.

    Resolution order:
    1. state.snapshots[tool_use_id] — set by handle_subagent_stop when the
       SubagentStop payload included tool_use_id directly.
    2. Disk glob — derive the subagents directory from the main transcript path
       (<dirname(transcript_path)>/<session_id>/subagents/), glob agent-*.meta.json,
       and return the sibling .jsonl whose meta "toolUseId" matches tool_use_id.

    Fail-open: returns None on any error (missing dir, bad JSON, no match).
    """
    # Primary: in-memory snapshot (written at SubagentStop time).
    st = _state.load_state(session_id)
    path = st.snapshots.get(tool_use_id)
    if path:
        return path

    # Fallback: disk scan of subagents directory.
    try:
        subagents_dir = os.path.join(
            os.path.dirname(transcript_path), session_id, "subagents"
        )
        for meta_file in glob.glob(os.path.join(subagents_dir, "agent-*.meta.json")):
            try:
                with open(meta_file) as fh:
                    meta = json.load(fh)
                if meta.get("toolUseId") == tool_use_id:
                    jsonl_path = meta_file.replace(".meta.json", ".jsonl")
                    if os.path.exists(jsonl_path):
                        return jsonl_path
            except Exception:
                continue
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Snapshot handler (called from emit.dispatch on SubagentStop)
# ---------------------------------------------------------------------------

def handle_subagent_stop(payload: dict) -> None:
    """
    Copy the sub-agent's transcript to the parent session's state dir and
    record the tool_use_id -> snapshot_path mapping.

    Fail-open: if the source path is missing/unreadable, or _parent_tool_use_id
    returns None, this function returns silently without raising.

    Must be called while the caller holds state.session_lock(session_id).
    """
    sid = payload.get("session_id") or ""
    src = payload.get("agent_transcript_path")
    tool_use_id = _parent_tool_use_id(payload)

    if not (sid and src and tool_use_id):
        return

    dst = _state._dir(sid) / f"subagent-{tool_use_id}.jsonl"
    try:
        shutil.copy(src, dst)
    except Exception:
        return

    st = _state.load_state(sid)
    st.snapshots[tool_use_id] = str(dst)
    _state.save_state(sid, st)


# ---------------------------------------------------------------------------
# Nested span emitter factory (called from emit.handle_stop)
# ---------------------------------------------------------------------------

def make_emitter(session_id: str, transcript_path: str = ""):
    """
    Return a callable that emits nested LLM spans from a sub-agent snapshot.

    The emitter resolves the sub-agent transcript via resolve_subagent_transcript(),
    which first checks state.snapshots then falls back to disk glob of meta.json
    sidecars in <dirname(transcript_path)>/<session_id>/subagents/.

    Signature:
        emitter(tracer, parent_ctx, tool_use_id, sess, max_chars) -> int | None

    Returns max_end_ns — the largest end-time (nanoseconds) of all sub-agent LLM
    spans emitted, or None if no spans were emitted.  The caller (build_turn_spans in
    spans.py) uses this to extend the parent TOOL span's end_time so that the
    async sub-agent work is contained within the Task/Agent span.

    Fail-open: missing snapshot, bad JSON, or empty assistant messages all
    result in a no-op returning None, never a raise.
    """

    def emit(
        tracer,
        parent_ctx,
        tool_use_id: str,
        sess: str,
        max_chars: int,
        parent_start_ns: int | None = None,
    ) -> int | None:
        path = resolve_subagent_transcript(session_id, tool_use_id, transcript_path)
        if not path:
            return None

        records = transcript.read_all_records(path)
        sub_turns = transcript.assemble_turns(records)
        if not sub_turns:
            return None

        max_end_ns: int | None = None

        for sub_turn in sub_turns:
            # v1: sub-sub-agents not recursed (subagent_emitter=None);
            # Task tool spans will appear but without nested internals.
            _, end = _s._build_turn_spans_into(
                tracer, parent_ctx, sub_turn, sess, max_chars,
                subagent_emitter=None,
                start_floor_ns=parent_start_ns,
            )
            if end is not None:
                if max_end_ns is None or end > max_end_ns:
                    max_end_ns = end

        return max_end_ns

    return emit
