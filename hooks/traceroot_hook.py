#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["traceroot>=0.1.11"]
# ///
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main() -> int:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return 0
    try:
        from traceroot_observability.emit import dispatch
        from traceroot_observability.debug import debug_log, debug_exc
        debug_log(
            f"hook fired: {payload.get('hook_event_name')}"
            f" session={payload.get('session_id')}"
        )
        dispatch(payload)
    except Exception:
        try:
            from traceroot_observability.debug import debug_exc
            debug_exc("dispatch failed")
        except Exception:
            pass
        pass  # fail-open: never block Claude Code
    return 0

if __name__ == "__main__":
    sys.exit(main())
