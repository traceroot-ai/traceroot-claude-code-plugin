import subprocess, sys, os, json
HOOK = os.path.join(os.path.dirname(__file__), "..", "hooks", "traceroot_hook.py")

def test_hook_exits_zero_on_empty_and_unknown():
    for payload in ["", json.dumps({"hook_event_name": "Nope"})]:
        p = subprocess.run([sys.executable, HOOK], input=payload, text=True, capture_output=True)
        assert p.returncode == 0
