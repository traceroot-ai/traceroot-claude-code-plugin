import json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from traceroot_observability import emit, state, spans


def test_handle_stop_emits_one_trace_per_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    # transcript with one complete turn
    tpath = tmp_path / "sess.jsonl"
    tpath.write_text("\n".join([
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}, "timestamp": "2026-06-22T00:00:00Z"}),
        json.dumps({"type": "assistant", "requestId": "r1", "timestamp": "2026-06-22T00:00:01Z",
            "message": {"id": "r1", "model": "claude-opus-4-8", "content": [{"type": "text", "text": "hello"}],
                        "usage": {"input_tokens": 5, "output_tokens": 9}}}),
    ]) + "\n")
    exp = InMemorySpanExporter()
    def fake_tracer(cfg, git):
        prov = TracerProvider(); prov.add_span_processor(SimpleSpanProcessor(exp))
        return prov.get_tracer("traceroot.claude-code"), (lambda: None)
    monkeypatch.setattr(emit, "_tracer", fake_tracer)
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    emit.handle_stop({"hook_event_name": "Stop", "session_id": "sess",
                      "transcript_path": str(tpath)})
    kinds = sorted({s.attributes.get(spans.OI["KIND"]) for s in exp.get_finished_spans()})
    assert kinds == ["AGENT", "LLM"]
    # emitted_turns advanced; a second Stop with same file emits nothing new
    exp2_before = len(exp.get_finished_spans())
    emit.handle_stop({"hook_event_name": "Stop", "session_id": "sess", "transcript_path": str(tpath)})
    assert len(exp.get_finished_spans()) == exp2_before
