import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot_observability import emit, state


def _transcript(tmp_path):
    tpath = tmp_path / "sess.jsonl"
    tpath.write_text("\n".join([
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": "hi"},
            "timestamp": "2026-06-22T00:00:00Z",
        }),
        json.dumps({
            "type": "assistant",
            "requestId": "r1",
            "timestamp": "2026-06-22T00:00:01Z",
            "message": {
                "id": "r1",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hello"}],
                "usage": {"input_tokens": 5, "output_tokens": 9},
            },
        }),
    ]) + "\n")
    return tpath


def _payload(tpath):
    return {
        "hook_event_name": "Stop",
        "session_id": "sess",
        "transcript_path": str(tpath),
    }


def _tracer_for(exporter, flush):
    def fake_tracer(cfg, git):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        return provider.get_tracer("traceroot.claude-code"), flush

    return fake_tracer


def test_a_failed_export_leaves_the_ask_unrecorded(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, lambda: False))

    emit.handle_stop(_payload(_transcript(tmp_path)))

    assert state.load_state("sess").emitted_ask_keys == []


def test_the_next_stop_sends_an_ask_whose_export_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    tpath = _transcript(tmp_path)
    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, lambda: False))
    emit.handle_stop(_payload(tpath))
    failed_spans = len(exporter.get_finished_spans())

    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, lambda: True))
    emit.handle_stop(_payload(tpath))

    assert state.load_state("sess").emitted_ask_keys == ["2026-06-22T00:00:00Z"]
    assert len(exporter.get_finished_spans()) > failed_spans


def test_a_tracer_that_raises_leaves_the_ask_unrecorded(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))

    def boom(cfg, git):
        raise RuntimeError("unauthorized")

    monkeypatch.setattr(emit, "_tracer", boom)
    emit.handle_stop(_payload(_transcript(tmp_path)))

    assert state.load_state("sess").emitted_ask_keys == []


def test_export_failure_result_is_recognized():
    class Failure:
        name = "FAILURE"

    class Success:
        name = "SUCCESS"

    assert emit._export_result_failed(Failure())
    assert not emit._export_result_failed(Success())
    assert not emit._export_result_failed(None)
