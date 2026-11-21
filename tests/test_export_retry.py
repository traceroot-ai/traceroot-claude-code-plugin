import json
import threading

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


def _two_asks(tmp_path):
    """One ask the span builder cannot parse, then one it can."""
    rows = []
    for ts, text, usage in (
        ("2026-06-22T00:00:00Z", "bad", {"input_tokens": 1, "output_tokens": "unknown"}),
        ("2026-06-22T00:01:00Z", "good", {"input_tokens": 2, "output_tokens": 3}),
    ):
        rows.append(json.dumps({
            "type": "user",
            "message": {"role": "user", "content": text},
            "timestamp": ts,
        }))
        rows.append(json.dumps({
            "type": "assistant",
            "requestId": ts,
            "timestamp": ts.replace("00Z", "01Z"),
            "message": {
                "id": ts,
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": text}],
                "usage": usage,
            },
        }))
    tpath = tmp_path / "sess.jsonl"
    tpath.write_text("\n".join(rows) + "\n")
    return tpath


def test_a_build_error_does_not_block_later_asks(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    tpath = _two_asks(tmp_path)
    flushes = []

    def flush():
        flushes.append(1)
        return True

    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, flush))
    payload = _payload(tpath)
    emit.handle_stop(payload)
    after_first = list(state.load_state("sess").emitted_ask_keys)
    spans_after_first = len(exporter.get_finished_spans())

    emit.handle_stop(payload)

    assert after_first == ["2026-06-22T00:00:00Z", "2026-06-22T00:01:00Z"]
    assert spans_after_first > 0
    assert flushes == [1]
    assert state.load_state("sess").emitted_ask_keys == after_first


def test_a_build_error_still_retries_the_ask_whose_upload_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    tpath = _two_asks(tmp_path)
    flushes = []

    def flush():
        flushes.append(1)
        return len(flushes) > 1

    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, flush))
    payload = _payload(tpath)
    emit.handle_stop(payload)
    assert state.load_state("sess").emitted_ask_keys == ["2026-06-22T00:00:00Z"]

    emit.handle_stop(payload)
    assert flushes == [1, 1]
    assert state.load_state("sess").emitted_ask_keys == [
        "2026-06-22T00:00:00Z",
        "2026-06-22T00:01:00Z",
    ]


def _multi_turn_bad_usage(tmp_path):
    """A human turn, then a task-notification whose token count cannot be parsed."""
    notif = "<task-notification>\n<task-id>t1</task-id>\n</task-notification>"
    rows = [
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
        json.dumps({
            "type": "user",
            "message": {"role": "user", "content": notif},
            "timestamp": "2026-06-22T00:00:02Z",
        }),
        json.dumps({
            "type": "assistant",
            "requestId": "r2",
            "timestamp": "2026-06-22T00:00:03Z",
            "message": {
                "id": "r2",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "done"}],
                "usage": {"input_tokens": 1, "output_tokens": "unknown"},
            },
        }),
    ]
    tpath = tmp_path / "sess.jsonl"
    tpath.write_text("\n".join(rows) + "\n")
    return tpath


def test_a_later_turn_with_bad_usage_exports_no_spans(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    tpath = _multi_turn_bad_usage(tmp_path)
    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, lambda: True))

    emit.handle_stop(_payload(tpath))

    assert exporter.get_finished_spans() == ()
    assert state.load_state("sess").emitted_ask_keys == ["2026-06-22T00:00:00Z"]


def test_an_unexpected_build_error_is_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    tpath = _transcript(tmp_path)
    calls = []
    real = emit._s.build_ask_spans

    def build(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("plugin bug")
        return real(*args, **kwargs)

    monkeypatch.setattr(emit._s, "build_ask_spans", build)
    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, lambda: True))
    payload = _payload(tpath)
    emit.handle_stop(payload)
    assert state.load_state("sess").emitted_ask_keys == []

    emit.handle_stop(payload)
    assert state.load_state("sess").emitted_ask_keys == ["2026-06-22T00:00:00Z"]
    assert len(exporter.get_finished_spans()) > 0


def test_a_tracer_that_raises_leaves_the_ask_unrecorded(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))

    def boom(cfg, git):
        raise RuntimeError("unauthorized")

    monkeypatch.setattr(emit, "_tracer", boom)
    emit.handle_stop(_payload(_transcript(tmp_path)))

    assert state.load_state("sess").emitted_ask_keys == []


class _Result:
    def __init__(self, name):
        self.name = name


class _Exporter:
    def __init__(self, export):
        self.export = export


class _Processor:
    """Stand in for BatchSpanProcessor: a raised export still flushes as success."""

    def __init__(self, exporter):
        self.span_exporter = exporter

    def force_flush(self):
        try:
            self.span_exporter.export([])
        except Exception:
            return True
        return True


class _Client:
    def __init__(self, processor):
        self.span_processor = processor


class _Root:
    def __init__(self, client):
        self._client = client

    def get_client(self):
        return self._client

    def shutdown(self):
        return None


def _root(export):
    return _Root(_Client(_Processor(_Exporter(export))))


def test_a_failure_result_makes_flush_false():
    root = _root(lambda spans: _Result("FAILURE"))
    assert emit._flush_client(root) is False


def test_a_raising_export_makes_flush_false():
    def boom(spans):
        raise RuntimeError("export down")

    assert emit._flush_client(_root(boom)) is False


def test_a_raising_export_leaves_the_ask_unrecorded(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))

    def boom(spans):
        raise RuntimeError("export down")

    root = _root(boom)
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, lambda: emit._flush_client(root)))

    emit.handle_stop(_payload(_transcript(tmp_path)))

    assert state.load_state("sess").emitted_ask_keys == []


def test_overlapping_stops_export_the_ask_once(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    tpath = _transcript(tmp_path)
    started = threading.Event()
    release = threading.Event()
    calls = []

    def slow_flush():
        calls.append(1)
        started.set()
        assert release.wait(3)
        return True

    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, slow_flush))
    payload = _payload(tpath)
    errors = []

    def run():
        try:
            emit.handle_stop(payload)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=run)
    first.start()
    assert started.wait(3)
    second = threading.Thread(target=run)
    second.start()
    second.join(0.2)
    release.set()
    first.join(3)
    second.join(3)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert calls == [1]
    assert state.load_state("sess").emitted_ask_keys == ["2026-06-22T00:00:00Z"]


def test_a_slow_upload_is_not_sent_again(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_ROOT", tmp_path)
    monkeypatch.setenv("TRACEROOT_API_KEY", "test-key")
    monkeypatch.setattr(emit, "_resolve_git", lambda payload, st: ("o/r", "abc"))
    exporter = InMemorySpanExporter()
    tpath = _transcript(tmp_path)
    release = threading.Event()
    calls = []

    def slow_flush():
        calls.append(1)
        release.wait(30)
        return True

    monkeypatch.setattr(emit, "_tracer", _tracer_for(exporter, slow_flush))
    payload = _payload(tpath)
    emit.handle_stop(payload)
    emit.handle_stop(payload)
    release.set()

    assert calls == [1]
    assert state.load_state("sess").emitted_ask_keys == ["2026-06-22T00:00:00Z"]


def test_export_failure_result_is_recognized():
    class Failure:
        name = "FAILURE"

    class Success:
        name = "SUCCESS"

    assert emit._export_result_failed(Failure())
    assert not emit._export_result_failed(Success())
    assert not emit._export_result_failed(None)
