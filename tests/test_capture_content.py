"""Tests for the TRACEROOT_PLUGIN_CAPTURE_CONTENT opt-out."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from traceroot_observability import spans
from traceroot_observability.config import load_config
from traceroot_observability.transcript import Turn

SECRET = "SECRET-PROMPT-xyz"
CONTENT_KEYS = (spans.OI["INPUT"], spans.OI["OUTPUT"], "gen_ai.thinking")
ENV = "TRACEROOT_PLUGIN_CAPTURE_CONTENT"
OPT = "CLAUDE_PLUGIN_OPTION_TRACEROOT_PLUGIN_CAPTURE_CONTENT"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Isolate every test from the developer's shell / plugin options."""
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv(OPT, raising=False)


def _tracer():
    prov = TracerProvider()
    exp = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return prov.get_tracer("t"), exp


def _turn(is_error=False):
    return Turn(
        user_msg={"message": {"role": "user", "content": SECRET}, "timestamp": "2026-06-22T00:00:00Z"},
        assistant_msgs=[{"type": "assistant", "requestId": "r1", "timestamp": "2026-06-22T00:00:01Z",
            "message": {"id": "r1", "model": "claude-x",
                "content": [
                    {"type": "thinking", "thinking": SECRET},
                    {"type": "text", "text": SECRET},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": SECRET}},
                ],
                "usage": {"input_tokens": 2, "output_tokens": 10}}}],
        tool_results={"t1": {"content": SECRET, "is_error": is_error,
                             "timestamp": "2026-06-22T00:00:02Z"}})


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "FALSE"])
def test_config_false_values(monkeypatch, val):
    monkeypatch.setenv(ENV, val)
    assert load_config().capture_content is False


@pytest.mark.parametrize("val", ["disabled", "none", "n", "flase", "2", "enabled"])
def test_config_unrecognised_values_fail_closed(monkeypatch, val):
    monkeypatch.setenv(ENV, val)
    assert load_config().capture_content is False


@pytest.mark.parametrize("val", [None, "", "   ", "true", "1", "yes", "on", " TRUE "])
def test_config_default_true(monkeypatch, val):
    if val is not None:
        monkeypatch.setenv(ENV, val)
    assert load_config().capture_content is True


def test_plugin_option_true_does_not_override_env_false(monkeypatch):
    monkeypatch.setenv(OPT, "true")
    monkeypatch.setenv(ENV, "false")
    assert load_config().capture_content is False


def test_plugin_option_false_does_not_override_env_true(monkeypatch):
    monkeypatch.setenv(OPT, "false")
    monkeypatch.setenv(ENV, "true")
    assert load_config().capture_content is False


def test_plugin_option_alone_is_honoured(monkeypatch):
    monkeypatch.setenv(OPT, "false")
    assert load_config().capture_content is False
    monkeypatch.setenv(OPT, "true")
    assert load_config().capture_content is True


def test_content_captured_by_default():
    tracer, exp = _tracer()
    spans.build_turn_spans(tracer, _turn(), "s", ("", ""), [], 20000)
    assert any(SECRET in str(v) for sp in exp.get_finished_spans() for v in sp.attributes.values())


def _assert_clean(exp):
    finished = exp.get_finished_spans()
    assert finished
    for sp in finished:
        for k in CONTENT_KEYS:
            assert k not in sp.attributes
        assert SECRET not in str(dict(sp.attributes))
        if sp.status.description:
            assert SECRET not in sp.status.description
    kinds = {sp.attributes[spans.OI["KIND"]] for sp in finished}
    assert kinds == {"AGENT", "LLM", "TOOL"}
    llm = next(s for s in finished if s.attributes[spans.OI["KIND"]] == "LLM")
    assert llm.attributes[spans.OI["TOK_PROMPT"]] == 2   # metadata kept
    assert llm.attributes[spans.OI["TOK_COMPLETION"]] == 10


def test_turn_spans_scrubbed(monkeypatch):
    monkeypatch.setenv(ENV, "false")
    tracer, exp = _tracer()
    spans.build_turn_spans(tracer, _turn(is_error=True), "s", ("o/r", "main"), [], 20000)
    _assert_clean(exp)
    tool = next(s for s in exp.get_finished_spans() if s.attributes[spans.OI["KIND"]] == "TOOL")
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.status.description == "tool error"


def test_ask_spans_scrubbed(monkeypatch):
    monkeypatch.setenv(ENV, "false")
    tracer, exp = _tracer()
    spans.build_ask_spans(tracer, [_turn(is_error=True)], "s", ("o/r", "main"), [], 20000)
    _assert_clean(exp)


def test_spans_scrubbed_on_unrecognised_value(monkeypatch):
    monkeypatch.setenv(ENV, "flase")
    tracer, exp = _tracer()
    spans.build_turn_spans(tracer, _turn(), "s", ("o/r", "main"), [], 20000)
    _assert_clean(exp)