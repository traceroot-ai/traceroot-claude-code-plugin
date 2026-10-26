import os
from dataclasses import dataclass

_TRUTHY = ("1", "true", "yes", "on")


def _env_opt(name: str) -> str:
    return os.environ.get(f"CLAUDE_PLUGIN_OPTION_{name}") or os.environ.get(name) or ""


def _capture_content_enabled() -> bool:
    """Default on (unset or empty). Any set, non-empty source that isn't a
    recognised truthy value turns capture off (fail closed). Both the plugin
    option and the plain env var are checked, so an opt-out from either wins."""
    name = "TRACEROOT_PLUGIN_CAPTURE_CONTENT"
    sources = (os.environ.get(f"CLAUDE_PLUGIN_OPTION_{name}"), os.environ.get(name))
    values = [v.strip().lower() for v in sources if v and v.strip()]
    if not values:
        return True
    return all(v in _TRUTHY for v in values)


@dataclass
class Config:
    api_key: str
    host_url: str
    max_chars: int
    capture_content: bool


def load_config() -> Config:
    try:
        max_chars = int(_env_opt("TRACEROOT_PLUGIN_MAX_CHARS") or "20000")
    except ValueError:
        max_chars = 20000
    return Config(
        api_key=_env_opt("TRACEROOT_API_KEY"),
        host_url=_env_opt("TRACEROOT_HOST_URL") or "https://app.traceroot.ai",
        max_chars=max_chars,
        capture_content=_capture_content_enabled(),
    )