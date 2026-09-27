import os
from dataclasses import dataclass


def _env_opt(name: str) -> str:
    return os.environ.get(f"CLAUDE_PLUGIN_OPTION_{name}") or os.environ.get(name) or ""


@dataclass
class Config:
    api_key: str
    host_url: str
    max_chars: int


def load_config() -> Config:
    try:
        max_chars = int(_env_opt("TRACEROOT_PLUGIN_MAX_CHARS") or "20000")
        # A 0/negative budget disables truncation entirely (see middle_truncate),
        # so treat an out-of-range value the same as a non-integer: fall back.
        if max_chars < 1:
            max_chars = 20000
    except ValueError:
        max_chars = 20000
    return Config(
        api_key=_env_opt("TRACEROOT_API_KEY"),
        host_url=_env_opt("TRACEROOT_HOST_URL") or "https://app.traceroot.ai",
        max_chars=max_chars,
    )
