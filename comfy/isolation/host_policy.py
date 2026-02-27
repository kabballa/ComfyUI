# pylint: disable=logging-fstring-interpolation
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, TypedDict

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class HostSecurityPolicy(TypedDict):
    allow_network: bool
    writable_paths: List[str]
    readonly_paths: List[str]
    whitelist: Dict[str, str]


DEFAULT_POLICY: HostSecurityPolicy = {
    "allow_network": False,
    "writable_paths": ["/dev/shm", "/tmp"],
    "readonly_paths": [],
    "whitelist": {},
}


def _default_policy() -> HostSecurityPolicy:
    return {
        "allow_network": DEFAULT_POLICY["allow_network"],
        "writable_paths": list(DEFAULT_POLICY["writable_paths"]),
        "readonly_paths": list(DEFAULT_POLICY["readonly_paths"]),
        "whitelist": dict(DEFAULT_POLICY["whitelist"]),
    }


def load_host_policy(comfy_root: Path) -> HostSecurityPolicy:
    config_path = comfy_root / "pyproject.toml"
    policy = _default_policy()

    if not config_path.exists():
        logger.debug("Host policy file missing at %s, using defaults.", config_path)
        return policy

    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.warning(
            "Failed to parse host policy from %s, using defaults.",
            config_path,
            exc_info=True,
        )
        return policy

    tool_config = data.get("tool", {}).get("comfy", {}).get("host", {})
    if not isinstance(tool_config, dict):
        logger.debug("No [tool.comfy.host] section found, using defaults.")
        return policy

    if "allow_network" in tool_config:
        policy["allow_network"] = bool(tool_config["allow_network"])

    if "writable_paths" in tool_config:
        policy["writable_paths"] = [str(p) for p in tool_config["writable_paths"]]

    if "readonly_paths" in tool_config:
        policy["readonly_paths"] = [str(p) for p in tool_config["readonly_paths"]]

    whitelist_raw = tool_config.get("whitelist")
    if isinstance(whitelist_raw, dict):
        policy["whitelist"] = {str(k): str(v) for k, v in whitelist_raw.items()}

    logger.debug(
        f"Loaded Host Policy: {len(policy['whitelist'])} whitelisted nodes, Network={policy['allow_network']}"
    )
    return policy


__all__ = ["HostSecurityPolicy", "load_host_policy", "DEFAULT_POLICY"]
