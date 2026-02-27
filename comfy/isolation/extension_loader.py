# pylint: disable=cyclic-import,import-outside-toplevel,redefined-outer-name
from __future__ import annotations

import logging
import os
import inspect
import sys
import types
import platform
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pyisolate
from pyisolate import ExtensionManager, ExtensionManagerConfig

from .extension_wrapper import ComfyNodeExtension
from .manifest_loader import is_cache_valid, load_from_cache, save_to_cache
from .host_policy import load_host_policy

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


async def _stop_extension_safe(
    extension: ComfyNodeExtension, extension_name: str
) -> None:
    try:
        stop_result = extension.stop()
        if inspect.isawaitable(stop_result):
            await stop_result
    except Exception:
        logger.debug("][ %s stop failed", extension_name, exc_info=True)


def _normalize_dependency_spec(dep: str, base_paths: list[Path]) -> str:
    req, sep, marker = dep.partition(";")
    req = req.strip()
    marker_suffix = f";{marker}" if sep else ""

    def _resolve_local_path(local_path: str) -> Path | None:
        for base in base_paths:
            candidate = (base / local_path).resolve()
            if candidate.exists():
                return candidate
        return None

    if req.startswith("./") or req.startswith("../"):
        resolved = _resolve_local_path(req)
        if resolved is not None:
            return f"{resolved}{marker_suffix}"

    if req.startswith("file://"):
        raw = req[len("file://") :]
        if raw.startswith("./") or raw.startswith("../"):
            resolved = _resolve_local_path(raw)
            if resolved is not None:
                return f"file://{resolved}{marker_suffix}"

    return dep


def get_enforcement_policy() -> Dict[str, bool]:
    return {
        "force_isolated": os.environ.get("PYISOLATE_ENFORCE_ISOLATED") == "1",
        "force_sandbox": os.environ.get("PYISOLATE_ENFORCE_SANDBOX") == "1",
    }


class ExtensionLoadError(RuntimeError):
    pass


def register_dummy_module(extension_name: str, node_dir: Path) -> None:
    normalized_name = extension_name.replace("-", "_").replace(".", "_")
    if normalized_name not in sys.modules:
        dummy_module = types.ModuleType(normalized_name)
        dummy_module.__file__ = str(node_dir / "__init__.py")
        dummy_module.__path__ = [str(node_dir)]
        dummy_module.__package__ = normalized_name
        sys.modules[normalized_name] = dummy_module


def _is_stale_node_cache(cached_data: Dict[str, Dict]) -> bool:
    for details in cached_data.values():
        if not isinstance(details, dict):
            return True
        if details.get("is_v3") and "schema_v1" not in details:
            return True
    return False


async def load_isolated_node(
    node_dir: Path,
    manifest_path: Path,
    logger: logging.Logger,
    build_stub_class: Callable[[str, Dict[str, object], ComfyNodeExtension], type],
    venv_root: Path,
    extension_managers: List[ExtensionManager],
) -> List[Tuple[str, str, type]]:
    try:
        with manifest_path.open("rb") as handle:
            manifest_data = tomllib.load(handle)
    except Exception as e:
        logger.warning(f"][ Failed to parse {manifest_path}: {e}")
        return []

    # Parse [tool.comfy.isolation]
    tool_config = manifest_data.get("tool", {}).get("comfy", {}).get("isolation", {})
    can_isolate = tool_config.get("can_isolate", False)
    share_torch = tool_config.get("share_torch", False)

    # Parse [project] dependencies
    project_config = manifest_data.get("project", {})
    dependencies = project_config.get("dependencies", [])
    if not isinstance(dependencies, list):
        dependencies = []

    # Get extension name (default to folder name if not in project.name)
    extension_name = project_config.get("name", node_dir.name)

    # LOGIC: Isolation Decision
    policy = get_enforcement_policy()
    isolated = can_isolate or policy["force_isolated"]

    if not isolated:
        return []

    logger.info(f"][ Loading isolated node: {extension_name}")

    import folder_paths

    base_paths = [Path(folder_paths.base_path), node_dir]
    dependencies = [
        _normalize_dependency_spec(dep, base_paths) if isinstance(dep, str) else dep
        for dep in dependencies
    ]

    manager_config = ExtensionManagerConfig(venv_root_path=str(venv_root))
    manager: ExtensionManager = pyisolate.ExtensionManager(
        ComfyNodeExtension, manager_config
    )
    extension_managers.append(manager)

    host_policy = load_host_policy(Path(folder_paths.base_path))

    sandbox_config = {}
    is_linux = platform.system() == "Linux"
    if is_linux and isolated:
        sandbox_config = {
            "network": host_policy["allow_network"],
            "writable_paths": host_policy["writable_paths"],
            "readonly_paths": host_policy["readonly_paths"],
        }
    share_cuda_ipc = share_torch and is_linux

    extension_config = {
        "name": extension_name,
        "module_path": str(node_dir),
        "isolated": True,
        "dependencies": dependencies,
        "share_torch": share_torch,
        "share_cuda_ipc": share_cuda_ipc,
        "sandbox": sandbox_config,
    }

    extension = manager.load_extension(extension_config)
    register_dummy_module(extension_name, node_dir)

    # Try cache first (lazy spawn)
    if is_cache_valid(node_dir, manifest_path, venv_root):
        cached_data = load_from_cache(node_dir, venv_root)
        if cached_data:
            if _is_stale_node_cache(cached_data):
                logger.debug(
                    "][ %s cache is stale/incompatible; rebuilding metadata",
                    extension_name,
                )
            else:
                logger.debug(f"][ {extension_name} loaded from cache")
                specs: List[Tuple[str, str, type]] = []
                for node_name, details in cached_data.items():
                    stub_cls = build_stub_class(node_name, details, extension)
                    specs.append(
                        (node_name, details.get("display_name", node_name), stub_cls)
                    )
                return specs

    # Cache miss - spawn process and get metadata
    logger.debug(f"][ {extension_name} cache miss, spawning process for metadata")

    try:
        remote_nodes: Dict[str, str] = await extension.list_nodes()
    except Exception as exc:
        logger.warning(
            "][ %s metadata discovery failed, skipping isolated load: %s",
            extension_name,
            exc,
        )
        await _stop_extension_safe(extension, extension_name)
        return []

    if not remote_nodes:
        logger.debug("][ %s exposed no isolated nodes; skipping", extension_name)
        await _stop_extension_safe(extension, extension_name)
        return []

    specs: List[Tuple[str, str, type]] = []
    cache_data: Dict[str, Dict] = {}

    for node_name, display_name in remote_nodes.items():
        try:
            details = await extension.get_node_details(node_name)
        except Exception as exc:
            logger.warning(
                "][ %s failed to load metadata for %s, skipping node: %s",
                extension_name,
                node_name,
                exc,
            )
            continue
        details["display_name"] = display_name
        cache_data[node_name] = details
        stub_cls = build_stub_class(node_name, details, extension)
        specs.append((node_name, display_name, stub_cls))

    if not specs:
        logger.warning(
            "][ %s produced no usable nodes after metadata scan; skipping",
            extension_name,
        )
        await _stop_extension_safe(extension, extension_name)
        return []

    # Save metadata to cache for future runs
    save_to_cache(node_dir, venv_root, cache_data, manifest_path)
    logger.debug(f"][ {extension_name} metadata cached")

    # EJECT: Kill process after getting metadata (will respawn on first execution)
    await _stop_extension_safe(extension, extension_name)

    return specs


__all__ = ["ExtensionLoadError", "register_dummy_module", "load_isolated_node"]
