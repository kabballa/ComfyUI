# pylint: disable=import-outside-toplevel,logging-fstring-interpolation,protected-access,raise-missing-from,useless-return,wrong-import-position
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pyisolate.interfaces import IsolationAdapter, SerializerRegistryProtocol  # type: ignore[import-untyped]
from pyisolate._internal.rpc_protocol import AsyncRPC, ProxiedSingleton  # type: ignore[import-untyped]

try:
    from comfy.isolation.clip_proxy import CLIPProxy, CLIPRegistry
    from comfy.isolation.model_patcher_proxy import (
        ModelPatcherProxy,
        ModelPatcherRegistry,
    )
    from comfy.isolation.model_sampling_proxy import (
        ModelSamplingProxy,
        ModelSamplingRegistry,
    )
    from comfy.isolation.vae_proxy import VAEProxy, VAERegistry, FirstStageModelRegistry
    from comfy.isolation.proxies.folder_paths_proxy import FolderPathsProxy
    from comfy.isolation.proxies.model_management_proxy import ModelManagementProxy
    from comfy.isolation.proxies.prompt_server_impl import PromptServerService
    from comfy.isolation.proxies.utils_proxy import UtilsProxy
    from comfy.isolation.proxies.progress_proxy import ProgressProxy
except ImportError as exc:  # Fail loud if Comfy environment is incomplete
    raise ImportError(f"ComfyUI environment incomplete: {exc}")

logger = logging.getLogger(__name__)

# Force /dev/shm for shared memory (bwrap makes /tmp private)
import tempfile

if os.path.exists("/dev/shm"):
    # Only override if not already set or if default is not /dev/shm
    current_tmp = tempfile.gettempdir()
    if not current_tmp.startswith("/dev/shm"):
        logger.debug(
            f"Configuring shared memory: Changing TMPDIR from {current_tmp} to /dev/shm"
        )
        os.environ["TMPDIR"] = "/dev/shm"
        tempfile.tempdir = None  # Clear cache to force re-evaluation


class ComfyUIAdapter(IsolationAdapter):
    # ComfyUI-specific IsolationAdapter implementation

    @property
    def identifier(self) -> str:
        return "comfyui"

    def get_path_config(self, module_path: str) -> Optional[Dict[str, Any]]:
        if "ComfyUI" in module_path and "custom_nodes" in module_path:
            parts = module_path.split("ComfyUI")
            if len(parts) > 1:
                comfy_root = parts[0] + "ComfyUI"
                return {
                    "preferred_root": comfy_root,
                    "additional_paths": [
                        os.path.join(comfy_root, "custom_nodes"),
                        os.path.join(comfy_root, "comfy"),
                    ],
                }
        return None

    def setup_child_environment(self, snapshot: Dict[str, Any]) -> None:
        comfy_root = snapshot.get("preferred_root")
        if not comfy_root:
            return

        requirements_path = Path(comfy_root) / "requirements.txt"
        if requirements_path.exists():
            import re

            for line in requirements_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                pkg_name = re.split(r"[<>=!~\[]", line)[0].strip()
                if pkg_name:
                    logging.getLogger(pkg_name).setLevel(logging.ERROR)

    def register_serializers(self, registry: SerializerRegistryProtocol) -> None:
        def serialize_model_patcher(obj: Any) -> Dict[str, Any]:
            # Child-side: must already have _instance_id (proxy)
            if os.environ.get("PYISOLATE_CHILD") == "1":
                if hasattr(obj, "_instance_id"):
                    return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}
                raise RuntimeError(
                    f"ModelPatcher in child lacks _instance_id: "
                    f"{type(obj).__module__}.{type(obj).__name__}"
                )
            # Host-side: register with registry
            if hasattr(obj, "_instance_id"):
                return {"__type__": "ModelPatcherRef", "model_id": obj._instance_id}
            model_id = ModelPatcherRegistry().register(obj)
            return {"__type__": "ModelPatcherRef", "model_id": model_id}

        def deserialize_model_patcher(data: Any) -> Any:
            """Deserialize ModelPatcher refs; pass through already-materialized objects."""
            if isinstance(data, dict):
                return ModelPatcherProxy(
                    data["model_id"], registry=None, manage_lifecycle=False
                )
            return data

        def deserialize_model_patcher_ref(data: Dict[str, Any]) -> Any:
            """Context-aware ModelPatcherRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return ModelPatcherProxy(
                    data["model_id"], registry=None, manage_lifecycle=False
                )
            else:
                return ModelPatcherRegistry()._get_instance(data["model_id"])

        # Register ModelPatcher type for serialization
        registry.register(
            "ModelPatcher", serialize_model_patcher, deserialize_model_patcher
        )
        # Register ModelPatcherProxy type (already a proxy, just return ref)
        registry.register(
            "ModelPatcherProxy", serialize_model_patcher, deserialize_model_patcher
        )
        # Register ModelPatcherRef for deserialization (context-aware: host or child)
        registry.register("ModelPatcherRef", None, deserialize_model_patcher_ref)

        def serialize_clip(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "_instance_id"):
                return {"__type__": "CLIPRef", "clip_id": obj._instance_id}
            clip_id = CLIPRegistry().register(obj)
            return {"__type__": "CLIPRef", "clip_id": clip_id}

        def deserialize_clip(data: Any) -> Any:
            if isinstance(data, dict):
                return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)
            return data

        def deserialize_clip_ref(data: Dict[str, Any]) -> Any:
            """Context-aware CLIPRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return CLIPProxy(data["clip_id"], registry=None, manage_lifecycle=False)
            else:
                return CLIPRegistry()._get_instance(data["clip_id"])

        # Register CLIP type for serialization
        registry.register("CLIP", serialize_clip, deserialize_clip)
        # Register CLIPProxy type (already a proxy, just return ref)
        registry.register("CLIPProxy", serialize_clip, deserialize_clip)
        # Register CLIPRef for deserialization (context-aware: host or child)
        registry.register("CLIPRef", None, deserialize_clip_ref)

        def serialize_vae(obj: Any) -> Dict[str, Any]:
            if hasattr(obj, "_instance_id"):
                return {"__type__": "VAERef", "vae_id": obj._instance_id}
            vae_id = VAERegistry().register(obj)
            return {"__type__": "VAERef", "vae_id": vae_id}

        def deserialize_vae(data: Any) -> Any:
            if isinstance(data, dict):
                return VAEProxy(data["vae_id"])
            return data

        def deserialize_vae_ref(data: Dict[str, Any]) -> Any:
            """Context-aware VAERef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                # Child: create a proxy
                return VAEProxy(data["vae_id"])
            else:
                # Host: lookup real VAE from registry
                return VAERegistry()._get_instance(data["vae_id"])

        # Register VAE type for serialization
        registry.register("VAE", serialize_vae, deserialize_vae)
        # Register VAEProxy type (already a proxy, just return ref)
        registry.register("VAEProxy", serialize_vae, deserialize_vae)
        # Register VAERef for deserialization (context-aware: host or child)
        registry.register("VAERef", None, deserialize_vae_ref)

        # ModelSampling serialization - handles ModelSampling* types
        # copyreg removed - no pickle fallback allowed

        def serialize_model_sampling(obj: Any) -> Dict[str, Any]:
            # Child-side: must already have _instance_id (proxy)
            if os.environ.get("PYISOLATE_CHILD") == "1":
                if hasattr(obj, "_instance_id"):
                    return {"__type__": "ModelSamplingRef", "ms_id": obj._instance_id}
                raise RuntimeError(
                    f"ModelSampling in child lacks _instance_id: "
                    f"{type(obj).__module__}.{type(obj).__name__}"
                )
            # Host-side: register with ModelSamplingRegistry and return JSON-safe dict
            ms_id = ModelSamplingRegistry().register(obj)
            return {"__type__": "ModelSamplingRef", "ms_id": ms_id}

        def deserialize_model_sampling(data: Any) -> Any:
            """Deserialize ModelSampling refs; pass through already-materialized objects."""
            if isinstance(data, dict):
                return ModelSamplingProxy(data["ms_id"])
            return data

        def deserialize_model_sampling_ref(data: Dict[str, Any]) -> Any:
            """Context-aware ModelSamplingRef deserializer for both host and child."""
            is_child = os.environ.get("PYISOLATE_CHILD") == "1"
            if is_child:
                return ModelSamplingProxy(data["ms_id"])
            else:
                return ModelSamplingRegistry()._get_instance(data["ms_id"])

        # Register ModelSampling type and proxy
        registry.register(
            "ModelSamplingDiscrete",
            serialize_model_sampling,
            deserialize_model_sampling,
        )
        registry.register(
            "ModelSamplingContinuousEDM",
            serialize_model_sampling,
            deserialize_model_sampling,
        )
        registry.register(
            "ModelSamplingContinuousV",
            serialize_model_sampling,
            deserialize_model_sampling,
        )
        registry.register(
            "ModelSamplingProxy", serialize_model_sampling, deserialize_model_sampling
        )
        # Register ModelSamplingRef for deserialization (context-aware: host or child)
        registry.register("ModelSamplingRef", None, deserialize_model_sampling_ref)

        def serialize_cond(obj: Any) -> Dict[str, Any]:
            type_key = f"{type(obj).__module__}.{type(obj).__name__}"
            return {
                "__type__": type_key,
                "cond": obj.cond,
            }

        def deserialize_cond(data: Dict[str, Any]) -> Any:
            import importlib

            type_key = data["__type__"]
            module_name, class_name = type_key.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(data["cond"])

        def _serialize_public_state(obj: Any) -> Dict[str, Any]:
            state: Dict[str, Any] = {}
            for key, value in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                if callable(value):
                    continue
                state[key] = value
            return state

        def serialize_latent_format(obj: Any) -> Dict[str, Any]:
            type_key = f"{type(obj).__module__}.{type(obj).__name__}"
            return {
                "__type__": type_key,
                "state": _serialize_public_state(obj),
            }

        def deserialize_latent_format(data: Dict[str, Any]) -> Any:
            import importlib

            type_key = data["__type__"]
            module_name, class_name = type_key.rsplit(".", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            obj = cls()
            for key, value in data.get("state", {}).items():
                prop = getattr(type(obj), key, None)
                if isinstance(prop, property) and prop.fset is None:
                    continue
                setattr(obj, key, value)
            return obj

        import comfy.conds

        for cond_cls in vars(comfy.conds).values():
            if not isinstance(cond_cls, type):
                continue
            if not issubclass(cond_cls, comfy.conds.CONDRegular):
                continue
            type_key = f"{cond_cls.__module__}.{cond_cls.__name__}"
            registry.register(type_key, serialize_cond, deserialize_cond)
            registry.register(cond_cls.__name__, serialize_cond, deserialize_cond)

        import comfy.latent_formats

        for latent_cls in vars(comfy.latent_formats).values():
            if not isinstance(latent_cls, type):
                continue
            if not issubclass(latent_cls, comfy.latent_formats.LatentFormat):
                continue
            type_key = f"{latent_cls.__module__}.{latent_cls.__name__}"
            registry.register(
                type_key, serialize_latent_format, deserialize_latent_format
            )
            registry.register(
                latent_cls.__name__, serialize_latent_format, deserialize_latent_format
            )

        # V3 API: unwrap NodeOutput.args
        def deserialize_node_output(data: Any) -> Any:
            return getattr(data, "args", data)

        registry.register("NodeOutput", None, deserialize_node_output)

        # KSAMPLER serializer: stores sampler name instead of function object
        # sampler_function is a callable which gets filtered out by JSONSocketTransport
        def serialize_ksampler(obj: Any) -> Dict[str, Any]:
            func_name = obj.sampler_function.__name__
            # Map function name back to sampler name
            if func_name == "sample_unipc":
                sampler_name = "uni_pc"
            elif func_name == "sample_unipc_bh2":
                sampler_name = "uni_pc_bh2"
            elif func_name == "dpm_fast_function":
                sampler_name = "dpm_fast"
            elif func_name == "dpm_adaptive_function":
                sampler_name = "dpm_adaptive"
            elif func_name.startswith("sample_"):
                sampler_name = func_name[7:]  # Remove "sample_" prefix
            else:
                sampler_name = func_name
            return {
                "__type__": "KSAMPLER",
                "sampler_name": sampler_name,
                "extra_options": obj.extra_options,
                "inpaint_options": obj.inpaint_options,
            }

        def deserialize_ksampler(data: Dict[str, Any]) -> Any:
            import comfy.samplers

            return comfy.samplers.ksampler(
                data["sampler_name"],
                data.get("extra_options", {}),
                data.get("inpaint_options", {}),
            )

        registry.register("KSAMPLER", serialize_ksampler, deserialize_ksampler)

        from comfy.isolation.model_patcher_proxy_utils import register_hooks_serializers

        register_hooks_serializers(registry)

        # Generic Numpy Serializer
        def serialize_numpy(obj: Any) -> Any:
            import torch

            try:
                # Attempt zero-copy conversion to Tensor
                return torch.from_numpy(obj)
            except Exception:
                # Fallback for non-numeric arrays (strings, objects, mixes)
                return obj.tolist()

        registry.register("ndarray", serialize_numpy, None)

    def provide_rpc_services(self) -> List[type[ProxiedSingleton]]:
        return [
            PromptServerService,
            FolderPathsProxy,
            ModelManagementProxy,
            UtilsProxy,
            ProgressProxy,
            VAERegistry,
            CLIPRegistry,
            ModelPatcherRegistry,
            ModelSamplingRegistry,
            FirstStageModelRegistry,
        ]

    def handle_api_registration(self, api: ProxiedSingleton, rpc: AsyncRPC) -> None:
        # Resolve the real name whether it's an instance or the Singleton class itself
        api_name = api.__name__ if isinstance(api, type) else api.__class__.__name__

        if api_name == "FolderPathsProxy":
            import folder_paths

            # Replace module-level functions with proxy methods
            # This is aggressive but necessary for transparent proxying
            # Handle both instance and class cases
            instance = api() if isinstance(api, type) else api
            for name in dir(instance):
                if not name.startswith("_"):
                    setattr(folder_paths, name, getattr(instance, name))
            return

        if api_name == "ModelManagementProxy":
            import comfy.model_management

            instance = api() if isinstance(api, type) else api
            # Replace module-level functions with proxy methods
            for name in dir(instance):
                if not name.startswith("_"):
                    setattr(comfy.model_management, name, getattr(instance, name))
            return

        if api_name == "UtilsProxy":
            import comfy.utils

            # Static Injection of RPC mechanism to ensure Child can access it
            # independent of instance lifecycle.
            api.set_rpc(rpc)

            # Don't overwrite host hook (infinite recursion)
            return

        if api_name == "PromptServerProxy":
            # Defer heavy import to child context
            import server

            instance = api() if isinstance(api, type) else api
            proxy = (
                instance.instance
            )  # PromptServerProxy instance has .instance property returning self

            original_register_route = proxy.register_route

            def register_route_wrapper(
                method: str, path: str, handler: Callable[..., Any]
            ) -> None:
                callback_id = rpc.register_callback(handler)
                loop = getattr(rpc, "loop", None)
                if loop and loop.is_running():
                    import asyncio

                    asyncio.create_task(
                        original_register_route(
                            method, path, handler=callback_id, is_callback=True
                        )
                    )
                else:
                    original_register_route(
                        method, path, handler=callback_id, is_callback=True
                    )
                return None

            proxy.register_route = register_route_wrapper

            class RouteTableDefProxy:
                def __init__(self, proxy_instance: Any):
                    self.proxy = proxy_instance

                def get(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("GET", path, handler)
                        return handler

                    return decorator

                def post(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("POST", path, handler)
                        return handler

                    return decorator

                def patch(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("PATCH", path, handler)
                        return handler

                    return decorator

                def put(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("PUT", path, handler)
                        return handler

                    return decorator

                def delete(
                    self, path: str, **kwargs: Any
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
                        self.proxy.register_route("DELETE", path, handler)
                        return handler

                    return decorator

            proxy.routes = RouteTableDefProxy(proxy)

            if (
                hasattr(server, "PromptServer")
                and getattr(server.PromptServer, "instance", None) != proxy
            ):
                server.PromptServer.instance = proxy
