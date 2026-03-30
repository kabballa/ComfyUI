# pylint: disable=import-outside-toplevel,logging-fstring-interpolation
# Child process initialization for PyIsolate
import logging
import os

logger = logging.getLogger(__name__)


def is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


def initialize_child_process() -> None:
    _setup_child_loop_bridge()

    # Manual RPC injection
    try:
        from pyisolate._internal.rpc_protocol import get_child_rpc_instance

        rpc = get_child_rpc_instance()
        if rpc:
            _setup_proxy_callers(rpc)
        else:
            logger.warning("Could not get child RPC instance for manual injection")
            _setup_proxy_callers()
    except Exception as e:
        logger.error(f"Manual RPC Injection failed: {e}")
        _setup_proxy_callers()

    _setup_logging()


def _setup_child_loop_bridge() -> None:
    import asyncio

    main_loop = None
    try:
        main_loop = asyncio.get_running_loop()
    except RuntimeError:
        try:
            main_loop = asyncio.get_event_loop()
        except RuntimeError:
            pass

    if main_loop is None:
        return

    try:
        from .proxies.base import set_global_loop

        set_global_loop(main_loop)
    except ImportError:
        pass


def _setup_prompt_server_stub(rpc=None) -> None:
    try:
        from .proxies.prompt_server_impl import PromptServerStub

        if rpc:
            PromptServerStub.set_rpc(rpc)
        elif hasattr(PromptServerStub, "clear_rpc"):
            PromptServerStub.clear_rpc()
        else:
            PromptServerStub._rpc = None  # type: ignore[attr-defined]

    except Exception as e:
        logger.error(f"Failed to setup PromptServerStub: {e}")


def _setup_proxy_callers(rpc=None) -> None:
    try:
        from .proxies.folder_paths_proxy import FolderPathsProxy
        from .proxies.helper_proxies import HelperProxiesService
        from .proxies.model_management_proxy import ModelManagementProxy
        from .proxies.progress_proxy import ProgressProxy
        from .proxies.prompt_server_impl import PromptServerStub
        from .proxies.utils_proxy import UtilsProxy

        if rpc is None:
            FolderPathsProxy.clear_rpc()
            HelperProxiesService.clear_rpc()
            ModelManagementProxy.clear_rpc()
            ProgressProxy.clear_rpc()
            PromptServerStub.clear_rpc()
            UtilsProxy.clear_rpc()
            return

        FolderPathsProxy.set_rpc(rpc)
        HelperProxiesService.set_rpc(rpc)
        ModelManagementProxy.set_rpc(rpc)
        ProgressProxy.set_rpc(rpc)
        PromptServerStub.set_rpc(rpc)
        UtilsProxy.set_rpc(rpc)

    except Exception as e:
        logger.error(f"Failed to setup child singleton proxy callers: {e}")


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.INFO)
