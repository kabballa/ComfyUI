# pylint: disable=import-outside-toplevel,logging-fstring-interpolation
# Child process initialization for PyIsolate
import logging
import os

logger = logging.getLogger(__name__)


def is_child_process() -> bool:
    return os.environ.get("PYISOLATE_CHILD") == "1"


def initialize_child_process() -> None:
    # Manual RPC injection
    try:
        from pyisolate._internal.rpc_protocol import get_child_rpc_instance

        rpc = get_child_rpc_instance()
        if rpc:
            _setup_prompt_server_stub(rpc)
            _setup_utils_proxy(rpc)
        else:
            logger.warning("Could not get child RPC instance for manual injection")
            _setup_prompt_server_stub()
            _setup_utils_proxy()
    except Exception as e:
        logger.error(f"Manual RPC Injection failed: {e}")
        _setup_prompt_server_stub()
        _setup_utils_proxy()

    _setup_logging()


def _setup_prompt_server_stub(rpc=None) -> None:
    try:
        from .proxies.prompt_server_impl import PromptServerStub
        import sys
        import types

        # Mock server module
        if "server" not in sys.modules:
            mock_server = types.ModuleType("server")
            sys.modules["server"] = mock_server

        server = sys.modules["server"]

        if not hasattr(server, "PromptServer"):

            class MockPromptServer:
                pass

            server.PromptServer = MockPromptServer

        stub = PromptServerStub()

        if rpc:
            PromptServerStub.set_rpc(rpc)
            if hasattr(stub, "set_rpc"):
                stub.set_rpc(rpc)

        server.PromptServer.instance = stub

    except Exception as e:
        logger.error(f"Failed to setup PromptServerStub: {e}")


def _setup_utils_proxy(rpc=None) -> None:
    try:
        import comfy.utils
        import asyncio

        # Capture main loop during initialization (safe context)
        main_loop = None
        try:
            main_loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                main_loop = asyncio.get_event_loop()
            except RuntimeError:
                pass

        try:
            from .proxies.base import set_global_loop

            if main_loop:
                set_global_loop(main_loop)
        except ImportError:
            pass

        # Sync hook wrapper for progress updates
        def sync_hook_wrapper(
            value: int, total: int, preview: None = None, node_id: None = None
        ) -> None:
            if node_id is None:
                try:
                    from comfy_execution.utils import get_executing_context

                    ctx = get_executing_context()
                    if ctx:
                        node_id = ctx.node_id
                    else:
                        pass
                except Exception:
                    pass

            # Bypass blocked event loop by direct outbox injection
            if rpc:
                try:
                    # Use captured main loop if available (for threaded execution), or current loop
                    loop = main_loop
                    if loop is None:
                        loop = asyncio.get_event_loop()

                    rpc.outbox.put(
                        {
                            "kind": "call",
                            "object_id": "UtilsProxy",
                            "parent_call_id": None,  # We are root here usually
                            "calling_loop": loop,
                            "future": loop.create_future(),  # Dummy future
                            "method": "progress_bar_hook",
                            "args": (value, total, preview, node_id),
                            "kwargs": {},
                        }
                    )

                except Exception as e:
                    logging.getLogger(__name__).error(f"Manual Inject Failed: {e}")
            else:
                logging.getLogger(__name__).warning(
                    "No RPC instance available for progress update"
                )

        comfy.utils.PROGRESS_BAR_HOOK = sync_hook_wrapper

    except Exception as e:
        logger.error(f"Failed to setup UtilsProxy hook: {e}")


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.INFO)
