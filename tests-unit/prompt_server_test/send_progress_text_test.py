"""Tests for send_progress_text routing and binary format logic.

These tests verify:
1. sid defaults to client_id (unicast) instead of None (broadcast)
2. Legacy binary format when prompt_id absent or client unsupported
3. New binary format with prompt_id when client supports the feature flag
"""

import struct

from comfy_api import feature_flags


# ---------------------------------------------------------------------------
# Helpers – replicate the packing logic so we can assert on the wire format
# ---------------------------------------------------------------------------


def _unpack_legacy(message: bytes):
    """Unpack a legacy progress_text binary message -> (node_id, text)."""
    offset = 0
    node_id_len = struct.unpack_from(">I", message, offset)[0]
    offset += 4
    node_id = message[offset : offset + node_id_len].decode("utf-8")
    offset += node_id_len
    text = message[offset:].decode("utf-8")
    return node_id, text


def _unpack_with_prompt_id(message: bytes):
    """Unpack new format -> (prompt_id, node_id, text)."""
    offset = 0
    prompt_id_len = struct.unpack_from(">I", message, offset)[0]
    offset += 4
    prompt_id = message[offset : offset + prompt_id_len].decode("utf-8")
    offset += prompt_id_len
    node_id_len = struct.unpack_from(">I", message, offset)[0]
    offset += 4
    node_id = message[offset : offset + node_id_len].decode("utf-8")
    offset += node_id_len
    text = message[offset:].decode("utf-8")
    return prompt_id, node_id, text


# ---------------------------------------------------------------------------
# Minimal stub that mirrors send_progress_text logic from server.py
# We can't import server.py directly (it pulls in torch via nodes.py),
# so we replicate the method body here. If the implementation changes,
# these tests should be updated in tandem.
# ---------------------------------------------------------------------------


class _StubServer:
    """Stub that captures send_sync calls and runs the real packing logic."""

    def __init__(self, client_id=None, sockets_metadata=None):
        self.client_id = client_id
        self.sockets_metadata = sockets_metadata or {}
        self.sent = []  # list of (event, data, sid)

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data, sid))

    def send_progress_text(self, text, node_id, prompt_id=None, sid=None):
        if isinstance(text, str):
            text = text.encode("utf-8")
        node_id_bytes = str(node_id).encode("utf-8")

        target_sid = sid if sid is not None else self.client_id

        if prompt_id and feature_flags.supports_feature(
            self.sockets_metadata, target_sid, "supports_progress_text_metadata"
        ):
            prompt_id_bytes = prompt_id.encode("utf-8")
            message = (
                struct.pack(">I", len(prompt_id_bytes))
                + prompt_id_bytes
                + struct.pack(">I", len(node_id_bytes))
                + node_id_bytes
                + text
            )
        else:
            message = struct.pack(">I", len(node_id_bytes)) + node_id_bytes + text

        self.send_sync(3, message, target_sid)  # 3 == BinaryEventTypes.TEXT


# ===========================================================================
# Routing tests
# ===========================================================================


class TestSendProgressTextRouting:
    """Verify sid resolution: defaults to client_id, overridable via sid param."""

    def test_defaults_to_client_id_when_sid_not_provided(self):
        server = _StubServer(client_id="active-client-123")
        server.send_progress_text("hello", "node1")

        _, _, sid = server.sent[0]
        assert sid == "active-client-123"

    def test_explicit_sid_overrides_client_id(self):
        server = _StubServer(client_id="active-client-123")
        server.send_progress_text("hello", "node1", sid="explicit-sid")

        _, _, sid = server.sent[0]
        assert sid == "explicit-sid"

    def test_broadcasts_when_no_client_id_and_no_sid(self):
        server = _StubServer(client_id=None)
        server.send_progress_text("hello", "node1")

        _, _, sid = server.sent[0]
        assert sid is None


# ===========================================================================
# Legacy format tests
# ===========================================================================


class TestSendProgressTextLegacyFormat:
    """Verify legacy binary format: [4B node_id_len][node_id][text]."""

    def test_legacy_format_no_prompt_id(self):
        server = _StubServer(client_id="c1")
        server.send_progress_text("some text", "node-42")

        _, data, _ = server.sent[0]
        node_id, text = _unpack_legacy(data)
        assert node_id == "node-42"
        assert text == "some text"

    def test_legacy_format_when_client_unsupported(self):
        server = _StubServer(
            client_id="c1",
            sockets_metadata={"c1": {"feature_flags": {}}},
        )
        server.send_progress_text("text", "node1", prompt_id="prompt-abc")

        _, data, _ = server.sent[0]
        node_id, text = _unpack_legacy(data)
        assert node_id == "node1"
        assert text == "text"

    def test_bytes_input_preserved(self):
        server = _StubServer(client_id="c1")
        server.send_progress_text(b"raw bytes", "node1")

        _, data, _ = server.sent[0]
        node_id, text = _unpack_legacy(data)
        assert text == "raw bytes"


# ===========================================================================
# New format tests
# ===========================================================================


class TestSendProgressTextNewFormat:
    """Verify new format: [4B prompt_id_len][prompt_id][4B node_id_len][node_id][text]."""

    def test_includes_prompt_id_when_supported(self):
        server = _StubServer(
            client_id="c1",
            sockets_metadata={
                "c1": {"feature_flags": {"supports_progress_text_metadata": True}}
            },
        )
        server.send_progress_text("progress!", "node-7", prompt_id="prompt-xyz")

        _, data, _ = server.sent[0]
        prompt_id, node_id, text = _unpack_with_prompt_id(data)
        assert prompt_id == "prompt-xyz"
        assert node_id == "node-7"
        assert text == "progress!"

    def test_new_format_with_explicit_sid(self):
        server = _StubServer(
            client_id=None,
            sockets_metadata={
                "my-sid": {"feature_flags": {"supports_progress_text_metadata": True}}
            },
        )
        server.send_progress_text("txt", "n1", prompt_id="p1", sid="my-sid")

        _, data, sid = server.sent[0]
        assert sid == "my-sid"
        prompt_id, node_id, text = _unpack_with_prompt_id(data)
        assert prompt_id == "p1"
        assert node_id == "n1"
        assert text == "txt"


# ===========================================================================
# Feature flag tests
# ===========================================================================


class TestProgressTextFeatureFlag:
    """Verify the supports_progress_text_metadata flag exists in server features."""

    def test_flag_in_server_features(self):
        features = feature_flags.get_server_features()
        assert "supports_progress_text_metadata" in features
        assert features["supports_progress_text_metadata"] is True
