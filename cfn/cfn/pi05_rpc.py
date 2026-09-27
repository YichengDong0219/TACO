"""Local RPC protocol and RoboTwin-compatible client for a shared PI0.5 policy."""

from __future__ import annotations

import pickle
import socket
import struct
import threading
import uuid
from typing import Any

import numpy as np


_HEADER = struct.Struct("!Q")


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("PI0.5 policy server closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)))
    sock.sendall(data)


def receive_message(sock: socket.socket) -> Any:
    size = _HEADER.unpack(_receive_exact(sock, _HEADER.size))[0]
    return pickle.loads(_receive_exact(sock, size))  # noqa: S301 - loopback-only trusted protocol


class PI05RPCClient:
    def __init__(self, address: str, timeout: float = 600.0):
        host, port = address.rsplit(":", 1)
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        self.sock.settimeout(timeout)
        self.lock = threading.Lock()

    def request(self, operation: str, **payload):
        with self.lock:
            send_message(self.sock, {"operation": operation, **payload})
            response = receive_message(self.sock)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown PI0.5 policy server error"))
        return response.get("result")

    def health(self):
        return self.request("health")

    def load_cfn(self, checkpoint: str):
        return self.request("load_cfn", checkpoint=checkpoint)

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


class RemotePI05Model:
    """Implements the model interface expected by RoboTwin's PI0.5 deploy code."""

    def __init__(self, address: str, mode: str, noise_seed: int = 42):
        self.client = PI05RPCClient(address)
        self.client.health()
        self.mode = mode
        # Which candidate-noise pool the server scores. Only consulted in "taco"
        # mode; 42 matches the pool every result before seed selection used.
        self.noise_seed = noise_seed
        self.session_id = uuid.uuid4().hex
        self.pi0_step = 50
        self.instruction = None
        self.observation_window = None

    def set_language(self, instruction):
        self.instruction = instruction

    def update_observation_window(self, img_arr, state):
        self.observation_window = {
            "images": [np.asarray(image) for image in img_arr],
            "state": np.asarray(state),
        }

    def get_action(self):
        if self.observation_window is None or self.instruction is None:
            raise RuntimeError("Remote PI0.5 request is missing observation or instruction")
        return self.client.request(
            "action",
            mode=self.mode,
            session_id=self.session_id,
            noise_seed=self.noise_seed,
            instruction=self.instruction,
            images=self.observation_window["images"],
            state=self.observation_window["state"],
        )

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
