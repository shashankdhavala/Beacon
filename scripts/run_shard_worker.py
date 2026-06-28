#!/usr/bin/env python3
"""
Run one shard worker process for the local TCP sharded inference simulation.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from executorch.extension.pybindings import portable_lib

from export_shards import make_fixed_attention_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a shard worker process")
    parser.add_argument("--artifact-dir", default="artifacts/gpt2")
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--downstream-host", default=None)
    parser.add_argument("--downstream-port", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        data = sock.recv(size - len(chunks))
        if not data:
            raise ConnectionError("socket closed while reading payload")
        chunks.extend(data)
    return bytes(chunks)


def send_message(sock: socket.socket, message: Dict[str, Any]) -> None:
    payload_buffer = io.BytesIO()
    torch.save(message, payload_buffer)
    payload = payload_buffer.getvalue()
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def receive_message(sock: socket.socket) -> Dict[str, Any]:
    header = read_exact(sock, 8)
    (payload_size,) = struct.unpack("!Q", header)
    payload = read_exact(sock, payload_size)
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)


class ShardWorker:
    def __init__(
        self,
        manifest: dict,
        shard_index: int,
        device: str,
        downstream_host: Optional[str],
        downstream_port: Optional[int],
    ):
        shard_info = manifest["shards"][shard_index]
        self.shard_index = shard_index
        self.max_cache_len = int(manifest["max_cache_len"])
        self.num_devices = int(manifest["num_devices"])
        self.is_last = shard_index == self.num_devices - 1
        self.num_layers = int(shard_info["num_layers"])
        self.num_heads = int(shard_info["num_heads"])
        self.head_dim = int(shard_info["head_dim"])
        self.cache_dtype = getattr(torch, shard_info["cache_dtype"])
        self.device = device
        artifact_path = Path(shard_info["artifact_path"])
        if not artifact_path.is_absolute():
            artifact_path = (Path(manifest["artifact_dir"]) / artifact_path).resolve()
        self.shard = portable_lib._load_for_executorch(str(artifact_path))
        self.kv_cache: tuple[torch.Tensor, ...] | None = None
        self.downstream_host = downstream_host
        self.downstream_port = downstream_port
        self.downstream_sock: Optional[socket.socket] = None

    def log(self, message: str) -> None:
        print(f"[shard_{self.shard_index}] {message}", flush=True)

    def connect_downstream(self) -> None:
        if self.is_last:
            return
        if not self.downstream_host or not self.downstream_port:
            raise ValueError(f"shard_{self.shard_index} requires downstream host/port")
        attempts = 0
        while True:
            try:
                sock = socket.create_connection((self.downstream_host, self.downstream_port), timeout=5)
                self.downstream_sock = sock
                self.log(f"connected downstream to {self.downstream_host}:{self.downstream_port}")
                return
            except OSError:
                attempts += 1
                if attempts >= 50:
                    raise
                time.sleep(0.2)

    def allocate_cache(self) -> tuple[torch.Tensor, ...]:
        if self.num_layers == 0:
            return ()
        tensors = []
        for _ in range(self.num_layers):
            tensors.append(
                torch.zeros(
                    1,
                    self.num_heads,
                    self.max_cache_len,
                    self.head_dim,
                    dtype=self.cache_dtype,
                    device=self.device,
                )
            )
            tensors.append(
                torch.zeros(
                    1,
                    self.num_heads,
                    self.max_cache_len,
                    self.head_dim,
                    dtype=self.cache_dtype,
                    device=self.device,
                )
            )
        return tuple(tensors)

    def start_request(self) -> None:
        self.kv_cache = self.allocate_cache()
        self.log("start_request: allocated local KV cache")
        if self.downstream_sock is not None:
            send_message(self.downstream_sock, {"type": "start_request"})
            receive_message(self.downstream_sock)

    def flush_request(self) -> None:
        self.kv_cache = None
        self.log("flush_request: cleared local KV cache")
        if self.downstream_sock is not None:
            send_message(self.downstream_sock, {"type": "flush_request"})
            receive_message(self.downstream_sock)

    def require_cache(self) -> tuple[torch.Tensor, ...]:
        if self.kv_cache is None:
            raise RuntimeError(f"shard_{self.shard_index} has no active request")
        return self.kv_cache

    def handle_step_hidden(self, message: Dict[str, Any]) -> Dict[str, Any]:
        hidden_states = message["hidden_states"].to(self.device)
        step = int(message["step"])
        current_length = int(message["current_length"])
        self.log(
            f"step_hidden: step={step} current_length={current_length} hidden_shape={tuple(hidden_states.shape)}"
        )
        attention_mask = make_fixed_attention_mask(current_length, self.max_cache_len, self.device)
        cache_position = torch.tensor([step], dtype=torch.long, device=self.device)
        cache = self.require_cache()
        outputs = self.shard.forward((hidden_states, attention_mask, cache_position, *cache))
        self.kv_cache = tuple(t.to(self.device) for t in outputs[1:])
        current = outputs[0].to(self.device)
        if self.is_last:
            self.log(f"produced hidden_states for master: shape={tuple(current.shape)}")
            return {"type": "step_result", "hidden_states": current.cpu()}

        self.log(f"forwarded hidden_states to downstream: shape={tuple(current.shape)}")
        send_message(
            self.downstream_sock,
            {
                "type": "step_hidden",
                "hidden_states": current.cpu(),
                "step": step,
                "current_length": current_length,
            },
        )
        return receive_message(self.downstream_sock)

    def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        message_type = message["type"]
        if message_type == "start_request":
            self.start_request()
            return {"type": "ack"}
        if message_type == "flush_request":
            self.flush_request()
            return {"type": "ack"}
        if message_type == "step_hidden":
            return self.handle_step_hidden(message)
        raise ValueError(f"Unknown message type: {message_type}")


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.artifact_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    worker = ShardWorker(
        manifest=manifest,
        shard_index=args.shard_index,
        device=args.device,
        downstream_host=args.downstream_host,
        downstream_port=args.downstream_port,
    )
    worker.connect_downstream()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    worker.log(f"listening on {args.host}:{args.port}")

    while True:
        conn, addr = server.accept()
        worker.log(f"accepted upstream connection from {addr}")
        with conn:
            while True:
                try:
                    message = receive_message(conn)
                except ConnectionError:
                    break
                response = worker.handle_message(message)
                send_message(conn, response)


if __name__ == "__main__":
    main()
