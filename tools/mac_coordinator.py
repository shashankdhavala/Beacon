#!/usr/bin/env python3
"""
MacBook coordinator for the Beacon/ClassMesh networking MVP.

Run this on the Mac. It connects to the Android worker TCP server,
sends one fake activation tensor, waits for a response tensor, and
prints latency/checksum information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any


HEADER_LEN_STRUCT = struct.Struct(">I")


@dataclass
class TensorMessage:
    message_type: str
    request_id: str
    step: int
    source_shard: int
    target_shard: int
    shape: list[int]
    dtype: str
    bytes_data: bytes

    def header(self) -> dict[str, Any]:
        return {
            "messageType": self.message_type,
            "requestId": self.request_id,
            "step": self.step,
            "sourceShard": self.source_shard,
            "targetShard": self.target_shard,
            "shape": self.shape,
            "dtype": self.dtype,
            "byteLength": len(self.bytes_data),
            "sha256": sha256_hex(self.bytes_data),
            "createdAtMs": int(time.time() * 1000),
        }


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError(f"socket closed with {remaining} bytes left to read")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_tensor(sock: socket.socket, msg: TensorMessage) -> None:
    header_bytes = json.dumps(msg.header(), separators=(",", ":")).encode("utf-8")
    sock.sendall(HEADER_LEN_STRUCT.pack(len(header_bytes)))
    sock.sendall(header_bytes)
    sock.sendall(msg.bytes_data)


def recv_tensor(sock: socket.socket) -> tuple[dict[str, Any], bytes]:
    header_len = HEADER_LEN_STRUCT.unpack(read_exact(sock, HEADER_LEN_STRUCT.size))[0]
    header = json.loads(read_exact(sock, header_len).decode("utf-8"))
    body = read_exact(sock, int(header["byteLength"]))
    actual = sha256_hex(body)
    expected = header.get("sha256")
    if expected and actual != expected:
        raise ValueError(f"checksum mismatch: expected {expected}, got {actual}")
    return header, body


def fake_fp16_tensor_bytes(num_values: int) -> bytes:
    # We only need realistic binary volume for the networking MVP. These are
    # random 16-bit words, not semantically valid model activations.
    values = bytearray(num_values * 2)
    for i in range(0, len(values), 2):
        word = random.getrandbits(16)
        values[i] = word & 0xFF
        values[i + 1] = (word >> 8) & 0xFF
    return bytes(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a fake tensor to an Android worker.")
    parser.add_argument("--host", required=True, help="Android phone IP address shown in the worker app")
    parser.add_argument("--port", type=int, default=9000, help="Android worker port")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16"])
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    shape = [1, args.seq_len, args.hidden_size]
    num_values = 1
    for dim in shape:
        num_values *= dim

    tensor_bytes = fake_fp16_tensor_bytes(num_values)
    request_id = f"mac-{int(time.time() * 1000)}"
    msg = TensorMessage(
        message_type="TENSOR",
        request_id=request_id,
        step=0,
        source_shard=0,
        target_shard=1,
        shape=shape,
        dtype=args.dtype,
        bytes_data=tensor_bytes,
    )

    print(f"Connecting to Android worker at {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        print(
            "Sending tensor "
            f"request={request_id} shape={shape} dtype={args.dtype} "
            f"bytes={len(tensor_bytes)} sha256={sha256_hex(tensor_bytes)[:12]}"
        )
        start = time.perf_counter()
        send_tensor(sock, msg)
        response_header, response_body = recv_tensor(sock)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    print("Received response:")
    print(json.dumps(response_header, indent=2))
    print(f"response bytes={len(response_body)} sha256={sha256_hex(response_body)[:12]}")
    print(f"round_trip_ms={elapsed_ms:.2f}")


if __name__ == "__main__":
    main()
