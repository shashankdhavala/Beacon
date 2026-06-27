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
import errno
import random
import socket
import struct
import sys
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
    parser.add_argument("--repeat", type=int, default=1, help="Number of request/response trials")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="Delay between repeated trials")
    parser.add_argument("--persistent", action="store_true", help="Reuse one TCP connection for repeated trials")
    args = parser.parse_args()

    shape = [1, args.seq_len, args.hidden_size]
    num_values = 1
    for dim in shape:
        num_values *= dim

    latencies_ms: list[float] = []
    response_sizes: list[int] = []

    sock: socket.socket | None = None
    if args.persistent:
        print(f"Opening persistent connection to Android worker at {args.host}:{args.port} ...")
        try:
            sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
            sock.settimeout(args.timeout)
        except TimeoutError:
            print_connection_help(args.host, args.port, "connection timed out")
            sys.exit(2)
        except OSError as error:
            reason = error.strerror or str(error)
            if error.errno == errno.ECONNREFUSED:
                reason = "connection refused"
            print_connection_help(args.host, args.port, reason)
            sys.exit(2)

    try:
        for trial in range(args.repeat):
            run_one_iteration(
                args=args,
                trial=trial,
                shape=shape,
                num_values=num_values,
                persistent_sock=sock,
                latencies_ms=latencies_ms,
                response_sizes=response_sizes,
            )
    finally:
        if sock is not None:
            sock.close()

    if len(latencies_ms) > 1:
        print()
        print("Summary:")
        print(f"trials={len(latencies_ms)}")
        print(f"mode={'persistent' if args.persistent else 'one_connection_per_trial'}")
        print(f"shape={shape} dtype={args.dtype} bytes_each_direction={response_sizes[0]}")
        print(f"min_ms={min(latencies_ms):.2f}")
        print(f"p50_ms={percentile(latencies_ms, 50):.2f}")
        print(f"p95_ms={percentile(latencies_ms, 95):.2f}")
        print(f"max_ms={max(latencies_ms):.2f}")


def run_one_iteration(
    args: argparse.Namespace,
    trial: int,
    shape: list[int],
    num_values: int,
    persistent_sock: socket.socket | None,
    latencies_ms: list[float],
    response_sizes: list[int],
) -> None:
        tensor_bytes = fake_fp16_tensor_bytes(num_values)
        request_id = f"mac-{int(time.time() * 1000)}-{trial}"
        msg = TensorMessage(
            message_type="TENSOR",
            request_id=request_id,
            step=trial,
            source_shard=0,
            target_shard=1,
            shape=shape,
            dtype=args.dtype,
            bytes_data=tensor_bytes,
        )

        if persistent_sock is None:
            print(f"Connecting to Android worker at {args.host}:{args.port} ...")
        try:
            response_header, response_body, elapsed_ms = run_trial(args, msg, persistent_sock)
        except TimeoutError:
            print_connection_help(args.host, args.port, "connection timed out")
            sys.exit(2)
        except OSError as error:
            reason = error.strerror or str(error)
            if error.errno == errno.ECONNREFUSED:
                reason = "connection refused"
            print_connection_help(args.host, args.port, reason)
            sys.exit(2)

        latencies_ms.append(elapsed_ms)
        response_sizes.append(len(response_body))
        print("Received response:")
        print(json.dumps(response_header, indent=2))
        print(f"response bytes={len(response_body)} sha256={sha256_hex(response_body)[:12]}")
        print(f"round_trip_ms={elapsed_ms:.2f}")
        if trial != args.repeat - 1 and args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)


def run_trial(
    args: argparse.Namespace,
    msg: TensorMessage,
    persistent_sock: socket.socket | None = None,
) -> tuple[dict[str, Any], bytes, float]:
    print(
        "Sending tensor "
        f"request={msg.request_id} shape={msg.shape} dtype={msg.dtype} "
        f"bytes={len(msg.bytes_data)} sha256={sha256_hex(msg.bytes_data)[:12]}"
    )
    if persistent_sock is not None:
        start = time.perf_counter()
        send_tensor(persistent_sock, msg)
        response_header, response_body = recv_tensor(persistent_sock)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return response_header, response_body, elapsed_ms

    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        start = time.perf_counter()
        send_tensor(sock, msg)
        response_header, response_body = recv_tensor(sock)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    return response_header, response_body, elapsed_ms


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile for empty values")
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def print_connection_help(host: str, port: int, reason: str) -> None:
    print(f"Could not connect to {host}:{port}: {reason}")
    print()
    print("Checklist:")
    print("- On Android, open Beacon Worker MVP and tap Start Worker Server.")
    print("- If using WiFi, use the app's wlan0 IP address, not rmnet/cellular IPs.")
    print(f"- Test WiFi reachability with: nc -vz {host} {port}")
    print("- If the phone is USB-connected, use ADB forwarding instead:")
    print(f"    adb forward tcp:{port} tcp:{port}")
    print(f"    python3 tools/mac_coordinator.py --host 127.0.0.1 --port {port}")


if __name__ == "__main__":
    main()
