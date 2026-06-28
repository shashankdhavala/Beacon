#!/usr/bin/env python3
"""
MacBook coordinator for the Beacon/ClassMesh networking MVP.

Run this on the Mac. It connects to the Android worker TCP server,
sends one fake activation tensor or text route message, waits for a
response, and prints latency/checksum information.
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
    response_mode: str = "echo"
    checksum: str = "sha256"
    route: str = ""

    def header(self) -> dict[str, Any]:
        header = {
            "messageType": self.message_type,
            "requestId": self.request_id,
            "step": self.step,
            "sourceShard": self.source_shard,
            "targetShard": self.target_shard,
            "shape": self.shape,
            "dtype": self.dtype,
            "byteLength": len(self.bytes_data),
            "responseMode": self.response_mode,
            "route": self.route,
            "createdAtMs": int(time.time() * 1000),
        }
        header["sha256"] = sha256_hex(self.bytes_data) if self.checksum == "sha256" else ""
        return header


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
    parser = argparse.ArgumentParser(description="Send a fake tensor or text route message to Android workers.")
    parser.add_argument("--host", help="Android phone IP address shown in the worker app")
    parser.add_argument("--port", type=int, default=9000, help="Android worker port")
    parser.add_argument("--message", help="Send this UTF-8 text instead of a fake tensor, e.g. hello")
    parser.add_argument(
        "--tokenize-message",
        action="store_true",
        help="Split --message on whitespace and send each token through the worker chain",
    )
    parser.add_argument(
        "--route",
        help='Shard route, e.g. "1=10.0.0.11:9000,2=10.0.0.12:9000"',
    )
    parser.add_argument(
        "--tensor-route",
        action="store_true",
        help="Send random tensor bytes through --route and verify byte-for-byte equality",
    )
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "float16"])
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--repeat", type=int, default=1, help="Number of request/response trials")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="Delay between repeated trials")
    parser.add_argument("--persistent", action="store_true", help="Reuse one TCP connection for repeated trials")
    parser.add_argument("--response-mode", default="echo", choices=["echo", "ack"], help="Android response body mode")
    parser.add_argument("--checksum", default="sha256", choices=["sha256", "none"], help="Checksum mode")
    args = parser.parse_args()

    if args.tokenize_message:
        run_token_route(args)
        return

    if args.tensor_route:
        run_tensor_route(args)
        return

    if not args.host:
        raise SystemExit("--host is required unless --tokenize-message or --tensor-route is used with --route")

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
            configure_socket(sock)
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
        if args.message is None:
            print(f"shape={shape} dtype={args.dtype} bytes_each_direction={response_sizes[0]}")
        else:
            print(f"message={args.message!r} response_bytes={response_sizes[0]}")
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
    if args.message is not None:
        message_bytes = args.message.encode("utf-8")
        request_id = f"mac-text-{int(time.time() * 1000)}-{trial}"
        msg = TensorMessage(
            message_type="TEXT",
            request_id=request_id,
            step=trial,
            source_shard=0,
            target_shard=1,
            shape=[len(message_bytes)],
            dtype="utf8",
            bytes_data=message_bytes,
            response_mode=args.response_mode,
            checksum=args.checksum,
        )
    else:
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
            response_mode=args.response_mode,
            checksum=args.checksum,
        )

    if persistent_sock is None:
        print(f"Connecting to Android worker at {args.host}:{args.port} ...")
    try:
        response_header, response_body, elapsed_ms = run_trial(args, msg, persistent_sock)
    except TimeoutError:
        print_connection_help(first_host, first_port, "connection timed out")
        sys.exit(2)
    except OSError as error:
        reason = error.strerror or str(error)
        if error.errno == errno.ECONNREFUSED:
            reason = "connection refused"
        print_connection_help(first_host, first_port, reason)
        sys.exit(2)

    latencies_ms.append(elapsed_ms)
    response_sizes.append(len(response_body))
    print("Received response:")
    print(json.dumps(response_header, indent=2))
    print(f"response bytes={len(response_body)} sha256={sha256_hex(response_body)[:12]}")
    if response_header.get("dtype") == "utf8":
        print(f"response text={response_body.decode('utf-8')!r}")
    print(f"round_trip_ms={elapsed_ms:.2f}")
    if trial != args.repeat - 1 and args.delay_ms > 0:
        time.sleep(args.delay_ms / 1000.0)


def run_token_route(args: argparse.Namespace) -> None:
    try:
        result = execute_text_route(
            route_arg=args.route,
            message=args.message,
            checksum=args.checksum,
            timeout=args.timeout,
            delay_ms=args.delay_ms,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(result["output"])


def execute_text_route(
    route_arg: str | None,
    message: str | None,
    checksum: str = "none",
    timeout: float = 10.0,
    delay_ms: float = 0.0,
) -> dict[str, Any]:
    if not message:
        raise ValueError("message is required")
    if not route_arg:
        raise ValueError('text route requires a route, e.g. "1=10.0.0.11:9000,2=10.0.0.12:9000"')

    tokens = message.split()
    if not tokens:
        raise ValueError("message did not contain any whitespace-delimited tokens")

    route = parse_route(route_arg)
    if not route:
        raise ValueError("route did not contain any shard entries")
    first_shard, first_host, first_port = route[0]
    route_json = json.dumps(
        [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in route],
        separators=(",", ":"),
    )

    received: list[str] = []
    trials: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    with socket.create_connection((first_host, first_port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        configure_socket(sock)
        for step, token in enumerate(tokens):
            body = token.encode("utf-8")
            msg = TensorMessage(
                message_type="TEXT",
                request_id=f"mac-token-{int(time.time() * 1000)}-{step}",
                step=step,
                source_shard=0,
                target_shard=first_shard,
                shape=[len(body)],
                dtype="utf8",
                bytes_data=body,
                response_mode="echo",
                checksum=checksum,
                route=route_json,
            )
            start = time.perf_counter()
            send_tensor(sock, msg)
            header, response_body = recv_tensor(sock)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if header.get("dtype") != "utf8":
                raise ValueError(f"expected utf8 response, got {header.get('dtype')}")
            text = response_body.decode("utf-8")
            received.append(text)
            latencies_ms.append(elapsed_ms)
            trials.append(
                {
                    "step": step,
                    "token": token,
                    "received": text,
                    "requestBytes": len(body),
                    "responseBytes": len(response_body),
                    "latencyMs": elapsed_ms,
                    "requestId": msg.request_id,
                    "responseHeader": header,
                }
            )
            if step != len(tokens) - 1 and delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    return {
        "ok": True,
        "message": message,
        "tokens": tokens,
        "output": "".join(received),
        "route": route_arg,
        "routeHops": [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in route],
        "trials": trials,
        "summary": {
            "minMs": min(latencies_ms) if latencies_ms else 0.0,
            "p50Ms": percentile(latencies_ms, 50) if latencies_ms else 0.0,
            "p95Ms": percentile(latencies_ms, 95) if latencies_ms else 0.0,
            "maxMs": max(latencies_ms) if latencies_ms else 0.0,
        },
    }


def run_tensor_route(args: argparse.Namespace) -> None:
    result = execute_tensor_route(
        route_arg=args.route,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        repeat=args.repeat,
        dtype=args.dtype,
        checksum=args.checksum,
        timeout=args.timeout,
        delay_ms=args.delay_ms,
    )
    for trial in result["trials"]:
        print(
            f"{trial['status']} step={trial['step']} shape={result['shape']} "
            f"bytes={result['requestBytes']} returned={trial['returnedBytes']} "
            f"expected_sha={trial['expectedSha12']} actual_sha={trial['actualSha12']} "
            f"latency_ms={trial['latencyMs']:.2f}"
        )
    if len(result["trials"]) > 1:
        print()
        print("Summary:")
        print(f"trials={len(result['trials'])}")
        print(f"route={result['route']}")
        print(f"shape={result['shape']} dtype={result['dtype']} bytes={result['requestBytes']}")
        print(f"min_ms={result['summary']['minMs']:.2f}")
        print(f"p50_ms={result['summary']['p50Ms']:.2f}")
        print(f"p95_ms={result['summary']['p95Ms']:.2f}")
        print(f"max_ms={result['summary']['maxMs']:.2f}")


def execute_tensor_route(
    route_arg: str | None,
    seq_len: int,
    hidden_size: int,
    repeat: int,
    dtype: str = "fp16",
    checksum: str = "none",
    timeout: float = 10.0,
    delay_ms: float = 0.0,
) -> dict[str, Any]:
    if not route_arg:
        raise ValueError('tensor route requires a route, e.g. "1=10.0.0.11:9000,2=10.0.0.12:9000"')
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")

    route = parse_route(route_arg)
    if not route:
        raise ValueError("route did not contain any shard entries")
    first_shard, first_host, first_port = route[0]
    route_json = json.dumps(
        [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in route],
        separators=(",", ":"),
    )

    shape = [1, seq_len, hidden_size]
    num_values = 1
    for dim in shape:
        num_values *= dim

    trials: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    with socket.create_connection((first_host, first_port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        configure_socket(sock)
        for trial in range(repeat):
            tensor_bytes = fake_fp16_tensor_bytes(num_values)
            expected_sha = sha256_hex(tensor_bytes)
            msg = TensorMessage(
                message_type="TENSOR",
                request_id=f"mac-tensor-route-{int(time.time() * 1000)}-{trial}",
                step=trial,
                source_shard=0,
                target_shard=first_shard,
                shape=shape,
                dtype=dtype,
                bytes_data=tensor_bytes,
                response_mode="echo",
                checksum=checksum,
                route=route_json,
            )
            start = time.perf_counter()
            send_tensor(sock, msg)
            header, response_body = recv_tensor(sock)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            actual_sha = sha256_hex(response_body)
            bytes_match = response_body == tensor_bytes
            shape_match = header.get("shape") == shape
            dtype_match = header.get("dtype") == dtype
            status = "PASS" if bytes_match and shape_match and dtype_match else "FAIL"
            latencies_ms.append(elapsed_ms)
            trials.append(
                {
                    "status": status,
                    "step": trial,
                    "requestId": msg.request_id,
                    "returnedBytes": len(response_body),
                    "expectedSha": expected_sha,
                    "actualSha": actual_sha,
                    "expectedSha12": expected_sha[:12],
                    "actualSha12": actual_sha[:12],
                    "latencyMs": elapsed_ms,
                    "bytesMatch": bytes_match,
                    "shapeMatch": shape_match,
                    "dtypeMatch": dtype_match,
                    "responseHeader": header,
                }
            )
            if status != "PASS":
                break
            if trial != repeat - 1 and delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    summary = {
        "minMs": min(latencies_ms) if latencies_ms else 0.0,
        "p50Ms": percentile(latencies_ms, 50) if latencies_ms else 0.0,
        "p95Ms": percentile(latencies_ms, 95) if latencies_ms else 0.0,
        "maxMs": max(latencies_ms) if latencies_ms else 0.0,
    }
    return {
        "ok": all(trial["status"] == "PASS" for trial in trials),
        "route": route_arg,
        "routeHops": [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in route],
        "shape": shape,
        "dtype": dtype,
        "requestBytes": num_values * 2,
        "repeat": repeat,
        "trials": trials,
        "summary": summary,
    }


def parse_route(route_arg: str) -> list[tuple[int, str, int]]:
    route: list[tuple[int, str, int]] = []
    for raw_entry in route_arg.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"invalid route entry {entry!r}; expected shard=host:port")
        shard_text, address = entry.split("=", 1)
        try:
            shard_id = int(shard_text)
        except ValueError as error:
            raise ValueError(f"invalid shard id in route entry {entry!r}") from error
        if ":" in address:
            host, port_text = address.rsplit(":", 1)
            try:
                port = int(port_text)
            except ValueError as error:
                raise ValueError(f"invalid port in route entry {entry!r}") from error
        else:
            host = address
            port = 9000
        if not host:
            raise ValueError(f"missing host in route entry {entry!r}")
        route.append((shard_id, host, port))
    return route


def run_trial(
    args: argparse.Namespace,
    msg: TensorMessage,
    persistent_sock: socket.socket | None = None,
) -> tuple[dict[str, Any], bytes, float]:
    if msg.dtype == "utf8":
        print(
            "Sending text "
            f"request={msg.request_id} text={msg.bytes_data.decode('utf-8')!r} "
            f"bytes={len(msg.bytes_data)} response_mode={msg.response_mode} checksum={msg.checksum} "
            f"sha256={sha256_hex(msg.bytes_data)[:12]}"
        )
    else:
        print(
            "Sending tensor "
            f"request={msg.request_id} shape={msg.shape} dtype={msg.dtype} "
            f"bytes={len(msg.bytes_data)} response_mode={msg.response_mode} checksum={msg.checksum} "
            f"sha256={sha256_hex(msg.bytes_data)[:12]}"
        )
    if persistent_sock is not None:
        start = time.perf_counter()
        send_tensor(persistent_sock, msg)
        response_header, response_body = recv_tensor(persistent_sock)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return response_header, response_body, elapsed_ms

    with socket.create_connection((args.host, args.port), timeout=args.timeout) as sock:
        sock.settimeout(args.timeout)
        configure_socket(sock)
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


def configure_socket(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)


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
