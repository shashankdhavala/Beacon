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
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    extra_headers: dict[str, Any] = field(default_factory=dict)

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
        header.update(self.extra_headers)
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


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def load_manifest(artifact_dir: str) -> dict[str, Any]:
    manifest_path = Path(artifact_dir) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["manifest_dir"] = str(manifest_path.parent.resolve())
    return manifest


def route_to_json(route_arg: str) -> tuple[list[tuple[int, str, int]], str]:
    route = parse_route(route_arg)
    if not route:
        raise ValueError("route did not contain any shard entries")
    route_json = json.dumps(
        [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in route],
        separators=(",", ":"),
    )
    return route, route_json


def tensor_to_float32_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().to(dtype=torch.float32).cpu().contiguous().numpy().astype("<f4", copy=False)
    return array.tobytes(order="C")


def float32_bytes_to_tensor(body: bytes, shape: list[int], device: str) -> torch.Tensor:
    array = np.frombuffer(body, dtype="<f4").copy().reshape(shape)
    return torch.from_numpy(array).to(device)


def raise_for_worker_error(header: dict[str, Any], body: bytes) -> None:
    if header.get("messageType") != "ERROR":
        return
    try:
        detail = body.decode("utf-8")
    except UnicodeDecodeError:
        detail = repr(body)
    raise RuntimeError(detail)


def send_control(
    sock: socket.socket,
    route_json: str,
    first_shard: int,
    message_type: str,
    request_id: str,
    model_id: str,
    checksum: str,
) -> tuple[dict[str, Any], bytes, float]:
    msg = TensorMessage(
        message_type=message_type,
        request_id=request_id,
        step=0,
        source_shard=0,
        target_shard=first_shard,
        shape=[0],
        dtype="control",
        bytes_data=bytes(),
        response_mode="ack",
        checksum=checksum,
        route=route_json,
        extra_headers={"modelId": model_id, "currentLength": 0, "architecture": "llama"},
    )
    started = time.perf_counter()
    send_tensor(sock, msg)
    header, body = recv_tensor(sock)
    raise_for_worker_error(header, body)
    return header, body, (time.perf_counter() - started) * 1000.0


def tokenize_llama_prompt(
    tokenizer,
    prompt: str,
    chat_template: bool,
    system: str | None,
    device: str,
) -> torch.Tensor:
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    use_chat_template = chat_template and has_chat_template
    if not use_chat_template:
        return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a fake tensor or text route message to Android workers.")
    parser.add_argument("--llama-service", action="store_true", help="Run persistent Llama coordinator service mode.")
    parser.add_argument("--artifact-dir", default="artifacts/llama32_3b_sm8750_3way")
    parser.add_argument("--model-id", default=None, help="Override manifest model_id for Llama coordinator mode.")
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
    parser.add_argument("--llama-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--repeat", type=int, default=1, help="Number of request/response trials")
    parser.add_argument("--delay-ms", type=float, default=0.0, help="Delay between repeated trials")
    parser.add_argument("--persistent", action="store_true", help="Reuse one TCP connection for repeated trials")
    parser.add_argument("--response-mode", default="echo", choices=["echo", "ack"], help="Android response body mode")
    parser.add_argument("--checksum", default="sha256", choices=["sha256", "none"], help="Checksum mode")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--default-max-new-tokens", type=int, default=8)
    parser.add_argument("--no-chat-template", action="store_false", dest="chat_template")
    parser.add_argument("--system", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=9300)
    parser.set_defaults(chat_template=True)
    args = parser.parse_args()

    if args.llama_service:
        run_llama_service(args)
        return

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


class LlamaCoordinatorService:
    def __init__(self, args: argparse.Namespace):
        self.manifest = load_manifest(args.artifact_dir)
        architecture = self.manifest.get("architecture") or self.manifest.get("shards", [{}])[0].get("architecture")
        if architecture != "llama":
            raise ValueError(f"expected a llama manifest, got architecture={architecture!r}")
        if not args.route:
            raise ValueError("--route is required for --llama-service")

        self.route, self.route_json = route_to_json(args.route)
        expected_devices = int(self.manifest["num_devices"])
        if len(self.route) != expected_devices:
            raise ValueError(f"manifest expects {expected_devices} shards, got route with {len(self.route)}")

        self.model_id = args.model_id or self.manifest["model_id"]
        self.timeout = args.timeout
        self.checksum = args.checksum
        self.device = args.device
        self.dtype = args.llama_dtype
        self.default_max_new_tokens = args.default_max_new_tokens
        self.force_chat_template = args.chat_template
        self.default_system = args.system
        self.max_cache_len = int(self.manifest["max_cache_len"])
        self.first_shard, self.first_host, self.first_port = self.route[0]

        print(f"[mac-coordinator] loading tokenizer from {self.model_id}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print(f"[mac-coordinator] loading model from {self.model_id}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=parse_torch_dtype(self.dtype),
        ).to(self.device)
        self.model.eval()

        print(
            f"[mac-coordinator] connecting to shard_{self.first_shard} at {self.first_host}:{self.first_port}",
            flush=True,
        )
        self.sock = socket.create_connection((self.first_host, self.first_port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        configure_socket(self.sock)
        print("[mac-coordinator] ready", flush=True)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        system: str | None = None,
        chat_template: bool | None = None,
    ) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must be non-empty")
        token_budget = self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if token_budget <= 0:
            raise ValueError("max_new_tokens must be positive")

        prompt_ids = tokenize_llama_prompt(
            self.tokenizer,
            prompt,
            self.force_chat_template if chat_template is None else bool(chat_template),
            self.default_system if system is None else system,
            self.device,
        )
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len + token_budget > self.max_cache_len:
            raise ValueError(
                f"prompt tokens ({prompt_len}) + max_new_tokens ({token_budget}) exceeds "
                f"max_cache_len ({self.max_cache_len})"
            )

        request_prefix = f"mac-llama-{int(time.time() * 1000)}"
        generated_ids: list[int] = prompt_ids[0].detach().cpu().tolist()
        predicted_next: int | None = None
        trials: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        eos_token_id = self.tokenizer.eos_token_id

        print(
            f"[mac-coordinator] start_request prompt_tokens={prompt_len} max_new_tokens={token_budget} "
            f"mode=single_token_decode worker_kv_cache_expected=true",
            flush=True,
        )

        start_header, _, start_ms = send_control(
            sock=self.sock,
            route_json=self.route_json,
            first_shard=self.first_shard,
            message_type="START_REQUEST",
            request_id=f"{request_prefix}-start",
            model_id=self.model_id,
            checksum=self.checksum,
        )

        started_at = time.time()
        try:
            with torch.inference_mode():
                for step in range(prompt_len + token_budget):
                    if step < prompt_len:
                        token = prompt_ids[:, step : step + 1]
                        token_id = int(token.item())
                        phase = "prefill"
                    else:
                        if predicted_next is None:
                            raise RuntimeError("decode step reached before a predicted token was available")
                        token_id = predicted_next
                        token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                        phase = "decode"

                    print(
                        f"[mac-coordinator] step={step} phase={phase} send_tokens=1 current_length={step + 1} "
                        f"token_id={token_id}",
                        flush=True,
                    )

                    hidden = self.model.model.embed_tokens(token)
                    body = tensor_to_float32_bytes(hidden)
                    msg = TensorMessage(
                        message_type="STEP_HIDDEN",
                        request_id=f"{request_prefix}-step-{step}",
                        step=step,
                        source_shard=0,
                        target_shard=self.first_shard,
                        shape=list(hidden.shape),
                        dtype="float32",
                        bytes_data=body,
                        response_mode="echo",
                        checksum=self.checksum,
                        route=self.route_json,
                        extra_headers={
                            "modelId": self.model_id,
                            "architecture": "llama",
                            "currentLength": step + 1,
                        },
                    )

                    step_started = time.perf_counter()
                    send_tensor(self.sock, msg)
                    header, response_body = recv_tensor(self.sock)
                    raise_for_worker_error(header, response_body)
                    elapsed_ms = (time.perf_counter() - step_started) * 1000.0
                    latencies_ms.append(elapsed_ms)

                    if header.get("dtype") != "float32":
                        raise ValueError(f"expected float32 hidden state, got {header.get('dtype')}")
                    output_shape = [int(value) for value in header.get("shape", [])]
                    final_hidden = float32_bytes_to_tensor(response_body, output_shape, self.device)
                    logits = self.model.lm_head(self.model.model.norm(final_hidden))
                    predicted_next = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                    kept = step >= prompt_len - 1
                    print(
                        f"[mac-coordinator] step={step} recv_hidden_shape={output_shape} "
                        f"predicted_next={predicted_next} kept={kept}",
                        flush=True,
                    )
                    if kept:
                        generated_ids.append(predicted_next)
                    trials.append(
                        {
                            "step": step,
                            "phase": phase,
                            "inputTokenId": token_id,
                            "inputTokenText": self.tokenizer.decode([token_id], skip_special_tokens=False),
                            "predictedTokenId": predicted_next,
                            "predictedTokenText": self.tokenizer.decode([predicted_next], skip_special_tokens=False),
                            "kept": kept,
                            "requestBytes": len(body),
                            "responseBytes": len(response_body),
                            "latencyMs": elapsed_ms,
                        }
                    )
                    if phase == "decode" and eos_token_id is not None and predicted_next == int(eos_token_id):
                        break
                    if len(generated_ids) >= prompt_len + token_budget:
                        break
        finally:
            try:
                flush_header, _, flush_ms = send_control(
                    sock=self.sock,
                    route_json=self.route_json,
                    first_shard=self.first_shard,
                    message_type="FLUSH_REQUEST",
                    request_id=f"{request_prefix}-flush",
                    model_id=self.model_id,
                    checksum=self.checksum,
                )
            except Exception as exc:
                print(f"[mac-coordinator] flush_request failed: {exc}", flush=True)
                flush_header = {"error": str(exc)}
                flush_ms = 0.0

        output_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        duration_ms = int((time.time() - started_at) * 1000)
        return {
            "ok": True,
            "mode": "llama",
            "modelId": self.model_id,
            "prompt": prompt,
            "promptTokenIds": prompt_ids[0].detach().cpu().tolist(),
            "generatedTokenIds": generated_ids,
            "promptTokenCount": prompt_len,
            "generatedTokenCount": max(len(generated_ids) - prompt_len, 0),
            "output": output_text,
            "durationMs": duration_ms,
            "route": ",".join(f"{sid}={host}:{port}" for sid, host, port in self.route),
            "routeHops": [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in self.route],
            "startHeader": start_header,
            "flushHeader": flush_header,
            "startMs": start_ms,
            "flushMs": flush_ms,
            "trials": trials,
            "summary": {
                "minMs": min(latencies_ms) if latencies_ms else 0.0,
                "p50Ms": percentile(latencies_ms, 50) if latencies_ms else 0.0,
                "p95Ms": percentile(latencies_ms, 95) if latencies_ms else 0.0,
                "maxMs": max(latencies_ms) if latencies_ms else 0.0,
            },
        }


class LlamaRequestHandler(BaseHTTPRequestHandler):
    service: LlamaCoordinatorService

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(content_length)
        try:
            request = json.loads(payload.decode("utf-8"))
            result = self.service.generate(
                prompt=str(request.get("prompt", "")),
                max_new_tokens=request.get("max_new_tokens"),
                system=request.get("system"),
                chat_template=request.get("chat_template"),
            )
            print(
                f"[mac-coordinator] prompt_tokens={result['promptTokenCount']} "
                f"generated_tokens={result['generatedTokenCount']} duration_ms={result['durationMs']}",
                flush=True,
            )
            print(f"[mac-coordinator] output: {result['output']}", flush=True)
            self._write_json(HTTPStatus.OK, result)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        return


def run_llama_service(args: argparse.Namespace) -> None:
    service = LlamaCoordinatorService(args)
    if args.message:
        result = service.generate(
            prompt=args.message,
            max_new_tokens=args.max_new_tokens,
            system=args.system,
            chat_template=args.chat_template,
        )
        print(json.dumps(result, indent=2))
        return
    LlamaRequestHandler.service = service
    server = HTTPServer((args.listen_host, args.listen_port), LlamaRequestHandler)
    print(f"[mac-coordinator] listening on http://{args.listen_host}:{args.listen_port}", flush=True)
    server.serve_forever()


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
