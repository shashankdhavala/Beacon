#!/usr/bin/env python3
"""
Mac coordinator for Llama shards served by native executor_bridge_worker.

The bridge worker runs beside executor_runner on each Android device and uses
raw input/output files internally. This coordinator speaks a small binary TCP
protocol to those workers and keeps model tokenization/final head on the Mac.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_coordinator import parse_route, percentile
from android_llama_coordinator import load_manifest, parse_torch_dtype, tokenize_prompt, tensor_to_float32_bytes


REQ_MAGIC = 0x42515731
RESP_MAGIC = 0x42515231
MSG_RESET = 1
MSG_STEP = 2
REQ_STRUCT = struct.Struct(">IIIII")
RESP_STRUCT = struct.Struct(">III")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Llama through native executor bridge workers.")
    parser.add_argument("--artifact-dir", default="artifacts/llama32_3b_sm8750_3way")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--route", required=True, help='e.g. "1=10.0.0.11:9100,2=10.0.0.12:9100,3=10.0.0.13:9100"')
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--default-max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--system", default=None)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=9301)
    return parser.parse_args()


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


def configure(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def send_bridge(sock: socket.socket, msg_type: int, step: int, current_length: int, body: bytes = b"") -> bytes:
    sock.sendall(REQ_STRUCT.pack(REQ_MAGIC, msg_type, step, current_length, len(body)))
    if body:
        sock.sendall(body)
    magic, status, byte_length = RESP_STRUCT.unpack(read_exact(sock, RESP_STRUCT.size))
    if magic != RESP_MAGIC:
        raise RuntimeError(f"bad response magic: 0x{magic:x}")
    response = read_exact(sock, byte_length)
    if status != 0:
        detail = response.decode("utf-8", errors="replace")
        raise RuntimeError(f"worker error status={status}: {detail}")
    return response


def float32_bytes_to_tensor(body: bytes, device: str, hidden_size: int) -> torch.Tensor:
    expected = hidden_size * 4
    if len(body) != expected:
        raise ValueError(f"expected {expected} hidden bytes, got {len(body)}")
    array = np.frombuffer(body, dtype="<f4").copy().reshape([1, 1, hidden_size])
    return torch.from_numpy(array).to(device)


class NativeExecutorLlamaCoordinatorService:
    def __init__(
        self,
        route_arg: str,
        artifact_dir: str,
        model_id: str | None,
        timeout: float,
        device: str,
        dtype: str,
        default_max_new_tokens: int,
        chat_template: bool,
        system: str | None,
    ) -> None:
        self.manifest = load_manifest(artifact_dir)
        architecture = self.manifest.get("architecture") or self.manifest.get("shards", [{}])[0].get("architecture")
        if architecture != "llama":
            raise ValueError(f"expected llama manifest, got architecture={architecture!r}")

        self.route = parse_route(route_arg)
        if len(self.route) != int(self.manifest["num_devices"]):
            raise ValueError(f"manifest expects {self.manifest['num_devices']} shards, got route with {len(self.route)}")

        self.hidden_size = int(self.manifest.get("hidden_size", 3072))
        self.model_id = model_id or self.manifest["model_id"]
        self.timeout = timeout
        self.device = device
        self.dtype = dtype
        self.default_max_new_tokens = default_max_new_tokens
        self.chat_template = chat_template
        self.system = system
        self.max_cache_len = int(self.manifest["max_cache_len"])

        print(f"[native-coordinator] loading tokenizer from {self.model_id}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        print(f"[native-coordinator] loading model from {self.model_id}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=parse_torch_dtype(dtype)).to(device)
        self.model.eval()

        self.sockets: list[tuple[int, socket.socket]] = []
        for shard_id, host, port in self.route:
            print(f"[native-coordinator] connecting to shard_{shard_id} at {host}:{port}", flush=True)
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.settimeout(timeout)
            configure(sock)
            send_bridge(sock, MSG_RESET, 0, 0)
            self.sockets.append((shard_id, sock))
        print("[native-coordinator] ready", flush=True)

    def close(self) -> None:
        for _, sock in self.sockets:
            sock.close()
        self.sockets = []

    def reset_all(self) -> None:
        for _, sock in self.sockets:
            send_bridge(sock, MSG_RESET, 0, 0)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        chat_template: bool | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must be non-empty")
        token_budget = self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if token_budget <= 0:
            raise ValueError("max_new_tokens must be positive")

        prompt_ids = tokenize_prompt(
            self.tokenizer,
            prompt,
            self.chat_template if chat_template is None else bool(chat_template),
            self.system if system is None else system,
            self.device,
        )
        prompt_len = int(prompt_ids.shape[1])
        if prompt_len + token_budget > self.max_cache_len:
            raise ValueError(
                f"prompt tokens ({prompt_len}) + max_new_tokens ({token_budget}) exceeds "
                f"max_cache_len ({self.max_cache_len})"
            )

        generated_ids: list[int] = prompt_ids[0].detach().cpu().tolist()
        predicted_next: int | None = None
        trials: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        eos_token_id = self.tokenizer.eos_token_id
        started_at = time.time()

        self.reset_all()
        try:
            with torch.inference_mode():
                for step in range(prompt_len + token_budget):
                    if step < prompt_len:
                        token = prompt_ids[:, step : step + 1]
                        token_id = int(token.item())
                        phase = "prefill"
                    else:
                        if predicted_next is None:
                            raise RuntimeError("decode step reached before predicted token was available")
                        token_id = predicted_next
                        token = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                        phase = "decode"

                    hidden = self.model.model.embed_tokens(token)
                    body = tensor_to_float32_bytes(hidden)

                    step_started = time.perf_counter()
                    current = body
                    for _, sock in self.sockets:
                        current = send_bridge(sock, MSG_STEP, step, step + 1, current)
                    elapsed_ms = (time.perf_counter() - step_started) * 1000.0
                    latencies_ms.append(elapsed_ms)

                    final_hidden = float32_bytes_to_tensor(current, self.device, self.hidden_size)
                    logits = self.model.lm_head(self.model.model.norm(final_hidden))
                    predicted_next = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                    kept = step >= prompt_len - 1
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
                            "responseBytes": len(current),
                            "latencyMs": elapsed_ms,
                        }
                    )
                    if phase == "decode" and eos_token_id is not None and predicted_next == int(eos_token_id):
                        break
                    if len(generated_ids) >= prompt_len + token_budget:
                        break
        finally:
            try:
                self.reset_all()
            except Exception as exc:
                print(f"[native-coordinator] reset failed: {exc}", flush=True)

        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {
            "ok": True,
            "mode": "llama",
            "modelId": self.model_id,
            "route": ",".join(f"{sid}={host}:{port}" for sid, host, port in self.route),
            "routeHops": [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in self.route],
            "prompt": prompt,
            "promptTokenIds": prompt_ids[0].detach().cpu().tolist(),
            "generatedTokenIds": generated_ids,
            "promptTokenCount": prompt_len,
            "generatedTokenCount": max(len(generated_ids) - prompt_len, 0),
            "output": output_text,
            "durationMs": int((time.time() - started_at) * 1000),
            "trials": trials,
            "summary": {
                "minMs": min(latencies_ms) if latencies_ms else 0.0,
                "p50Ms": percentile(latencies_ms, 50) if latencies_ms else 0.0,
                "p95Ms": percentile(latencies_ms, 95) if latencies_ms else 0.0,
                "maxMs": max(latencies_ms) if latencies_ms else 0.0,
            },
        }


def execute_native_route(
    route_arg: str,
    prompt: str,
    artifact_dir: str,
    model_id: str | None,
    max_new_tokens: int,
    timeout: float,
    device: str,
    dtype: str,
    chat_template: bool,
    system: str | None,
) -> dict[str, Any]:
    service = NativeExecutorLlamaCoordinatorService(
        route_arg=route_arg,
        artifact_dir=artifact_dir,
        model_id=model_id,
        timeout=timeout,
        device=device,
        dtype=dtype,
        default_max_new_tokens=max_new_tokens,
        chat_template=chat_template,
        system=system,
    )
    try:
        return service.generate(prompt, max_new_tokens=max_new_tokens, chat_template=chat_template, system=system)
    finally:
        service.close()


class CoordinatorRequestHandler(BaseHTTPRequestHandler):
    service: NativeExecutorLlamaCoordinatorService

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
                f"[native-coordinator] prompt_tokens={result['promptTokenCount']} "
                f"generated_tokens={result['generatedTokenCount']} duration_ms={result['durationMs']}",
                flush=True,
            )
            print(f"[native-coordinator] output: {result['output']}", flush=True)
            self._write_json(HTTPStatus.OK, result)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        return


def run_oneshot(service: NativeExecutorLlamaCoordinatorService, args: argparse.Namespace) -> None:
    result = service.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        system=args.system,
        chat_template=args.chat_template,
    )
    print(f"Model: {result['modelId']}")
    print(f"Route: {result['route']}")
    print(f"Prompt: {args.prompt!r}")
    print()
    for trial in result["trials"]:
        disposition = "kept" if trial["kept"] else "discarded"
        print(
            f"step={trial['step']} phase={trial['phase']} in={trial['inputTokenText']!r} "
            f"pred={trial['predictedTokenText']!r} {disposition} "
            f"bytes={trial['requestBytes']}->{trial['responseBytes']} latency_ms={trial['latencyMs']:.2f}"
        )
    print()
    print("Output:")
    print(result["output"])
    print()
    print(
        "Summary: "
        f"steps={len(result['trials'])} p50_ms={result['summary']['p50Ms']:.2f} "
        f"p95_ms={result['summary']['p95Ms']:.2f} max_ms={result['summary']['maxMs']:.2f}"
    )


def run_server(service: NativeExecutorLlamaCoordinatorService, host: str, port: int) -> None:
    CoordinatorRequestHandler.service = service
    server = HTTPServer((host, port), CoordinatorRequestHandler)
    print(f"[native-coordinator] listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    args = parse_args()
    service = NativeExecutorLlamaCoordinatorService(
        route_arg=args.route,
        artifact_dir=args.artifact_dir,
        model_id=args.model_id,
        timeout=args.timeout,
        device=args.device,
        dtype=args.dtype,
        default_max_new_tokens=args.default_max_new_tokens,
        chat_template=args.chat_template,
        system=args.system,
    )
    if args.prompt is not None:
        run_oneshot(service, args)
        service.close()
        return
    try:
        run_server(service, args.listen_host, args.listen_port)
    finally:
        service.close()


if __name__ == "__main__":
    main()
