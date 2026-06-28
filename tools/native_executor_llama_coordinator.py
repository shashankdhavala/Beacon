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
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--system", default=None)
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
    manifest = load_manifest(artifact_dir)
    architecture = manifest.get("architecture") or manifest.get("shards", [{}])[0].get("architecture")
    if architecture != "llama":
        raise ValueError(f"expected llama manifest, got architecture={architecture!r}")

    route = parse_route(route_arg)
    if len(route) != int(manifest["num_devices"]):
        raise ValueError(f"manifest expects {manifest['num_devices']} shards, got route with {len(route)}")

    hidden_size = int(manifest.get("hidden_size", 3072))
    resolved_model_id = model_id or manifest["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    model = AutoModelForCausalLM.from_pretrained(resolved_model_id, dtype=parse_torch_dtype(dtype)).to(device)
    model.eval()

    prompt_ids = tokenize_prompt(tokenizer, prompt, chat_template, system, device)
    prompt_len = int(prompt_ids.shape[1])
    max_cache_len = int(manifest["max_cache_len"])
    if prompt_len + max_new_tokens > max_cache_len:
        raise ValueError(f"prompt tokens ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds max_cache_len ({max_cache_len})")

    sockets: list[tuple[int, socket.socket]] = []
    try:
        for shard_id, host, port in route:
            sock = socket.create_connection((host, port), timeout=timeout)
            sock.settimeout(timeout)
            configure(sock)
            send_bridge(sock, MSG_RESET, 0, 0)
            sockets.append((shard_id, sock))

        generated_ids: list[int] = prompt_ids[0].detach().cpu().tolist()
        predicted_next: int | None = None
        trials: list[dict[str, Any]] = []
        latencies_ms: list[float] = []
        eos_token_id = tokenizer.eos_token_id

        with torch.no_grad():
            for step in range(prompt_len + max_new_tokens):
                if step < prompt_len:
                    token = prompt_ids[:, step : step + 1]
                    token_id = int(token.item())
                    phase = "prefill"
                else:
                    if predicted_next is None:
                        raise RuntimeError("decode step reached before predicted token was available")
                    token_id = predicted_next
                    token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                    phase = "decode"

                hidden = model.model.embed_tokens(token)
                body = tensor_to_float32_bytes(hidden)

                started = time.perf_counter()
                current = body
                for _, sock in sockets:
                    current = send_bridge(sock, MSG_STEP, step, step + 1, current)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies_ms.append(elapsed_ms)

                final_hidden = float32_bytes_to_tensor(current, device, hidden_size)
                logits = model.lm_head(model.model.norm(final_hidden))
                predicted_next = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                kept = step >= prompt_len - 1
                if kept:
                    generated_ids.append(predicted_next)

                trials.append(
                    {
                        "step": step,
                        "phase": phase,
                        "inputTokenId": token_id,
                        "inputTokenText": tokenizer.decode([token_id], skip_special_tokens=False),
                        "predictedTokenId": predicted_next,
                        "predictedTokenText": tokenizer.decode([predicted_next], skip_special_tokens=False),
                        "kept": kept,
                        "requestBytes": len(body),
                        "responseBytes": len(current),
                        "latencyMs": elapsed_ms,
                    }
                )
                if phase == "decode" and eos_token_id is not None and predicted_next == int(eos_token_id):
                    break
                if len(generated_ids) >= prompt_len + max_new_tokens:
                    break

        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {
            "modelId": resolved_model_id,
            "route": route_arg,
            "prompt": prompt,
            "output": output_text,
            "trials": trials,
            "summary": {
                "minMs": min(latencies_ms) if latencies_ms else 0.0,
                "p50Ms": percentile(latencies_ms, 50) if latencies_ms else 0.0,
                "p95Ms": percentile(latencies_ms, 95) if latencies_ms else 0.0,
                "maxMs": max(latencies_ms) if latencies_ms else 0.0,
            },
        }
    finally:
        for _, sock in sockets:
            sock.close()


def main() -> None:
    args = parse_args()
    result = execute_native_route(
        route_arg=args.route,
        prompt=args.prompt,
        artifact_dir=args.artifact_dir,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
        device=args.device,
        dtype=args.dtype,
        chat_template=args.chat_template,
        system=args.system,
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


if __name__ == "__main__":
    main()
