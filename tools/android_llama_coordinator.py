#!/usr/bin/env python3
"""
Mac coordinator for Llama over Android ExecuTorch shards.

The Mac owns tokenization, token embeddings, final RMSNorm, and LM head
projection. Android workers own exported decoder-layer shard .pte files and
keep their local KV cache across one-token prefill/decode steps.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mac_coordinator import (
    TensorMessage,
    configure_socket,
    parse_route,
    percentile,
    recv_tensor,
    send_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Llama through Android shard workers.")
    parser.add_argument("--artifact-dir", default="artifacts/llama32_3b_sm8750_3way")
    parser.add_argument("--model-id", default=None, help="Override manifest model_id for the Mac-side tokenizer/model.")
    parser.add_argument("--route", required=True, help='Shard route, e.g. "1=10.0.0.11:9000,2=10.0.0.12:9000,3=10.0.0.13:9000"')
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--checksum", default="none", choices=["sha256", "none"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--chat-template", action="store_true", help="Wrap --prompt with the tokenizer chat template for instruct checkpoints.")
    parser.add_argument("--system", default=None, help="Optional system prompt when --chat-template is used.")
    return parser.parse_args()


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


def tokenize_prompt(tokenizer, prompt: str, chat_template: bool, system: str | None, device: str) -> torch.Tensor:
    if not chat_template:
        return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)


def execute_llama_route(
    route_arg: str,
    prompt: str,
    artifact_dir: str = "artifacts/llama32_3b_sm8750_3way",
    model_id: str | None = None,
    max_new_tokens: int = 8,
    timeout: float = 120.0,
    checksum: str = "none",
    device: str = "cpu",
    dtype: str = "float32",
    chat_template: bool = False,
    system: str | None = None,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    manifest = load_manifest(artifact_dir)
    architecture = manifest.get("architecture") or manifest.get("shards", [{}])[0].get("architecture")
    if architecture != "llama":
        raise ValueError(f"expected a llama manifest, got architecture={architecture!r}")

    route, route_json = route_to_json(route_arg)
    expected_devices = int(manifest["num_devices"])
    if len(route) != expected_devices:
        raise ValueError(f"manifest expects {expected_devices} shards, got route with {len(route)}")

    resolved_model_id = model_id or manifest["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_id,
        dtype=parse_torch_dtype(dtype),
    ).to(device)
    model.eval()

    prompt_ids = tokenize_prompt(tokenizer, prompt, chat_template, system, device)
    prompt_len = int(prompt_ids.shape[1])
    max_cache_len = int(manifest["max_cache_len"])
    if prompt_len + max_new_tokens > max_cache_len:
        raise ValueError(
            f"prompt tokens ({prompt_len}) + max_new_tokens ({max_new_tokens}) exceeds max_cache_len ({max_cache_len})"
        )

    first_shard, first_host, first_port = route[0]
    request_prefix = f"android-llama-{int(time.time() * 1000)}"
    generated_ids: list[int] = prompt_ids[0].detach().cpu().tolist()
    predicted_next: int | None = None
    trials: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    eos_token_id = tokenizer.eos_token_id

    with socket.create_connection((first_host, first_port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        configure_socket(sock)
        start_header, _, start_ms = send_control(
            sock=sock,
            route_json=route_json,
            first_shard=first_shard,
            message_type="START_REQUEST",
            request_id=f"{request_prefix}-start",
            model_id=resolved_model_id,
            checksum=checksum,
        )

        try:
            with torch.no_grad():
                for step in range(prompt_len + max_new_tokens):
                    if step < prompt_len:
                        token = prompt_ids[:, step : step + 1]
                        token_id = int(token.item())
                        phase = "prefill"
                    else:
                        if predicted_next is None:
                            raise RuntimeError("decode step reached before a predicted token was available")
                        token_id = predicted_next
                        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                        phase = "decode"

                    hidden = model.model.embed_tokens(token)
                    body = tensor_to_float32_bytes(hidden)
                    shape = list(hidden.shape)
                    msg = TensorMessage(
                        message_type="STEP_HIDDEN",
                        request_id=f"{request_prefix}-step-{step}",
                        step=step,
                        source_shard=0,
                        target_shard=first_shard,
                        shape=shape,
                        dtype="float32",
                        bytes_data=body,
                        response_mode="echo",
                        checksum=checksum,
                        route=route_json,
                        extra_headers={
                            "modelId": resolved_model_id,
                            "architecture": "llama",
                            "currentLength": step + 1,
                        },
                    )

                    started = time.perf_counter()
                    send_tensor(sock, msg)
                    header, response_body = recv_tensor(sock)
                    raise_for_worker_error(header, response_body)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    latencies_ms.append(elapsed_ms)

                    if header.get("dtype") != "float32":
                        raise ValueError(f"expected float32 hidden state, got {header.get('dtype')}")
                    output_shape = [int(value) for value in header.get("shape", [])]
                    final_hidden = float32_bytes_to_tensor(response_body, output_shape, device)
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
                            "responseBytes": len(response_body),
                            "latencyMs": elapsed_ms,
                            "responseHeader": header,
                        }
                    )
                    if phase == "decode" and eos_token_id is not None and predicted_next == int(eos_token_id):
                        break
                    if len(generated_ids) >= prompt_len + max_new_tokens:
                        break
        finally:
            flush_header, _, flush_ms = send_control(
                sock=sock,
                route_json=route_json,
                first_shard=first_shard,
                message_type="FLUSH_REQUEST",
                request_id=f"{request_prefix}-flush",
                model_id=resolved_model_id,
                checksum=checksum,
            )

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return {
        "ok": True,
        "mode": "llama",
        "modelId": resolved_model_id,
        "prompt": prompt,
        "promptTokenIds": prompt_ids[0].detach().cpu().tolist(),
        "generatedTokenIds": generated_ids,
        "output": output_text,
        "route": route_arg,
        "routeHops": [{"shardId": shard_id, "host": host, "port": port} for shard_id, host, port in route],
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


def main() -> None:
    args = parse_args()
    result = execute_llama_route(
        route_arg=args.route,
        prompt=args.prompt,
        artifact_dir=args.artifact_dir,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
        checksum=args.checksum,
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
