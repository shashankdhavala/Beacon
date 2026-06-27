#!/usr/bin/env python3
"""
Run the master process for the local TCP sharded inference simulation.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from export_shards import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the master process")
    parser.add_argument("--artifact-dir", default="artifacts/gpt2")
    parser.add_argument("--prompt", default="Beacon helps coordinate offline medical guidance.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
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


class MasterNode:
    def __init__(self, model, tokenizer, shard0_sock: socket.socket, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.shard0_sock = shard0_sock
        self.device = device

    def tokenize(self, prompt: str) -> torch.Tensor:
        return self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

    def detokenize(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def start_request(self) -> None:
        send_message(self.shard0_sock, {"type": "start_request"})
        receive_message(self.shard0_sock)

    def flush_request(self) -> None:
        send_message(self.shard0_sock, {"type": "flush_request"})
        receive_message(self.shard0_sock)

    def generate_reference(self, prompt: str, max_new_tokens: int) -> List[int]:
        prompt_ids = self.tokenize(prompt)
        generated = prompt_ids.clone()
        if max_new_tokens == 0:
            return generated[0].tolist()
        past_key_values = None
        for step in range(prompt_ids.shape[1] + max_new_tokens - 1):
            token = prompt_ids[:, step : step + 1] if step < prompt_ids.shape[1] else generated[:, -1:]
            total_length = step + 1 if step < prompt_ids.shape[1] else generated.shape[1]
            attention_mask = torch.ones(1, total_length, dtype=torch.long, device=self.device)
            outputs = self.model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            predicted = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            if step >= prompt_ids.shape[1] - 1:
                generated = torch.cat([generated, predicted], dim=1)
                if generated.shape[1] >= prompt_ids.shape[1] + max_new_tokens:
                    break
        return generated[0].tolist()

    def generate_distributed(self, prompt: str, max_new_tokens: int) -> tuple[List[int], List[float]]:
        prompt_ids = self.tokenize(prompt)
        prompt_len = int(prompt_ids.shape[1])
        generated = prompt_ids.clone()
        if max_new_tokens == 0:
            return generated[0].tolist(), []
        per_step_max_abs_diff: List[float] = []
        last_logits = None

        self.start_request()
        try:
            total_steps = prompt_len + max_new_tokens
            for step in range(total_steps):
                if step < prompt_len:
                    token = prompt_ids[:, step : step + 1]
                else:
                    token = torch.argmax(last_logits[:, -1, :], dim=-1, keepdim=True)
                    generated = torch.cat([generated, token], dim=1)

                send_message(
                    self.shard0_sock,
                    {
                        "type": "step_token",
                        "token_ids": token.cpu(),
                        "step": step,
                        "current_length": step + 1,
                    },
                )
                response = receive_message(self.shard0_sock)
                last_logits = response["logits"].to(self.device)

                if step >= prompt_len - 1:
                    reference_outputs = self.model(
                        input_ids=generated,
                        attention_mask=torch.ones(1, generated.shape[1], dtype=torch.long, device=self.device),
                        use_cache=False,
                    )
                    diff = torch.max(
                        torch.abs(reference_outputs.logits[:, -1, :] - last_logits[:, -1, :])
                    ).item()
                    per_step_max_abs_diff.append(diff)

            if generated.shape[1] == prompt_len:
                generated = torch.cat(
                    [generated, torch.argmax(last_logits[:, -1, :], dim=-1, keepdim=True)],
                    dim=1,
                )
            return generated[0].tolist(), per_step_max_abs_diff
        finally:
            self.flush_request()


def connect_with_retry(host: str, port: int) -> socket.socket:
    attempts = 0
    while True:
        try:
            return socket.create_connection((host, port), timeout=5)
        except OSError:
            attempts += 1
            if attempts >= 50:
                raise
            time.sleep(0.2)


def main() -> None:
    args = parse_args()
    manifest = json.loads((Path(args.artifact_dir) / "manifest.json").read_text())
    if int(manifest["num_devices"]) != 3:
        raise ValueError("the multi-process simulation currently expects exactly 3 shards")

    model, tokenizer = load_model_and_tokenizer(manifest["model_id"], args.device)
    shard0_sock = connect_with_retry(args.host, args.port)
    master = MasterNode(model=model, tokenizer=tokenizer, shard0_sock=shard0_sock, device=args.device)

    reference_ids = master.generate_reference(args.prompt, args.max_new_tokens)
    distributed_ids, per_step_diffs = master.generate_distributed(args.prompt, args.max_new_tokens)

    print(f"Model: {manifest['model_id']}")
    print(f"Prompt: {args.prompt}")
    print(f"Num devices: {manifest['num_devices']}")
    print(f"Layer ranges: {manifest['layer_ranges']}")
    print(f"Max cache len: {manifest['max_cache_len']}")
    print(f"Per-step max |logit diff|: {[round(v, 8) for v in per_step_diffs]}")
    print()
    print("Reference text:")
    print(master.detokenize(reference_ids))
    print()
    print("Distributed sharded text:")
    print(master.detokenize(distributed_ids))

    if reference_ids != distributed_ids:
        raise SystemExit("Parity check failed: distributed sharded generation does not match reference generation.")

    print()
    print("Parity check passed.")


if __name__ == "__main__":
    main()
