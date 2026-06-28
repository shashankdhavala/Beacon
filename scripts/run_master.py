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
from transformers import AutoTokenizer

from export_shards import is_gpt2_model, is_llama_model, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the master process")
    parser.add_argument("--artifact-dir", default="artifacts/gpt2")
    parser.add_argument("--prompt", default="Beacon helps coordinate offline medical guidance.")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
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
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def embed_token(self, token_ids: torch.Tensor, step: int) -> torch.Tensor:
        if is_gpt2_model(self.model):
            position_ids = torch.tensor([[step]], dtype=torch.long, device=self.device)
            hidden_states = self.model.transformer.wte(token_ids) + self.model.transformer.wpe(position_ids)
            return self.model.transformer.drop(hidden_states)
        if is_llama_model(self.model):
            return self.model.model.embed_tokens(token_ids)
        raise TypeError(f"Unsupported model type: {type(self.model)}")

    def project_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_gpt2_model(self.model):
            hidden_states = self.model.transformer.ln_f(hidden_states)
            return self.model.lm_head(hidden_states)
        if is_llama_model(self.model):
            hidden_states = self.model.model.norm(hidden_states)
            return self.model.lm_head(hidden_states)
        raise TypeError(f"Unsupported model type: {type(self.model)}")

    def start_request(self) -> None:
        send_message(self.shard0_sock, {"type": "start_request"})
        receive_message(self.shard0_sock)

    def flush_request(self) -> None:
        send_message(self.shard0_sock, {"type": "flush_request"})
        receive_message(self.shard0_sock)

    def generate_distributed(self, prompt: str, max_new_tokens: int) -> List[int]:
        prompt_ids = self.tokenize(prompt)
        prompt_len = int(prompt_ids.shape[1])
        generated = prompt_ids.clone()
        if max_new_tokens == 0:
            return generated[0].tolist()
        eos_token_id = self.tokenizer.eos_token_id
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
                    if eos_token_id is not None and int(token.item()) == int(eos_token_id):
                        break

                hidden_states = self.embed_token(token, step)
                send_message(
                    self.shard0_sock,
                    {
                        "type": "step_hidden",
                        "hidden_states": hidden_states.cpu(),
                        "step": step,
                        "current_length": step + 1,
                    },
                )
                response = receive_message(self.shard0_sock)
                last_hidden = response["hidden_states"].to(self.device)
                last_logits = self.project_logits(last_hidden)

            if generated.shape[1] == prompt_len:
                generated = torch.cat(
                    [generated, torch.argmax(last_logits[:, -1, :], dim=-1, keepdim=True)],
                    dim=1,
                )
            return generated[0].tolist()
        finally:
            self.flush_request()


def connect_with_retry(host: str, port: int) -> socket.socket:
    attempts = 0
    while True:
        try:
            sock = socket.create_connection((host, port), timeout=5)
            sock.settimeout(None)
            return sock
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

    model = load_model(manifest["model_id"], args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_id"])
    shard0_sock = connect_with_retry(args.host, args.port)
    master = MasterNode(model=model, tokenizer=tokenizer, shard0_sock=shard0_sock, device=args.device)

    distributed_ids = master.generate_distributed(args.prompt, args.max_new_tokens)

    print(f"Model: {manifest['model_id']}")
    print(f"Prompt: {args.prompt}")
    print(f"Num devices: {manifest['num_devices']}")
    print(f"Layer ranges: {manifest['layer_ranges']}")
    print(f"Max cache len: {manifest['max_cache_len']}")
    print()
    print("Distributed sharded text:")
    print(master.detokenize(distributed_ids))


if __name__ == "__main__":
    main()
