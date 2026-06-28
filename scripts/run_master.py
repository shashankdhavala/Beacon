#!/usr/bin/env python3
"""
Run the persistent master process for the local TCP sharded inference simulation.
"""

from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer

from export_shards import is_gpt2_model, is_llama_model, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the persistent master process")
    parser.add_argument("--artifact-dir", default="artifacts/gpt2")
    parser.add_argument("--shard-host", default="127.0.0.1")
    parser.add_argument("--shard-port", type=int, required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=9200)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default-max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
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
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.shard0_sock = shard0_sock
        self.device = device

    def format_prompt(self, prompt: str) -> str:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def tokenize(self, prompt: str) -> torch.Tensor:
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
        formatted_prompt = self.format_prompt(prompt)
        return self.tokenizer(formatted_prompt, return_tensors="pt")["input_ids"].to(self.device)

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
        return self.generate_distributed_from_ids(prompt_ids, max_new_tokens)

    def generate_distributed_from_ids(self, prompt_ids: torch.Tensor, max_new_tokens: int) -> List[int]:
        prompt_len = int(prompt_ids.shape[1])
        generated = prompt_ids.clone()
        if max_new_tokens == 0:
            return generated[0].tolist()
        eos_token_id = self.tokenizer.eos_token_id
        last_logits = None
        request_started = False

        self.start_request()
        request_started = True
        try:
            with torch.inference_mode():
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
            if request_started:
                try:
                    self.flush_request()
                except Exception as exc:
                    print(f"[master] flush_request failed: {exc}", flush=True)


class MasterService:
    def __init__(self, master: MasterNode, default_max_new_tokens: int):
        self.master = master
        self.default_max_new_tokens = default_max_new_tokens

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> Dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must be non-empty")
        token_budget = self.default_max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        if token_budget < 0:
            raise ValueError("max_new_tokens must be >= 0")
        started_at = time.time()
        prompt_ids = self.master.tokenize(prompt)
        prompt_token_count = int(prompt_ids.shape[1])
        token_ids = self.master.generate_distributed_from_ids(prompt_ids, token_budget)
        text = self.master.detokenize(token_ids)
        duration_ms = int((time.time() - started_at) * 1000)
        total_token_count = len(token_ids)
        return {
            "prompt": prompt,
            "max_new_tokens": token_budget,
            "prompt_token_count": prompt_token_count,
            "generated_token_count": max(total_token_count - prompt_token_count, 0),
            "total_token_count": total_token_count,
            "token_ids": token_ids,
            "text": text,
            "duration_ms": duration_ms,
        }


class MasterRequestHandler(BaseHTTPRequestHandler):
    service: MasterService

    def _write_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
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
            )
            print(
                f"[master] prompt_tokens={result['prompt_token_count']} "
                f"generated_tokens={result['generated_token_count']} "
                f"duration_ms={result['duration_ms']}",
                flush=True,
            )
            print(f"[master] output: {result['text']}", flush=True)
            self._write_json(HTTPStatus.OK, result)
        except Exception as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        return


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


def load_master_service(args: argparse.Namespace) -> MasterService:
    manifest = json.loads((Path(args.artifact_dir) / "manifest.json").read_text())
    if int(manifest["num_devices"]) != 3:
        raise ValueError("the multi-process simulation currently expects exactly 3 shards")

    print(f"[master] loading model from {manifest['model_id']}", flush=True)
    model = load_model(manifest["model_id"], args.device, args.dtype)
    print(f"[master] loading tokenizer from {manifest['model_id']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_id"])
    print(f"[master] connecting to shard_0 at {args.shard_host}:{args.shard_port}", flush=True)
    shard0_sock = connect_with_retry(args.shard_host, args.shard_port)
    master = MasterNode(model=model, tokenizer=tokenizer, shard0_sock=shard0_sock, device=args.device)
    print("[master] ready", flush=True)
    return MasterService(master=master, default_max_new_tokens=args.default_max_new_tokens)


def run_oneshot(service: MasterService, prompt: str, max_new_tokens: int | None) -> None:
    result = service.generate(prompt, max_new_tokens)
    print(json.dumps(result, indent=2))


def run_server(service: MasterService, host: str, port: int) -> None:
    MasterRequestHandler.service = service
    server = HTTPServer((host, port), MasterRequestHandler)
    print(f"[master] listening on http://{host}:{port}", flush=True)
    server.serve_forever()


def main() -> None:
    args = parse_args()
    service = load_master_service(args)
    if args.prompt is not None:
        run_oneshot(service, args.prompt, args.max_new_tokens)
        return
    run_server(service, args.listen_host, args.listen_port)


if __name__ == "__main__":
    main()
