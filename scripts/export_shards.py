#!/usr/bin/env python3
"""
Export GPT-2 shard artifacts for local multi-device ExecuTorch testing.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from executorch.exir import to_edge
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel


def prepare_attention_mask(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    expanded_mask = attention_mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - expanded_mask) * torch.finfo(dtype).min


def make_fixed_attention_mask(current_length: int, max_cache_len: int, device: str) -> torch.Tensor:
    if current_length > max_cache_len:
        raise ValueError(f"current_length={current_length} exceeds max_cache_len={max_cache_len}")
    mask = torch.zeros(1, max_cache_len, dtype=torch.long, device=device)
    mask[:, :current_length] = 1
    return mask


def compute_layer_ranges(num_layers: int, num_shards: int) -> List[Tuple[int, int]]:
    base = num_layers // num_shards
    remainder = num_layers % num_shards
    ranges: List[Tuple[int, int]] = []
    start = 0
    for idx in range(num_shards):
        count = base + (1 if idx < remainder else 0)
        end = start + count
        ranges.append((start, end))
        start = end
    return ranges


class LayerCacheView:
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        self.keys = keys
        self.values = values


class FixedSizeTensorCache:
    def __init__(
        self,
        layer_ids: Sequence[int],
        keys_list: Sequence[torch.Tensor],
        values_list: Sequence[torch.Tensor],
        cache_position: torch.Tensor,
    ):
        self.layer_to_slot = {layer_id: idx for idx, layer_id in enumerate(layer_ids)}
        self.layers = [LayerCacheView(k, v) for k, v in zip(keys_list, values_list)]
        self.cache_position = cache_position

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slot = self.layer_to_slot[layer_idx]
        layer = self.layers[slot]
        new_keys = layer.keys.clone()
        new_values = layer.values.clone()
        new_keys.index_copy_(2, self.cache_position, key_states.contiguous())
        new_values.index_copy_(2, self.cache_position, value_states.contiguous())
        layer.keys = new_keys
        layer.values = new_values
        return layer.keys, layer.values


class GPT2DecodeOnlyShard(nn.Module):
    def __init__(
        self,
        full_model: GPT2LMHeadModel,
        start_idx: int,
        end_idx: int,
        max_cache_len: int,
        is_first: bool,
        is_last: bool,
    ):
        super().__init__()
        self.transformer = full_model.transformer
        self.blocks = nn.ModuleList(self.transformer.h[start_idx:end_idx])
        self.layer_ids = [block.attn.layer_idx for block in self.blocks]
        self.max_cache_len = max_cache_len
        self.is_first = is_first
        self.is_last = is_last
        self.has_blocks = len(self.blocks) > 0
        if self.is_last:
            self.ln_f = self.transformer.ln_f
            self.lm_head = full_model.lm_head

    def _allocate_cache(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        keys = [
            torch.zeros(batch_size, num_heads, self.max_cache_len, head_dim, device=device, dtype=dtype)
            for _ in range(len(self.blocks))
        ]
        values = [
            torch.zeros(batch_size, num_heads, self.max_cache_len, head_dim, device=device, dtype=dtype)
            for _ in range(len(self.blocks))
        ]
        return keys, values

    @staticmethod
    def _flatten_cache(cache: FixedSizeTensorCache) -> tuple[torch.Tensor, ...]:
        flat: List[torch.Tensor] = []
        for layer in cache.layers:
            flat.extend([layer.keys.contiguous(), layer.values.contiguous()])
        return tuple(flat)

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.is_first:
            input_ids, attention_mask, position_ids, cache_position, *past_kv = args
            hidden_states = self.transformer.wte(input_ids) + self.transformer.wpe(position_ids)
            hidden_states = self.transformer.drop(hidden_states)
        else:
            hidden_states, attention_mask, cache_position, *past_kv = args

        if self.has_blocks:
            if len(past_kv) == 0:
                num_heads = self.blocks[0].attn.num_heads
                head_dim = self.blocks[0].attn.head_dim
                keys, values = self._allocate_cache(
                    batch_size=hidden_states.shape[0],
                    num_heads=num_heads,
                    head_dim=head_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            else:
                keys = list(past_kv[0::2])
                values = list(past_kv[1::2])
            cache = FixedSizeTensorCache(self.layer_ids, keys, values, cache_position)
            block_attention_mask = prepare_attention_mask(attention_mask, hidden_states.dtype)
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    past_key_values=cache,
                    attention_mask=block_attention_mask,
                    use_cache=True,
                )
            cache_outputs = self._flatten_cache(cache)
        else:
            cache_outputs = ()

        if self.is_last:
            hidden_states = self.ln_f(hidden_states)
            output = self.lm_head(hidden_states).contiguous()
        else:
            output = hidden_states.contiguous()
        return (output, *cache_outputs)


@dataclass
class ShardArtifactPath:
    path: str


@dataclass
class ShardManifest:
    shard_index: int
    artifact_path: str
    start_layer: int
    end_layer: int
    num_layers: int
    num_heads: int
    head_dim: int
    cache_dtype: str
    is_first: bool
    is_last: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GPT-2 ExecuTorch shard artifacts")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--prompt", default="Beacon helps coordinate offline medical guidance.")
    parser.add_argument("--num-devices", type=int, default=3)
    parser.add_argument("--artifact-dir", default="artifacts/gpt2")
    parser.add_argument("--max-cache-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_model_and_tokenizer(model_id: str, device: str) -> tuple[GPT2LMHeadModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    if not isinstance(model, GPT2LMHeadModel):
        raise TypeError(f"Expected a GPT-2 style model, got {type(model)}")
    model.to(device)
    model.eval()
    return model, tokenizer


def build_shard_modules(
    model: GPT2LMHeadModel,
    layer_ranges: Sequence[Tuple[int, int]],
    max_cache_len: int,
) -> List[GPT2DecodeOnlyShard]:
    return [
        GPT2DecodeOnlyShard(
            model,
            start,
            end,
            max_cache_len,
            is_first=(idx == 0),
            is_last=(idx == len(layer_ranges) - 1),
        ).eval()
        for idx, (start, end) in enumerate(layer_ranges)
    ]


def build_export_examples(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer_ranges: Sequence[Tuple[int, int]],
    max_cache_len: int,
    device: str,
) -> Tuple[List[GPT2DecodeOnlyShard], List[tuple[torch.Tensor, ...]]]:
    modules = build_shard_modules(model, layer_ranges, max_cache_len)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    first_token = input_ids[:, :1]
    attention_mask = make_fixed_attention_mask(1, max_cache_len, device)
    position_ids = torch.tensor([[0]], dtype=torch.long, device=device)
    cache_position = torch.tensor([0], dtype=torch.long, device=device)

    args_list: List[tuple[torch.Tensor, ...]] = []
    current = first_token
    for idx, module in enumerate(modules):
        if module.has_blocks:
            num_heads = module.blocks[0].attn.num_heads
            head_dim = module.blocks[0].attn.head_dim
            keys = [
                torch.zeros(1, num_heads, max_cache_len, head_dim, device=device, dtype=model.dtype)
                for _ in range(len(module.blocks))
            ]
            values = [
                torch.zeros(1, num_heads, max_cache_len, head_dim, device=device, dtype=model.dtype)
                for _ in range(len(module.blocks))
            ]
            cache_args = tuple(t for pair in zip(keys, values) for t in pair)
        else:
            cache_args = ()

        shard_args = (
            (current, attention_mask, position_ids, cache_position, *cache_args)
            if idx == 0
            else (current, attention_mask, cache_position, *cache_args)
        )
        args_list.append(shard_args)
        current = module(*shard_args)[0]
    return modules, args_list


def export_to_pte(module: nn.Module, args: tuple, output_path: Path) -> str:
    exported = torch.export.export(module.eval(), args, strict=False)
    executorch_program = to_edge(exported).to_executorch()
    with open(output_path, "wb") as fp:
        executorch_program.write_to_file(fp)
    return str(output_path)


def export_artifacts(
    modules: Sequence[nn.Module],
    args_list: Sequence[tuple[torch.Tensor, ...]],
    artifact_dir: Path,
) -> List[ShardArtifactPath]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths: List[ShardArtifactPath] = []
    for idx, (module, args) in enumerate(zip(modules, args_list)):
        output_path = (artifact_dir / f"shard_{idx}.pte").resolve()
        paths.append(ShardArtifactPath(path=export_to_pte(module, args, output_path)))
    return paths


def write_manifest(
    artifact_dir: Path,
    model_id: str,
    num_devices: int,
    max_cache_len: int,
    layer_ranges: Sequence[Tuple[int, int]],
    modules: Sequence[GPT2DecodeOnlyShard],
    artifact_paths: Sequence[ShardArtifactPath],
) -> Path:
    shards: List[dict] = []
    for idx, (layer_range, module, artifact_path) in enumerate(zip(layer_ranges, modules, artifact_paths)):
        if module.has_blocks:
            num_heads = module.blocks[0].attn.num_heads
            head_dim = module.blocks[0].attn.head_dim
        else:
            num_heads = 0
            head_dim = 0
        shards.append(
            ShardManifest(
                shard_index=idx,
                artifact_path=artifact_path.path,
                start_layer=layer_range[0],
                end_layer=layer_range[1],
                num_layers=layer_range[1] - layer_range[0],
                num_heads=num_heads,
                head_dim=head_dim,
                cache_dtype=str(next(module.parameters()).dtype).replace("torch.", ""),
                is_first=module.is_first,
                is_last=module.is_last,
            ).__dict__
        )

    manifest = {
        "model_id": model_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "num_devices": num_devices,
        "max_cache_len": max_cache_len,
        "layer_ranges": [list(r) for r in layer_ranges],
        "shards": shards,
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)

    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device)
    num_layers = len(model.transformer.h)
    if args.num_devices < 2:
        raise ValueError("num_devices must be at least 2")

    prompt_len = int(tokenizer(args.prompt, return_tensors="pt")["input_ids"].shape[1])
    required_cache_len = prompt_len + args.max_new_tokens
    if args.max_cache_len < required_cache_len:
        raise ValueError(
            f"max_cache_len={args.max_cache_len} is too small for prompt_len={prompt_len} "
            f"and max_new_tokens={args.max_new_tokens}"
        )

    layer_ranges = compute_layer_ranges(num_layers, args.num_devices)
    modules, args_list = build_export_examples(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        layer_ranges=layer_ranges,
        max_cache_len=args.max_cache_len,
        device=args.device,
    )
    artifact_paths = export_artifacts(modules, args_list, Path(args.artifact_dir))
    manifest_path = write_manifest(
        artifact_dir=Path(args.artifact_dir),
        model_id=args.model_id,
        num_devices=args.num_devices,
        max_cache_len=args.max_cache_len,
        layer_ranges=layer_ranges,
        modules=modules,
        artifact_paths=artifact_paths,
    )

    print(f"Model: {args.model_id}")
    print(f"Layers: {num_layers}")
    print(f"Num devices: {args.num_devices}")
    print(f"Layer ranges: {layer_ranges}")
    print(f"Max cache len: {args.max_cache_len}")
    print("Artifacts:")
    for artifact in artifact_paths:
        print(f"  {artifact.path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
