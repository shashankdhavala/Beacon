#!/usr/bin/env python3
"""
Export GPT-2 or Llama shard artifacts for local multi-device ExecuTorch testing.
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
from transformers import AutoModelForCausalLM
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM

try:
    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        to_edge_transform_and_lower_to_qnn,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
except ImportError:
    QcomChipset = None
    generate_htp_compiler_spec = None
    generate_qnn_executorch_compiler_spec = None
    to_edge_transform_and_lower_to_qnn = None
    convert_pt2e = None
    prepare_pt2e = None

QNN_PT2E_QUANT_CHOICES = ("qnn_16a4w", "qnn_8a8w", "qnn_16a16w")


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


def is_gpt2_model(model) -> bool:
    return isinstance(model, GPT2LMHeadModel)


def is_llama_model(model) -> bool:
    return isinstance(model, LlamaForCausalLM)


def get_hidden_size(model) -> int:
    if is_gpt2_model(model):
        return int(model.config.n_embd)
    if is_llama_model(model):
        return int(model.config.hidden_size)
    raise TypeError(f"Unsupported model type: {type(model)}")


def get_num_layers(model) -> int:
    if is_gpt2_model(model):
        return len(model.transformer.h)
    if is_llama_model(model):
        return len(model.model.layers)
    raise TypeError(f"Unsupported model type: {type(model)}")


class GPT2BlockShard(nn.Module):
    def __init__(
        self,
        full_model: GPT2LMHeadModel,
        start_idx: int,
        end_idx: int,
        max_cache_len: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(full_model.transformer.h[start_idx:end_idx])
        self.layer_ids = [block.attn.layer_idx for block in self.blocks]
        self.max_cache_len = max_cache_len
        self.has_blocks = len(self.blocks) > 0

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

        output = hidden_states.contiguous()
        return (output, *cache_outputs)


class LlamaBlockShard(nn.Module):
    def __init__(
        self,
        full_model: LlamaForCausalLM,
        start_idx: int,
        end_idx: int,
        max_cache_len: int,
    ):
        super().__init__()
        self.core = full_model.model
        self.blocks = nn.ModuleList(self.core.layers[start_idx:end_idx])
        self.layer_ids = [block.self_attn.layer_idx for block in self.blocks]
        self.max_cache_len = max_cache_len
        self.has_blocks = len(self.blocks) > 0

    def _allocate_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        num_kv_heads = self.blocks[0].self_attn.config.num_key_value_heads
        head_dim = self.blocks[0].self_attn.head_dim
        keys = [
            torch.zeros(batch_size, num_kv_heads, self.max_cache_len, head_dim, device=device, dtype=dtype)
            for _ in range(len(self.blocks))
        ]
        values = [
            torch.zeros(batch_size, num_kv_heads, self.max_cache_len, head_dim, device=device, dtype=dtype)
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
        hidden_states, attention_mask, position_ids, cache_position, *past_kv = args

        if self.has_blocks:
            if len(past_kv) == 0:
                keys, values = self._allocate_cache(
                    batch_size=hidden_states.shape[0],
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            else:
                keys = list(past_kv[0::2])
                values = list(past_kv[1::2])
            cache = FixedSizeTensorCache(self.layer_ids, keys, values, cache_position)
            block_attention_mask = prepare_attention_mask(attention_mask, hidden_states.dtype)
            position_embeddings = self.core.rotary_emb(hidden_states, position_ids)
            for block in self.blocks:
                hidden_states = block(
                    hidden_states,
                    attention_mask=block_attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
            cache_outputs = self._flatten_cache(cache)
        else:
            cache_outputs = ()

        output = hidden_states.contiguous()
        return (output, *cache_outputs)


@dataclass
class ShardArtifactPath:
    path: str


@dataclass
class ShardManifest:
    shard_index: int
    artifact_path: str
    architecture: str
    export_backend: str
    start_layer: int
    end_layer: int
    num_layers: int
    num_heads: int
    head_dim: int
    cache_dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GPT-2 ExecuTorch shard artifacts")
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--num-devices", type=int, default=3)
    parser.add_argument("--artifact-dir", default="artifacts/gpt2")
    parser.add_argument("--max-cache-len", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument(
        "--qnn",
        action="store_true",
        help="Export Qualcomm QNN-backed ExecuTorch artifacts for SM8750",
    )
    parser.add_argument(
        "--pt2e-quantize",
        choices=list(QNN_PT2E_QUANT_CHOICES),
        default=None,
        help=(
            "PT2E quantization recipe for QNN HTP (recommended: qnn_16a4w). "
            "Requires --qnn and --dtype float32."
        ),
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def load_model(model_id: str, device: str, dtype_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=parse_torch_dtype(dtype_name),
    )
    if not (is_gpt2_model(model) or is_llama_model(model)):
        raise TypeError(f"Expected a GPT-2 or Llama model, got {type(model)}")
    model.to(device)
    model.eval()
    return model


def ensure_qualcomm_export_available() -> None:
    if (
        QcomChipset is None
        or generate_htp_compiler_spec is None
        or generate_qnn_executorch_compiler_spec is None
        or to_edge_transform_and_lower_to_qnn is None
    ):
        raise ImportError(
            "Qualcomm ExecuTorch export is unavailable. Install Qualcomm backend dependencies, "
            "including `py-cpuinfo`, in the active environment."
        )


def build_shard_modules(
    model,
    layer_ranges: Sequence[Tuple[int, int]],
    max_cache_len: int,
):
    modules = []
    for start, end in layer_ranges:
        if is_gpt2_model(model):
            modules.append(GPT2BlockShard(model, start, end, max_cache_len).eval())
        elif is_llama_model(model):
            modules.append(LlamaBlockShard(model, start, end, max_cache_len).eval())
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
    return modules


def build_export_examples(
    model,
    layer_ranges: Sequence[Tuple[int, int]],
    max_cache_len: int,
    device: str,
) -> Tuple[List[nn.Module], List[tuple[torch.Tensor, ...]]]:
    modules = build_shard_modules(model, layer_ranges, max_cache_len)
    hidden_size = get_hidden_size(model)
    hidden_states = torch.zeros(1, 1, hidden_size, dtype=model.dtype, device=device)
    attention_mask = make_fixed_attention_mask(1, max_cache_len, device)
    position_ids = torch.tensor([[0]], dtype=torch.long, device=device)
    cache_position = torch.tensor([0], dtype=torch.long, device=device)

    args_list: List[tuple[torch.Tensor, ...]] = []
    current = hidden_states
    for module in modules:
        if module.has_blocks:
            if is_gpt2_model(model):
                num_heads = module.blocks[0].attn.num_heads
                head_dim = module.blocks[0].attn.head_dim
            else:
                num_heads = module.blocks[0].self_attn.config.num_key_value_heads
                head_dim = module.blocks[0].self_attn.head_dim
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

        if is_gpt2_model(model):
            shard_args = (current, attention_mask, cache_position, *cache_args)
        else:
            shard_args = (current, attention_mask, position_ids, cache_position, *cache_args)
        args_list.append(shard_args)
        current = module(*shard_args)[0]
    return modules, args_list


def export_to_pte(module: nn.Module, args: tuple, output_path: Path) -> str:
    exported = torch.export.export(module.eval(), args, strict=False)
    executorch_program = to_edge(exported).to_executorch()
    with open(output_path, "wb") as fp:
        executorch_program.write_to_file(fp)
    return str(output_path)


def get_qnn_pt2e_quantizer(pt2e_quantize: str):
    try:
        from executorch.extension.llm.export.quantizer_lib import get_qnn_quantizer
    except ImportError:
        from executorch.backends.qualcomm.quantizer.custom_annotation import (
            custom_annotate_llama_matmul_16a8w,
        )
        from executorch.backends.qualcomm.quantizer.quantizer import QnnQuantizer, QuantDtype
        from torchao.quantization.pt2e import MinMaxObserver

        backend, quant_config = pt2e_quantize.split("_")
        if backend != "qnn":
            raise ValueError(f"Expected qnn_* quant recipe, got {pt2e_quantize}")
        qnn_quantizer = QnnQuantizer()
        custom_annotations = ()
        if quant_config == "8a8w":
            quant_dtype = QuantDtype.use_8a8w
            qnn_quantizer.set_default_quant_config(
                quant_dtype,
                is_qat=False,
                is_conv_per_channel=True,
                is_linear_per_channel=True,
            )
        elif quant_config == "16a16w":
            quant_dtype = QuantDtype.use_16a16w
            qnn_quantizer.set_default_quant_config(
                quant_dtype,
                is_qat=False,
                is_conv_per_channel=False,
                is_linear_per_channel=False,
                act_observer=MinMaxObserver,
            )
        elif quant_config == "16a4w":
            quant_dtype = QuantDtype.use_16a4w
            qnn_quantizer.set_default_quant_config(
                quant_dtype,
                is_qat=False,
                is_conv_per_channel=True,
                is_linear_per_channel=True,
                act_observer=MinMaxObserver,
            )
            custom_annotations = (custom_annotate_llama_matmul_16a8w,)
        else:
            raise ValueError(f"Unsupported QNN quant recipe: {pt2e_quantize}")
        qnn_quantizer.add_custom_quant_annotations(custom_annotations)
        return qnn_quantizer, quant_dtype

    quantizer, quant_dtype = get_qnn_quantizer(pt2e_quantize)
    return quantizer, quant_dtype


def quantize_module_for_qnn(module: nn.Module, args: tuple, pt2e_quantize: str) -> nn.Module:
    if prepare_pt2e is None or convert_pt2e is None:
        raise ImportError(
            "PT2E quantization requires torchao. Install Qualcomm ExecuTorch backend dependencies."
        )
    quantizer, _ = get_qnn_pt2e_quantizer(pt2e_quantize)
    exported = torch.export.export(module.eval(), args, strict=False)
    prepared = prepare_pt2e(exported.module(), quantizer)
    prepared(*args)
    return convert_pt2e(prepared)


def export_to_qnn_pte(
    module: nn.Module,
    args: tuple,
    output_path: Path,
    pt2e_quantize: str | None = None,
    use_fp16: bool = False,
) -> str:
    ensure_qualcomm_export_available()

    export_module = module.eval()
    if pt2e_quantize is not None:
        print(f"Quantizing shard with PT2E recipe: {pt2e_quantize}")
        export_module = quantize_module_for_qnn(export_module, args, pt2e_quantize)

    backend_options = generate_htp_compiler_spec(use_fp16=use_fp16)
    compile_spec = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8750,
        backend_options=backend_options,
    )
    program = to_edge_transform_and_lower_to_qnn(export_module, args, compile_spec).to_executorch()
    with open(output_path, "wb") as fp:
        fp.write(program.buffer)
    return str(output_path)


def export_artifacts(
    modules: Sequence[nn.Module],
    args_list: Sequence[tuple[torch.Tensor, ...]],
    artifact_dir: Path,
    use_qnn: bool,
    pt2e_quantize: str | None = None,
    use_fp16: bool = False,
) -> List[ShardArtifactPath]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths: List[ShardArtifactPath] = []
    for idx, (module, args) in enumerate(zip(modules, args_list)):
        output_path = (artifact_dir / f"shard_{idx}.pte").resolve()
        if use_qnn:
            print(f"Exporting QNN shard {idx}...")
            path = export_to_qnn_pte(
                module,
                args,
                output_path,
                pt2e_quantize=pt2e_quantize,
                use_fp16=use_fp16,
            )
        else:
            path = export_to_pte(module, args, output_path)
        paths.append(ShardArtifactPath(path=path))
    return paths


def write_manifest(
    artifact_dir: Path,
    model_id: str,
    num_devices: int,
    max_cache_len: int,
    model_dtype: torch.dtype,
    architecture: str,
    export_backend: str,
    layer_ranges: Sequence[Tuple[int, int]],
    modules: Sequence[nn.Module],
    artifact_paths: Sequence[ShardArtifactPath],
    pt2e_quantize: str | None = None,
) -> Path:
    shards: List[dict] = []
    for idx, (layer_range, module, artifact_path) in enumerate(zip(layer_ranges, modules, artifact_paths)):
        if module.has_blocks:
            if architecture == "gpt2":
                num_heads = module.blocks[0].attn.num_heads
                head_dim = module.blocks[0].attn.head_dim
            else:
                num_heads = module.blocks[0].self_attn.config.num_key_value_heads
                head_dim = module.blocks[0].self_attn.head_dim
        else:
            num_heads = 0
            head_dim = 0
        cache_dtype = (
            str(next(module.parameters()).dtype).replace("torch.", "")
            if any(True for _ in module.parameters())
            else str(model_dtype).replace("torch.", "")
        )
        shards.append(
            ShardManifest(
                shard_index=idx,
                artifact_path=artifact_path.path,
                architecture=architecture,
                export_backend=export_backend,
                start_layer=layer_range[0],
                end_layer=layer_range[1],
                num_layers=layer_range[1] - layer_range[0],
                num_heads=num_heads,
                head_dim=head_dim,
                cache_dtype=cache_dtype,
            ).__dict__
        )

    manifest = {
        "model_id": model_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "num_devices": num_devices,
        "max_cache_len": max_cache_len,
        "export_backend": export_backend,
        "qualcomm_chipset": "SM8750" if export_backend == "qnn:SM8750" else None,
        "pt2e_quantize": pt2e_quantize,
        "layer_ranges": [list(r) for r in layer_ranges],
        "shards": shards,
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)

    if args.qnn:
        ensure_qualcomm_export_available()

    if args.pt2e_quantize and not args.qnn:
        raise ValueError("--pt2e-quantize requires --qnn")
    if args.pt2e_quantize and args.dtype != "float32":
        raise ValueError(
            f"--pt2e-quantize {args.pt2e_quantize} requires --dtype float32 "
            "(quantization replaces fp16 lowering)"
        )
    if args.qnn and not args.pt2e_quantize and args.dtype == "float16":
        print(
            "WARNING: --qnn with --dtype float16 often causes QNN dtype mismatches "
            "(float32 activations vs float16 weights). Prefer --pt2e-quantize qnn_16a4w "
            "with --dtype float32, or use --dtype float32 without quantization."
        )

    model = load_model(args.model_id, args.device, args.dtype)
    architecture = "gpt2" if is_gpt2_model(model) else "llama"
    num_layers = get_num_layers(model)
    if args.num_devices < 2:
        raise ValueError("num_devices must be at least 2")

    use_fp16 = args.qnn and args.pt2e_quantize is None and args.dtype == "float16"
    layer_ranges = compute_layer_ranges(num_layers, args.num_devices)
    modules, args_list = build_export_examples(
        model=model,
        layer_ranges=layer_ranges,
        max_cache_len=args.max_cache_len,
        device=args.device,
    )
    export_backend = "qnn:SM8750" if args.qnn else "portable"
    artifact_paths = export_artifacts(
        modules,
        args_list,
        Path(args.artifact_dir),
        args.qnn,
        pt2e_quantize=args.pt2e_quantize,
        use_fp16=use_fp16,
    )
    manifest_path = write_manifest(
        artifact_dir=Path(args.artifact_dir),
        model_id=args.model_id,
        num_devices=args.num_devices,
        max_cache_len=args.max_cache_len,
        model_dtype=model.dtype,
        architecture=architecture,
        export_backend=export_backend,
        layer_ranges=layer_ranges,
        modules=modules,
        artifact_paths=artifact_paths,
        pt2e_quantize=args.pt2e_quantize,
    )

    print(f"Model: {args.model_id}")
    print(f"Layers: {num_layers}")
    print(f"Num devices: {args.num_devices}")
    print(f"Layer ranges: {layer_ranges}")
    print(f"Max cache len: {args.max_cache_len}")
    print(f"Export backend: {export_backend}")
    if args.pt2e_quantize:
        print(f"PT2E quantization: {args.pt2e_quantize}")
    print("Artifacts:")
    for artifact in artifact_paths:
        print(f"  {artifact.path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
