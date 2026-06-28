#!/usr/bin/env python3
"""
Create and optionally push the manifest needed by Beacon's Android worker app
for pre-deployed QNN Llama shard directories.

This does not push large .pte files or QNN libraries. It assumes those are
already present on the phone, for example in /data/local/tmp/beacon_et.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


QNN_RUNTIME_DEPS: dict[str, tuple[str, ...]] = {
    "libc++.so": (
        "/system/lib64/libc++.so",
        "/vendor/lib64/libc++.so",
        "/apex/com.android.runtime/lib64/libc++.so",
    ),
    "libbase.so": (
        "/system/lib64/libbase.so",
        "/vendor/lib64/libbase.so",
        "/apex/com.android.runtime/lib64/bionic/libbase.so",
    ),
    "libdl_android.so": (
        "/system/lib64/libdl_android.so",
        "/vendor/lib64/libdl_android.so",
        "/apex/com.android.runtime/lib64/bionic/libdl_android.so",
    ),
    "libcutils.so": (
        "/system/lib64/libcutils.so",
        "/vendor/lib64/libcutils.so",
        "/apex/com.android.vndk.current/lib64/libcutils.so",
    ),
    "libutils.so": (
        "/system/lib64/libutils.so",
        "/vendor/lib64/libutils.so",
        "/apex/com.android.vndk.current/lib64/libutils.so",
    ),
    "libvndksupport.so": (
        "/system/lib64/libvndksupport.so",
        "/apex/com.android.vndk.current/lib64/libvndksupport.so",
    ),
    "libhidlbase.so": (
        "/system/lib64/libhidlbase.so",
        "/vendor/lib64/libhidlbase.so",
        "/apex/com.android.vndk.current/lib64/libhidlbase.so",
    ),
    "libhardware.so": (
        "/system/lib64/libhardware.so",
        "/vendor/lib64/libhardware.so",
        "/apex/com.android.vndk.current/lib64/libhardware.so",
    ),
    "libbinder_ndk.so": (
        "/system/lib64/libbinder_ndk.so",
        "/apex/com.android.vndk.current/lib64/libbinder_ndk.so",
    ),
    "libbinder.so": (
        "/system/lib64/libbinder.so",
        "/vendor/lib64/libbinder.so",
        "/apex/com.android.vndk.current/lib64/libbinder.so",
    ),
    "libdmabufheap.so": (
        "/system/lib64/libdmabufheap.so",
        "/vendor/lib64/libdmabufheap.so",
        "/apex/com.android.vndk.current/lib64/libdmabufheap.so",
    ),
    "libvmmem.so": (
        "/vendor/lib64/libvmmem.so",
        "/system/lib64/libvmmem.so",
    ),
    "vendor.qti.hardware.dsp-V1-ndk.so": (
        "/vendor/lib64/vendor.qti.hardware.dsp-V1-ndk.so",
        "/system/lib64/vendor.qti.hardware.dsp-V1-ndk.so",
    ),
    "android.hardware.common-V2-ndk.so": (
        "/system/lib64/android.hardware.common-V2-ndk.so",
        "/vendor/lib64/android.hardware.common-V2-ndk.so",
        "/apex/com.android.vndk.current/lib64/android.hardware.common-V2-ndk.so",
    ),
    "libcdsprpc.so": (
        "/vendor/lib64/libcdsprpc.so",
        "/system/vendor/lib64/libcdsprpc.so",
    ),
}


def compute_layer_ranges(num_layers: int, num_shards: int) -> list[tuple[int, int]]:
    base = num_layers // num_shards
    remainder = num_layers % num_shards
    ranges: list[tuple[int, int]] = []
    start = 0
    for idx in range(num_shards):
        count = base + (1 if idx < remainder else 0)
        end = start + count
        ranges.append((start, end))
        start = end
    return ranges


def build_manifest(args: argparse.Namespace) -> dict:
    layer_ranges = compute_layer_ranges(args.num_layers, args.num_devices)
    shards = []
    for idx, (start, end) in enumerate(layer_ranges):
        shards.append(
            {
                "shard_index": idx,
                "artifact_path": f"shard_{idx}.pte",
                "architecture": "llama",
                "export_backend": "qnn:SM8750",
                "start_layer": start,
                "end_layer": end,
                "num_layers": end - start,
                "num_heads": args.num_kv_heads,
                "head_dim": args.head_dim,
                "cache_dtype": args.cache_dtype,
            }
        )

    return {
        "model_id": args.model_id,
        "architecture": "llama",
        "artifact_dir": args.device_dir,
        "num_devices": args.num_devices,
        "hidden_size": args.hidden_size,
        "max_cache_len": args.max_cache_len,
        "export_backend": "qnn:SM8750",
        "qualcomm_chipset": "SM8750",
        "pt2e_quantize": args.pt2e_quantize,
        "layer_ranges": [list(item) for item in layer_ranges],
        "shards": shards,
    }


def adb(args: argparse.Namespace, *parts: str) -> None:
    command = ["adb"]
    if args.serial:
        command.extend(["-s", args.serial])
    command.extend(parts)
    subprocess.run(command, check=True)


def adb_shell(args: argparse.Namespace, script: str) -> None:
    adb(args, "shell", script)


def copy_qnn_runtime_deps(args: argparse.Namespace) -> None:
    for library, candidates in QNN_RUNTIME_DEPS.items():
        quoted_candidates = " ".join(f"'{candidate}'" for candidate in candidates)
        script = (
            f"set -e; mkdir -p '{args.device_dir}'; "
            f"found=''; "
            f"for src in {quoted_candidates}; do "
            f"if [ -f \"$src\" ]; then cp \"$src\" '{args.device_dir}/{library}'; found=\"$src\"; break; fi; "
            f"done; "
            f"if [ -z \"$found\" ]; then echo 'missing {library}' >&2; exit 1; fi"
        )
        adb_shell(args, script)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Beacon QNN Llama manifest deployment.")
    parser.add_argument("--serial", help="Optional adb serial. If present, push manifest to this device.")
    parser.add_argument("--device-dir", default="/data/local/tmp/beacon_et")
    parser.add_argument("--out", default=None, help="Optional local manifest path. Default: temporary file when pushing.")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--num-devices", type=int, default=3)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--hidden-size", type=int, default=3072)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--max-cache-len", type=int, default=32)
    parser.add_argument("--cache-dtype", default="float32")
    parser.add_argument("--pt2e-quantize", default="qnn_16a4w")
    parser.add_argument(
        "--skip-runtime-deps",
        action="store_true",
        help="Only push manifest; do not copy Android/QNN transitive runtime dependencies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)

    if args.out:
        manifest_path = Path(args.out)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        temp = tempfile.NamedTemporaryFile("w", suffix="-manifest.json", delete=False)
        with temp:
            temp.write(json.dumps(manifest, indent=2) + "\n")
        manifest_path = Path(temp.name)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Layer ranges: {manifest['layer_ranges']}")

    if not args.serial:
        return

    adb(args, "shell", f"mkdir -p {args.device_dir}")
    adb(args, "push", str(manifest_path), f"{args.device_dir}/manifest.json")
    if not args.skip_runtime_deps:
        copy_qnn_runtime_deps(args)
    adb(args, "shell", f"chmod 755 {args.device_dir} && chmod 644 {args.device_dir}/manifest.json {args.device_dir}/shard_*.pte")
    adb(args, "shell", f"ls -lh {args.device_dir}/manifest.json {args.device_dir}/shard_*.pte {args.device_dir}/lib*.so {args.device_dir}/vendor.qti.hardware.dsp-V1-ndk.so")


if __name__ == "__main__":
    main()
