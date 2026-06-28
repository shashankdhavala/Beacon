# Shard Llama 3.2 3B for Qualcomm SM8750

This document covers the exact local flow to export `meta-llama/Llama-3.2-3B` into 3 ExecuTorch shard artifacts targeting Qualcomm `SM8750`.

Assumptions:
- You are in the repo root: `Beacon`
- You are using the project venv: `.venv`
- You have already been granted access to `meta-llama/Llama-3.2-3B` on Hugging Face
- You want 3 shards, one per Galaxy S25 Ultra

## 1. Activate the environment

```bash
source .venv/bin/activate
```

## 2. Install the required Python packages

`export_shards.py` needs the Qualcomm ExecuTorch backend and `py-cpuinfo`.

```bash
pip install -r requirements.txt
pip install py-cpuinfo huggingface_hub
```

## 3. Log in to Hugging Face

```bash
huggingface-cli login
```

## 4. Download Llama 3.2 3B locally

Use this command to download the model repository before sharding:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B \
  --local-dir models/meta-llama/Llama-3.2-3B
```

Notes:
- The download step fetches the model files from Hugging Face.
- The fp16 choice is applied during export with `--dtype float16`.
- You do not need a separate “fp16 download” command for this workflow.

## 5. Export 3 Qualcomm shard artifacts for SM8750

Run:

```bash
python scripts/export_shards.py \
  --model-id models/meta-llama/Llama-3.2-3B \
  --artifact-dir artifacts/llama32_3b_sm8750 \
  --num-devices 3 \
  --max-cache-len 32 \
  --dtype float16 \
  --qnn \
  --device cpu
```

What this does:
- loads the local Llama 3.2 3B checkpoint
- splits transformer layers across 3 shards
- exports each shard as a Qualcomm QNN-backed ExecuTorch `.pte`
- writes a `manifest.json` describing the shard layout

Expected outputs:

```text
artifacts/llama32_3b_sm8750_3way/
  shard_0.pte
  shard_1.pte
  shard_2.pte
  manifest.json
```

## 6. What to verify after export

Check that the files exist:

```bash
ls -lh artifacts/llama32_3b_sm8750_3way
```

Inspect the manifest:

```bash
cat artifacts/llama32_3b_sm8750_3way/manifest.json
```

You should verify:
- `model_id` points to `models/meta-llama/Llama-3.2-3B`
- `num_devices` is `3`
- `export_backend` is `qnn:SM8750`
- there are 3 shard entries

## 7. Common failure modes

### `ImportError: ... install py-cpuinfo`

Install it in the active venv:

```bash
pip install py-cpuinfo
```

### Hugging Face access error

You either are not logged in or do not have access to the model:

```bash
huggingface-cli whoami
```

Then re-run:

```bash
huggingface-cli login
```

### Export fails due to memory pressure

Llama 3.2 3B is materially larger than `tiny-gpt2`. Close other processes and keep export on CPU. If needed, reduce parallel activity on the machine before retrying.

## 8. Run command summary

Download:

```bash
huggingface-cli download meta-llama/Llama-3.2-3B \
  --local-dir models/meta-llama/Llama-3.2-3B
```

Shard for Qualcomm SM8750:

```bash
python scripts/export_shards.py \
  --model-id models/meta-llama/Llama-3.2-3B \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --num-devices 3 \
  --max-cache-len 32 \
  --dtype float16 \
  --qnn \
  --device cpu
```
