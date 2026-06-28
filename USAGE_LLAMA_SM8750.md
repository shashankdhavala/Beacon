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
- For QNN HTP export, use `--pt2e-quantize qnn_16a4w` with `--dtype float32` (not `--dtype float16`).
- fp16 lowering causes QNN dtype mismatches (float32 activations vs float16 weights) and is not HTP-native.

## 5. Export 3 Qualcomm shard artifacts for SM8750

Recommended (16-bit activations, 4-bit weights — fits on-device NPU):

```bash
python scripts/export_shards.py \
  --model-id models/meta-llama/Llama-3.2-3B \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --num-devices 3 \
  --max-cache-len 32 \
  --dtype float32 \
  --pt2e-quantize qnn_16a4w \
  --qnn \
  --device cpu
```

Fallback (fp32, no quantization — compiles but shards are ~3× larger):

```bash
python scripts/export_shards.py \
  --model-id models/meta-llama/Llama-3.2-3B \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --num-devices 3 \
  --max-cache-len 32 \
  --dtype float32 \
  --qnn \
  --device cpu
```

What this does:
- loads the local Llama 3.2 3B checkpoint in float32
- builds realistic calibration activations by running representative tokens through the fp32 shard pipeline
- applies PT2E `qnn_16a4w` quantization per shard (HTP-native int kernels), calibrating observers on those activations
- splits transformer layers across 3 shards
- exports each shard as a Qualcomm QNN-backed ExecuTorch `.pte`
- writes a `manifest.json` describing the shard layout

Calibration controls (optional):
- `--calib-sequences N` — number of calibration token sequences (default 4)
- `--calib-seq-len T` — tokens per sequence, capped at `--max-cache-len` (default 16)

Calibration matters: PT2E observers must see real activation ranges. Calibrating on
all-zero tensors produces degenerate (near-zero) quantization scales and broken output.
Raise `--calib-sequences` / `--calib-seq-len` for better scale estimates at the cost of
longer export time.

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
- `pt2e_quantize` is `qnn_16a4w` (when using the recommended command)
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
  --dtype float32 \
  --pt2e-quantize qnn_16a4w \
  --qnn \
  --device cpu
```

## 9. Run the persistent master + shard simulation

Start shard workers in three terminals:

```bash
python scripts/run_shard_worker.py \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --shard-index 2 \
  --port 9102
```

```bash
python scripts/run_shard_worker.py \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --shard-index 1 \
  --port 9101 \
  --downstream-host 127.0.0.1 \
  --downstream-port 9102
```

```bash
python scripts/run_shard_worker.py \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --shard-index 0 \
  --port 9100 \
  --downstream-host 127.0.0.1 \
  --downstream-port 9101
```

Start the master once in a fourth terminal:

```bash
python scripts/run_master.py \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --shard-port 9100 \
  --listen-port 9200 \
  --default-max-new-tokens 8 \
  --dtype float32 \
  --device cpu
```

Then send prompts to the already-running master:

```bash
curl -X POST http://127.0.0.1:9200/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Give me one short sentence about offline medical triage.","max_new_tokens":8}'
```

Health check:

```bash
curl http://127.0.0.1:9200/health
```

Behavior:
- the master loads tokenizer + model once at startup
- the master keeps the shard_0 socket open across requests
- each request still gets a fresh shard-local KV lifecycle via `start_request` / `flush_request`
