# Beacon Coordinator UI

Local browser UI for the native Llama shard demo.

The browser does not run inference. It sends prompts to the persistent native coordinator server, which then routes hidden states through the Android shard bridge workers.

## 1. Start shard bridge workers

Each phone must already be running `beacon_executor_bridge_worker` on port `9100`.

Verify from the Mac:

```bash
nc -vz 10.154.197.31 9100
nc -vz 10.154.197.171 9100
nc -vz 10.154.197.225 9100
```

## 2. Start the persistent coordinator

From the repo root:

```bash
python3 tools/native_executor_llama_coordinator.py \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --route "1=10.154.197.31:9100,2=10.154.197.171:9100,3=10.154.197.225:9100" \
  --listen-port 9301 \
  --default-max-new-tokens 8 \
  --timeout 300 \
  --model-id meta-llama/Llama-3.2-3B-Instruct
```

Wait for:

```text
[native-coordinator] ready
[native-coordinator] listening on http://127.0.0.1:9301
```

## 3. Start the UI

In a second terminal:

```bash
python3 UI/coordinator_ui.py \
  --route "1=10.154.197.31:9100,2=10.154.197.171:9100,3=10.154.197.225:9100" \
  --mode llama \
  --coordinator-url http://127.0.0.1:9301/generate \
  --max-new-tokens 8 \
  --timeout 300 \
  --port 8081
```

Open:

```text
http://127.0.0.1:8081
```
