# Beacon Coordinator UI

Local browser UI for the coordinator demo. The browser only takes user text; the shard route is configured when starting the server.

For Llama mode, the UI does not run inference itself. It forwards prompts to the
persistent coordinator HTTP service in `tools/mac_coordinator.py --llama-service`.

Run from the repo root:

```bash
python3 UI/coordinator_ui.py \
  --route "1=10.154.197.31:9000,2=10.154.197.225:9000" \
  --port 8081
```

Then open:

```text
http://127.0.0.1:8081
```

## Llama mode

Start the persistent coordinator first:

```bash
python3 tools/mac_coordinator.py \
  --llama-service \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --route "1=10.154.197.31:9000,2=10.154.197.225:9000,3=10.154.197.226:9000" \
  --listen-port 9300 \
  --default-max-new-tokens 8 \
  --llama-dtype float32 \
  --device cpu
```

Then start the UI in Llama mode:

```bash
python3 UI/coordinator_ui.py \
  --route "1=10.154.197.31:9000,2=10.154.197.225:9000,3=10.154.197.226:9000" \
  --mode llama \
  --coordinator-url http://127.0.0.1:9300/generate \
  --port 8081
```
