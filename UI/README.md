# Beacon Coordinator UI

Local browser UI for the coordinator demo. The browser only takes user text; the shard route is configured when starting the server.

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
