# Llama Native Distributed Demo Runbook

This runbook brings up the working 3-phone Llama demo:

```text
Mac coordinator -> Phone 1 shard -> Phone 2 shard -> Phone 3 shard -> Mac head/decode
```

The demo uses the native `beacon_executor_bridge_worker` on port `9100`. The Android APK/app port `9000` is for the earlier networking MVP and is not used for this Llama QNN path.

## Known Device Map

```text
Shard 1: R3CXC07ZXZB   10.154.197.31   shard_0.pte   10 layers
Shard 2: R3CXC0801DZ   10.154.197.171  shard_1.pte    9 layers
Shard 3: R3CXC0804CA   10.154.197.225  shard_2.pte    9 layers
```

All files live on-device under:

```text
/data/local/tmp/beacon_et
```

## What Runs Where

Mac:

```text
tools/native_executor_llama_coordinator.py
```

Phones:

```text
/data/local/tmp/beacon_et/beacon_executor_bridge_worker
/data/local/tmp/beacon_et/executor_runner
/data/local/tmp/beacon_et/shard_*.pte
```

The Mac handles tokenization, embeddings, final norm/head, and decoding. Each phone runs one ExecuTorch/QNN shard and returns the next hidden-state tensor.

## 1. Put Everything On The Same Network

Connect the Mac and all phones to the same Android hotspot/WiFi. Avoid iPhone hotspot for the demo if it shows peer-to-peer weirdness.

Check the Mac WiFi IP:

```bash
ipconfig getifaddr en0
```

Check phone IPs:

```bash
adb -s R3CXC07ZXZB shell "ip -f inet addr show wlan0"
adb -s R3CXC0801DZ shell "ip -f inet addr show wlan0"
adb -s R3CXC0804CA shell "ip -f inet addr show wlan0"
```

If the IPs changed, update the coordinator route.

## 2. Verify Files On Each Phone

From the repo root:

```bash
cd /Users/shashankdhavala/Desktop/Exec/Beacon
```

Check shard 1:

```bash
adb -s R3CXC07ZXZB shell "ls -lh /data/local/tmp/beacon_et/shard_0.pte /data/local/tmp/beacon_et/executor_runner /data/local/tmp/beacon_et/beacon_executor_bridge_worker"
```

Check shard 2:

```bash
adb -s R3CXC0801DZ shell "ls -lh /data/local/tmp/beacon_et/shard_1.pte /data/local/tmp/beacon_et/executor_runner /data/local/tmp/beacon_et/beacon_executor_bridge_worker"
```

Check shard 3:

```bash
adb -s R3CXC0804CA shell "ls -lh /data/local/tmp/beacon_et/shard_2.pte /data/local/tmp/beacon_et/executor_runner /data/local/tmp/beacon_et/beacon_executor_bridge_worker"
```

If the bridge binary is missing or stale, push the local rebuilt one:

```bash
for SERIAL in R3CXC07ZXZB R3CXC0801DZ R3CXC0804CA; do
  adb -s $SERIAL push native/out/beacon_executor_bridge_worker /data/local/tmp/beacon_et/
  adb -s $SERIAL shell "chmod +x /data/local/tmp/beacon_et/beacon_executor_bridge_worker"
done
```

## 3. Start Native Bridge Workers

Use `setsid`, not just `nohup`. `setsid` starts the bridge in a detached session so it is less likely to die when USB/ADB disconnects.

Start shard 1:

```bash
adb -s R3CXC07ZXZB shell "cd /data/local/tmp/beacon_et; rm -f bridge_1.log; setsid ./beacon_executor_bridge_worker --model_path ./shard_0.pte --shard_id 1 --num_layers 10 --port 9100 > bridge_1.log 2>&1 < /dev/null &"
```

Start shard 2:

```bash
adb -s R3CXC0801DZ shell "cd /data/local/tmp/beacon_et; rm -f bridge_2.log; setsid ./beacon_executor_bridge_worker --model_path ./shard_1.pte --shard_id 2 --num_layers 9 --port 9100 > bridge_2.log 2>&1 < /dev/null &"
```

Start shard 3:

```bash
adb -s R3CXC0804CA shell "cd /data/local/tmp/beacon_et; rm -f bridge_3.log; setsid ./beacon_executor_bridge_worker --model_path ./shard_2.pte --shard_id 3 --num_layers 9 --port 9100 > bridge_3.log 2>&1 < /dev/null &"
```

Verify logs:

```bash
adb -s R3CXC07ZXZB shell "cat /data/local/tmp/beacon_et/bridge_1.log"
adb -s R3CXC0801DZ shell "cat /data/local/tmp/beacon_et/bridge_2.log"
adb -s R3CXC0804CA shell "cat /data/local/tmp/beacon_et/bridge_3.log"
```

Expected:

```text
beacon executor bridge listening on port 9100
```

## 4. Verify Ports Before The Demo

While USB is still connected:

```bash
nc -vz 10.154.197.31 9100
nc -vz 10.154.197.171 9100
nc -vz 10.154.197.225 9100
```

All three should say `succeeded`.

Now unplug USB cables and run the same check again:

```bash
nc -vz 10.154.197.31 9100
nc -vz 10.154.197.171 9100
nc -vz 10.154.197.225 9100
```

If all three still succeed, the demo is truly wireless.

## 5. Run The Llama Coordinator

Use this exact command:

```bash
python3 tools/native_executor_llama_coordinator.py \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --route "1=10.154.197.31:9100,2=10.154.197.171:9100,3=10.154.197.225:9100" \
  --prompt "First aid for a minor burn:" \
  --max-new-tokens 8 \
  --timeout 300 \
  --model-id meta-llama/Llama-3.2-3B-Instruct
```

Expected output shape:

```text
step=0 phase=prefill ...
...
Output:
First aid for a minor burn: Apply a cool compress to the affected area
```

Do not leave trailing spaces after a line-continuation backslash.

## What The Demo Proves

For every token:

```text
Mac creates [1, 1, 3072] hidden state
Phone 1 runs shard_0.pte
Phone 2 runs shard_1.pte
Phone 3 runs shard_2.pte
Mac applies final norm/head and decodes next token
```

Each hop sends `12288` bytes of float32 hidden state:

```text
1 * 1 * 3072 * 4 = 12288 bytes
```

USB is only used for setup/process control. The inference tensor traffic uses WiFi/hotspot.

## Edge Cases And Fixes

### `Connection refused`

Meaning:

```text
Phone is reachable, but nothing is listening on port 9100.
```

Fix: restart the bridge worker for that shard with `setsid`.

Example for shard 2:

```bash
adb -s R3CXC0801DZ shell "cd /data/local/tmp/beacon_et; rm -f bridge_2.log; setsid ./beacon_executor_bridge_worker --model_path ./shard_1.pte --shard_id 2 --num_layers 9 --port 9100 > bridge_2.log 2>&1 < /dev/null &"
```

Then verify:

```bash
adb -s R3CXC0801DZ shell "cat /data/local/tmp/beacon_et/bridge_2.log"
nc -vz 10.154.197.171 9100
```

### `Operation timed out`

Meaning:

```text
Mac cannot reach the phone over the network.
```

Fix:

```bash
ping 10.154.197.31
ipconfig getifaddr en0
adb -s R3CXC07ZXZB shell "ip -f inet addr show wlan0"
```

Make sure Mac and phones are on the same hotspot/WiFi. Some networks block client-to-client traffic; use the Android hotspot that worked.

### Phone Rebooted

Files usually remain in `/data/local/tmp/beacon_et`, but the bridge process is gone.

Fix: rerun the `setsid` bridge start command for that phone.

### USB Disconnect Kills A Worker

Symptom:

```text
nc succeeds while wired, then Connection refused after unplug.
```

Fix: use `setsid` instead of a plain background process. We saw this on device 2.

### `nohup` Did Not Work

Use `setsid`. `nohup` was not enough on one device because the worker still died with the ADB shell/session.

### `CANNOT LINK EXECUTABLE ... libc++_shared.so not found`

Meaning:

```text
The phone has an old/non-static bridge binary.
```

Fix: rebuild/push the static bridge:

```bash
cd /Users/shashankdhavala/Desktop/Exec/Beacon
export ANDROID_NDK=/Users/shashankdhavala/Library/Android/sdk/ndk/27.2.12479018
./native/build_executor_bridge_worker.sh

for SERIAL in R3CXC07ZXZB R3CXC0801DZ R3CXC0804CA; do
  adb -s $SERIAL push native/out/beacon_executor_bridge_worker /data/local/tmp/beacon_et/
  adb -s $SERIAL shell "chmod +x /data/local/tmp/beacon_et/beacon_executor_bridge_worker"
done
```

### `bind: Address already in use`

Meaning:

```text
Another bridge process is already listening on 9100.
```

Check PID:

```bash
adb -s R3CXC0801DZ shell "pidof beacon_executor_bridge_worker"
```

If needed, kill the PID and restart:

```bash
adb -s R3CXC0801DZ shell "kill <PID>"
```

### `adb: device ... not found`

Meaning:

```text
That phone is not currently visible over USB/ADB.
```

Fix:

```bash
adb devices
```

Then reconnect USB, unlock the phone, and accept the USB debugging prompt if it says `unauthorized`.

### Hugging Face Access Error

The coordinator loads tokenizer/model metadata from `meta-llama/Llama-3.2-3B-Instruct`. Make sure the account has gated model access and is logged in:

```bash
python3 -c "from huggingface_hub import whoami; print(whoami())"
```

If needed, log out and log in with the right account:

```bash
hf auth logout
hf auth login
```

### Coordinator Loads Model, Then Times Out

This means Python/model setup succeeded, but a shard connection failed.

Run:

```bash
nc -vz 10.154.197.31 9100
nc -vz 10.154.197.171 9100
nc -vz 10.154.197.225 9100
```

Fix the first shard that fails.

## Clean Restart Checklist

Use this when the demo state feels messy:

```bash
cd /Users/shashankdhavala/Desktop/Exec/Beacon
```

1. Connect USB to all phones.
2. Confirm `adb devices` shows all three.
3. Confirm phone IPs.
4. Start all three workers with `setsid`.
5. Check all three logs.
6. Run all three `nc` checks.
7. Unplug USB.
8. Run all three `nc` checks again.
9. Run `tools/native_executor_llama_coordinator.py`.

## Good Final Demo Command

```bash
python3 tools/native_executor_llama_coordinator.py \
  --artifact-dir artifacts/llama32_3b_sm8750_3way \
  --route "1=10.154.197.31:9100,2=10.154.197.171:9100,3=10.154.197.225:9100" \
  --prompt "First aid for a minor burn:" \
  --max-new-tokens 8 \
  --timeout 300 \
  --model-id meta-llama/Llama-3.2-3B-Instruct
```
