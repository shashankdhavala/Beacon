# Beacon Networking MVP Runbook

This is the first MacBook-coordinator to Android-worker test.

## What Runs Where

### MacBook

Runs:

```text
tools/mac_coordinator.py
```

Role:

```text
Coordinator / Phone A stand-in
TCP client
Sends fake activation tensor
Receives result tensor
Logs round-trip latency and checksum
```

### Android Phone

Runs:

```text
android-worker-mvp
```

Role:

```text
Phone B worker
TCP server
Receives tensor
Echoes tensor back as RESULT
Logs shape, dtype, bytes, checksum
```

## Protocol

Every tensor message is:

```text
4 bytes: big-endian JSON header length
N bytes: UTF-8 JSON header
M bytes: raw tensor bytes
```

The JSON header includes:

```json
{
  "messageType": "TENSOR",
  "requestId": "mac-123",
  "step": 0,
  "sourceShard": 0,
  "targetShard": 1,
  "shape": [1, 64, 768],
  "dtype": "fp16",
  "byteLength": 98304,
  "sha256": "..."
}
```

## How To Run

### 1. Put MacBook and phone on the same WiFi

For the first test, use normal WiFi. Offline hotspot can come later.

### 2. Open the Android app

Open `android-worker-mvp` in Android Studio, connect the phone with USB debugging, then run the app.

Tap:

```text
Start Worker Server
```

The app will show the phone IP and port, for example:

```text
Phone IP: 192.168.1.42
Port: 9000
```

### 3. Run the Mac coordinator

From the repo root:

```bash
python3 tools/mac_coordinator.py --host 192.168.1.42 --port 9000
```

Replace the IP with the one shown in the Android app.

Expected Mac output:

```text
Connecting to Android worker at 192.168.1.42:9000 ...
Sending tensor request=mac-... shape=[1, 64, 768] dtype=fp16 bytes=98304 sha256=...
Received response:
{
  "messageType": "RESULT",
  ...
}
round_trip_ms=...
```

Expected Android log:

```text
Worker listening on port 9000
Client connected: 192.168.1.x:...
Received type=TENSOR request=... shape=[1, 64, 768] dtype=fp16 bytes=98304 sha256=...
Sent type=RESULT request=...
```

## Bigger Tensor Test

To mimic the README Llama 3.2 3B activation shape `[1, S, 3200]`:

```bash
python3 tools/mac_coordinator.py --host 192.168.1.42 --port 9000 --seq-len 128 --hidden-size 3200
```

For fp16, this sends:

```text
1 * 128 * 3200 * 2 = 819200 bytes
```

## Two-Phone Text Route Test

Use this when the Mac and both phones are connected to the same hotspot.

### Phone 2, the last mobile node

Start this one first.

```text
Port: 9000
Append suffix: 2
Next phone IP: leave blank
Next phone port: 9000
```

Tap:

```text
Start Worker Server
```

Note Phone 2's `wlan0` IP address from the app.

### Phone 1, the first mobile node

Configure Phone 1 to point at Phone 2.

```text
Port: 9000
Append suffix: 1
Next phone IP: <Phone 2 wlan0 IP>
Next phone port: 9000
```

Tap:

```text
Start Worker Server
```

Note Phone 1's `wlan0` IP address from the app.

### MacBook

Send `hello` to Phone 1:

```bash
python3 tools/mac_coordinator.py --host <Phone 1 wlan0 IP> --port 9000 --message hello
```

Expected Mac output includes:

```text
response text='hello12'
```

Route:

```text
Mac -> Phone 1 appends 1 -> Phone 2 appends 2 -> Phone 1 -> Mac
```

## Troubleshooting

If the Mac cannot connect:

- Make sure both devices are on the same WiFi.
- Keep the Android app open in the foreground.
- Try port `9001` in the app and Mac command.
- Check that no VPN/firewall is isolating local network devices.
- On some networks, client isolation blocks phone-to-laptop traffic. Use a phone hotspot or another router.

## USB / ADB Forwarding Fallback

If the phone is connected to the Mac with USB debugging, you can avoid WiFi routing completely.

1. Start the Android worker server in the app.
2. On the Mac, run:

```bash
adb forward tcp:9000 tcp:9000
```

3. Then connect to localhost:

```bash
python3 tools/mac_coordinator.py --host 127.0.0.1 --port 9000
```

This maps Mac port `9000` to Android device port `9000`.

## Next Integration Step

Replace the echo block in:

```text
android-worker-mvp/app/src/main/java/com/beacon/workermvp/ActivationServer.kt
```

with:

```text
ExecuTorch shard B run(inputTensor) -> outputTensor
```

The wire protocol can stay the same.
