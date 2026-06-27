# ClassMesh Team Task Split and Integration Plan

## Project Direction

We are building:

> **Aggregated model-parallel serving over a P2P phone pipeline using ExecuTorch.**

This means a single student query is served by multiple nearby phones. The model is split layer-wise, each phone owns one shard, and activations flow through the phone pipeline.

This is **not** distributed RAG, not multi-agent answer aggregation, and not prefill/decode disaggregated serving.

---

## Overall System Target

Final pipeline:

```text
Student prompt
   ↓
Coordinator phone
   ↓
Phone A: shard_0 — embedding + layers 0–2
   ↓ activation tensor
Phone B: shard_1 — layers 3–5
   ↓ activation tensor
Phone C: shard_2 — layers 6–8
   ↓ activation tensor
Phone D: shard_3 — layers 9–11 + lm_head
   ↓ logits
Coordinator samples token
   ↓
Repeat for 10–30 tokens
```

---

## Team Split

We do **not** need five equal parallel tracks. The model/export side is heavy, so two people should work on that.

Recommended split:

```text
Person 1: Python model sharding
Person 2: ExecuTorch export + model artifacts
Person 3: Android ExecuTorch runtime
Person 4: P2P tensor transport
Person 5: Coordinator + integration harness
```

Polished UI is not a priority right now. We only need simple logs/debug screens.

---

# Person 1: Python Model Sharding Lead

## Goal

Prove that the selected model can be split into layer shards and still produce the same output as the full model.

## What They Implement

Start entirely in Python. No Android work initially.

### Tasks

1. Pick a tiny decoder-only model.
2. Run the full model normally.
3. Inspect the model architecture:
   ```text
   embeddings
   transformer layers
   final norm
   lm_head
   ```
4. Split the model into shards:
   ```text
   shard_0 = embedding + layers 0–2
   shard_1 = layers 3–5
   shard_2 = layers 6–8
   shard_3 = layers 9–11 + final_norm + lm_head
   ```
5. Write a Python sharded forward pass:
   ```python
   h0 = shard_0(input_ids)
   h1 = shard_1(h0)
   h2 = shard_2(h1)
   logits = shard_3(h2)
   ```
6. Compare full-model logits with sharded-model logits.
7. Print and document tensor shapes at every boundary.

## Files to Create

```text
/model
  choose_model.md
  run_full_model.py
  run_sharded_model.py
  shard_modules.py
  compare_logits.py
  shape_report.md
```

## How to Test

### Test 1: Full Model Runs

```text
Input: "Explain gravity simply."
Output: logits or generated text
```

### Test 2: Sharded Model Runs

```text
input_ids
 → shard_0
 → shard_1
 → shard_2
 → shard_3
 → logits
```

### Test 3: Logit Comparison

```text
max_abs_diff(full_logits, sharded_logits) should be very small
```

For fp32 Python, the output should be near-identical.

## Deliverable to the Team

Person 1 should hand over:

```text
- working Python sharded model
- exact layer split
- tensor shapes between shards
- model hidden size
- vocabulary size
- sequence length assumptions
```

Example handoff:

```json
{
  "hidden_size": 768,
  "vocab_size": 50257,
  "dtype": "float32",
  "shards": [
    {"id": 0, "input": "input_ids", "output": "[1, seq_len, 768]"},
    {"id": 1, "input": "[1, seq_len, 768]", "output": "[1, seq_len, 768]"},
    {"id": 2, "input": "[1, seq_len, 768]", "output": "[1, seq_len, 768]"},
    {"id": 3, "input": "[1, seq_len, 768]", "output": "[1, seq_len, vocab_size]"}
  ]
}
```

---

# Person 2: ExecuTorch Export + Model Artifact Lead

## Goal

Take the Python shards from Person 1 and export each shard into ExecuTorch `.pte` files.

## What They Implement

This person owns the model export pipeline.

### Tasks

1. Learn/export a tiny sample PyTorch model to ExecuTorch first.
2. Build `export_shards.py`.
3. Export each shard separately:
   ```text
   shard_0.pte
   shard_1.pte
   shard_2.pte
   shard_3.pte
   ```
4. Create dummy test inputs for each shard.
5. Save expected outputs from Python for each shard.
6. Create `model_config.json`.
7. Document exact input/output names, shapes, and dtypes.
8. Confirm each exported shard can be loaded by a small ExecuTorch test script, if possible.

## Files to Create

```text
/model_export
  export_shards.py
  export_test_model.py
  model_config.json
  generate_test_tensors.py
  verify_exported_shards.py

/model_artifacts
  shard_0.pte
  shard_1.pte
  shard_2.pte
  shard_3.pte
  tokenizer.json
  test_inputs/
  expected_outputs/
```

## How to Test

### Test 1: Export One Dummy Model

Before real shards:

```text
simple PyTorch model → simple_model.pte
```

This confirms that the export toolchain works.

### Test 2: Export One Real Shard

Start with `shard_1`, because it likely takes hidden states as input and outputs hidden states. It is simpler than `shard_0` or `shard_3`.

```text
hidden_states → shard_1.pte → hidden_states
```

### Test 3: Compare Exported Output with Python Output

For each shard:

```text
Python shard output ≈ ExecuTorch shard output
```

## Deliverable to the Team

Person 2 should hand over:

```text
/model_artifacts
  shard_0.pte
  shard_1.pte
  shard_2.pte
  shard_3.pte
  model_config.json
  test input/output tensors
```

Most important handoff is `model_config.json`:

```json
{
  "model_name": "tiny-edu-transformer",
  "num_shards": 4,
  "dtype": "float32",
  "hidden_size": 768,
  "max_seq_len": 64,
  "vocab_size": 50257,
  "shards": [
    {
      "shard_id": 0,
      "description": "embedding + layers 0-2",
      "input_shape": [1, 64],
      "input_dtype": "int64",
      "output_shape": [1, 64, 768],
      "output_dtype": "float32"
    },
    {
      "shard_id": 1,
      "description": "layers 3-5",
      "input_shape": [1, 64, 768],
      "input_dtype": "float32",
      "output_shape": [1, 64, 768],
      "output_dtype": "float32"
    },
    {
      "shard_id": 2,
      "description": "layers 6-8",
      "input_shape": [1, 64, 768],
      "input_dtype": "float32",
      "output_shape": [1, 64, 768],
      "output_dtype": "float32"
    },
    {
      "shard_id": 3,
      "description": "layers 9-11 + lm_head",
      "input_shape": [1, 64, 768],
      "input_dtype": "float32",
      "output_shape": [1, 64, 50257],
      "output_dtype": "float32"
    }
  ]
}
```

---

# Person 3: Android ExecuTorch Runtime Lead

## Goal

Make Android run one ExecuTorch shard locally.

This person should not wait for the final model. They can start with a dummy `.pte`.

## What They Implement

### Tasks

1. Create Android project.
2. Add ExecuTorch runtime.
3. Load a `.pte` model from assets.
4. Implement a local `ShardRunner`.
5. Convert raw tensor bytes into ExecuTorch input tensors.
6. Run inference.
7. Convert ExecuTorch output tensor into `TensorPayload`.
8. Print output shape, dtype, checksum, and latency.
9. Later load the real shards from Person 2.

## Files to Create

```text
/android/app/src/main/java/...
  executorch/
    ShardRunner.kt
    ExecuTorchShardRunner.kt
    TensorPayload.kt
    TensorUtils.kt
    LocalShardTest.kt
```

## Core Interface

```kotlin
data class TensorPayload(
    val requestId: String,
    val step: Int,
    val sourceShard: Int,
    val targetShard: Int,
    val shape: IntArray,
    val dtype: String,
    val bytes: ByteArray
)

interface ShardRunner {
    fun loadShard(shardId: Int, modelPath: String)
    fun run(input: TensorPayload): TensorPayload
}
```

## How to Test

### Test 1: Load Dummy `.pte`

```text
Android app starts
loads simple_model.pte
runs test input
prints output
```

### Test 2: Load Real Shard

```text
load shard_1.pte
input: test hidden state from Person 2
output: compare checksum/shape with expected output
```

### Test 3: Local Chain on One Phone

If possible, run all shards locally on one Android phone:

```text
shard_0 → shard_1 → shard_2 → shard_3
```

This is very useful before doing P2P.

## Deliverable to the Team

They expose a working function:

```kotlin
runShard(shardId, tensorPayload) -> tensorPayload
```

This is what Person 5 will call from the coordinator later.

---

# Person 4: P2P Tensor Transport Lead

## Goal

Make phones communicate locally and send tensors over the P2P network.

This person does not need models or ExecuTorch initially.

## What They Implement

### Tasks

1. Build create/join session.
2. Discover nearby peers.
3. Send simple text message.
4. Send JSON control message.
5. Send binary payload.
6. Send `TensorPayload`.
7. Add peer registry.
8. Add heartbeats.
9. Add timeout/disconnect handling.
10. Support linear chain routing:
    ```text
    coordinator → shard_0 peer → shard_1 peer → shard_2 peer → shard_3 peer → coordinator
    ```

## Files to Create

```text
/android/app/src/main/java/...
  network/
    P2PTransport.kt
    NearbyP2PTransport.kt
    ControlMessage.kt
    PeerInfo.kt
    PeerRegistry.kt
    TensorPayloadSender.kt
    HeartbeatManager.kt
```

## Core Interface

```kotlin
interface P2PTransport {
    fun startAdvertising(sessionId: String)
    fun startDiscovery(sessionId: String)

    fun sendControl(peerId: String, message: ControlMessage)
    fun sendTensor(peerId: String, payload: TensorPayload)

    fun onControl(handler: (peerId: String, message: ControlMessage) -> Unit)
    fun onTensor(handler: (peerId: String, payload: TensorPayload) -> Unit)
}
```

Control message:

```kotlin
data class ControlMessage(
    val type: String,
    val sessionId: String,
    val requestId: String?,
    val payload: Map<String, String>
)
```

## How to Test

### Test 1: Two-Phone Hello

```text
Phone A discovers Phone B
Phone A sends "hello"
Phone B receives "hello"
```

### Test 2: Control Message

```json
{
  "type": "CAPABILITY_ADVERTISE",
  "sessionId": "classmesh-test",
  "payload": {
    "availableShards": "1,2",
    "battery": "0.82"
  }
}
```

### Test 3: Fake Tensor

Send random bytes:

```text
shape = [1, 64, 768]
dtype = float32
bytes = random tensor bytes
```

Receiver prints:

```text
received tensor
shape: [1, 64, 768]
dtype: float32
checksum: abc123
```

### Test 4: Chain Forwarding

```text
Phone A → Phone B → Phone C → Phone D → Phone A
```

Each phone modifies metadata:

```text
visited = visited + currentPeerId
```

## Deliverable to the Team

They expose:

```kotlin
sendTensor(peerId, tensorPayload)
onTensor { ... }
```

Person 5 uses this for the distributed pipeline.

---

# Person 5: Coordinator + Integration Harness Lead

## Goal

Own the end-to-end distributed inference control flow.

This is not UI. This is the system orchestrator and integration harness.

## What They Implement

### Tasks

1. Define shared interfaces.
2. Implement fake shard runner.
3. Implement local fake pipeline.
4. Implement static shard assignment.
5. Implement coordinator state machine.
6. Implement prefill flow.
7. Implement decode loop.
8. Implement token sampler.
9. Add logs and metrics.
10. Later plug in Person 3’s `ShardRunner`.
11. Later plug in Person 4’s `P2PTransport`.

## Files to Create

```text
/android/app/src/main/java/...
  inference/
    InferenceSession.kt
    ShardAssignment.kt
    DistributedInferenceCoordinator.kt
    FakeShardRunner.kt
    RemoteShardRunner.kt
    TokenSampler.kt
    PipelineLogger.kt
    IntegrationHarness.kt
```

## Core State

```kotlin
data class ShardAssignment(
    val shardId: Int,
    val peerId: String,
    val layerRange: String,
    val modelPath: String
)

data class InferenceSession(
    val requestId: String,
    val prompt: String,
    val shardPlan: List<ShardAssignment>,
    var step: Int,
    val generatedTokens: MutableList<Int>,
    val maxNewTokens: Int
)
```

## What the Coordinator Does

For prefill:

```text
1. Tokenize prompt.
2. Send input to shard 0.
3. Wait for output from shard 3.
4. Receive logits.
5. Sample next token.
```

For decode:

```text
1. Append sampled token.
2. Send updated input through same phone pipeline.
3. Receive logits.
4. Sample next token.
5. Repeat until max_new_tokens or stop token.
```

For MVP without KV cache:

```text
At every step, recompute the full sequence.
```

This is slower but much easier.

## How to Test

### Test 1: Local Fake Pipeline

No phones, no ExecuTorch.

```text
Fake shard 0 receives tensor
Fake shard 1 receives tensor
Fake shard 2 receives tensor
Fake shard 3 produces fake logits
Coordinator samples token
```

### Test 2: Fake P2P Pipeline

Use Person 4’s transport.

```text
Coordinator sends fake tensor across real phones.
Each phone acts as fake shard.
Coordinator receives fake logits.
```

### Test 3: One-Hop Real Shard

Use Person 3’s `ShardRunner`.

```text
Coordinator sends tensor to one remote phone.
Remote phone runs real ExecuTorch shard.
Coordinator receives result.
```

### Test 4: Real Multi-Shard Pipeline

```text
shard_0 on Phone A
shard_1 on Phone B
shard_2 on Phone C
shard_3 on Phone D
Coordinator receives logits.
```

## Deliverable to the Team

They provide the integration command/debug mode:

```text
Mode 1: local fake pipeline
Mode 2: P2P fake pipeline
Mode 3: one-hop real shard
Mode 4: full real distributed pipeline
```

---

# Shared Contracts Everyone Must Agree on Early

Before everyone codes separately, freeze these.

## 1. TensorPayload

```kotlin
data class TensorPayload(
    val requestId: String,
    val step: Int,
    val sourceShard: Int,
    val targetShard: Int,
    val shape: IntArray,
    val dtype: String,
    val bytes: ByteArray
)
```

## 2. ShardRunner

```kotlin
interface ShardRunner {
    fun loadShard(shardId: Int, modelPath: String)
    fun run(input: TensorPayload): TensorPayload
}
```

## 3. P2PTransport

```kotlin
interface P2PTransport {
    fun sendControl(peerId: String, message: ControlMessage)
    fun sendTensor(peerId: String, payload: TensorPayload)
    fun onControl(handler: (peerId: String, message: ControlMessage) -> Unit)
    fun onTensor(handler: (peerId: String, payload: TensorPayload) -> Unit)
}
```

## 4. ShardAssignment

```kotlin
data class ShardAssignment(
    val shardId: Int,
    val peerId: String,
    val layerRange: String,
    val modelPath: String
)
```

## 5. Model Config

```json
{
  "num_shards": 4,
  "hidden_size": 768,
  "dtype": "float32",
  "max_seq_len": 64,
  "shards": [
    {
      "shard_id": 0,
      "description": "embedding + layers 0-2",
      "output_shape": [1, 64, 768]
    },
    {
      "shard_id": 1,
      "description": "layers 3-5",
      "output_shape": [1, 64, 768]
    },
    {
      "shard_id": 2,
      "description": "layers 6-8",
      "output_shape": [1, 64, 768]
    },
    {
      "shard_id": 3,
      "description": "layers 9-11 + lm_head",
      "output_shape": [1, 64, 50257]
    }
  ]
}
```

---

# Integration Sequence

Do integration in this order only.

## Integration 1: Python Proof

People involved: Person 1 + Person 2

```text
Full model vs sharded model comparison.
```

Success:

```text
same logits or very close logits
```

---

## Integration 2: Android Local Shard

People involved: Person 2 + Person 3

```text
Person 2 exports shard_1.pte.
Person 3 runs shard_1.pte on Android using test tensor.
```

Success:

```text
Android output shape matches expected shape.
```

---

## Integration 3: P2P Fake Tensor

People involved: Person 4 + Person 5

```text
Coordinator sends fake tensor across phone chain.
```

Success:

```text
Tensor reaches final phone and returns to coordinator.
```

---

## Integration 4: One-Hop Remote ExecuTorch

People involved: Person 3 + Person 4 + Person 5

```text
Phone A sends tensor to Phone B.
Phone B runs real shard.
Phone B sends result back.
```

Success:

```text
remote shard execution works
```

---

## Integration 5: Full Real Pipeline, One Token

People involved: everyone

```text
Prompt
 → shard_0 phone
 → shard_1 phone
 → shard_2 phone
 → shard_3 phone
 → logits
 → sampled token
```

Success:

```text
one real next token generated through the phone pipeline
```

---

## Integration 6: Full Short Answer

People involved: everyone

```text
Repeat one-token generation 10–30 times.
```

Success:

```text
short educational answer generated
```

---

# Testing Strategy by Layer

## Unit Tests

Each person owns unit tests for their layer.

```text
Person 1:
  full vs sharded logits

Person 2:
  exported shard vs Python shard output

Person 3:
  Android local shard output shape/checksum

Person 4:
  P2P message delivery and tensor checksum

Person 5:
  coordinator fake pipeline and decode loop
```

## Integration Tests

```text
Test A:
  fake tensor over P2P chain

Test B:
  one-hop remote shard

Test C:
  full pipeline with fake shards

Test D:
  full pipeline with real shards

Test E:
  full answer generation
```

## Debug Logs Required

Every stage should log:

```text
request_id
step
source_shard
target_shard
tensor_shape
dtype
payload_size
local_inference_latency_ms
network_latency_ms
checksum
```

Example:

```text
[request=q1 step=4] shard_1 received tensor shape=[1,64,768] bytes=196608
[request=q1 step=4] shard_1 inference latency=43ms
[request=q1 step=4] shard_1 sent output to shard_2
```

---

# What to Build First

## First Priority

```text
Python sharded model proof
+
Android ExecuTorch single-shard proof
+
P2P fake tensor transfer
+
Coordinator fake pipeline
```

These can happen in parallel.

## Do Not Start With

```text
- polished UI
- KV cache
- dynamic shard placement
- large model
- full mesh routing
- multiple students
- RAG
- voice
```

---

# Final Assignment Summary

| Person | Owns | Main Deliverable | First Test |
|---:|---|---|---|
| 1 | Python model sharding | Full vs sharded model parity | Compare logits |
| 2 | ExecuTorch export | `.pte` shards + config | Export and verify one shard |
| 3 | Android ExecuTorch runtime | `ShardRunner` | Run one shard locally on Android |
| 4 | P2P tensor transport | `P2PTransport` | Send fake tensor between phones |
| 5 | Coordinator + integration harness | Distributed pipeline state machine | Fake 4-shard pipeline |

This split gives two people to the riskiest model/export track, keeps Android runtime separate, keeps networking separate, and keeps one person focused on making the pieces integrate.
