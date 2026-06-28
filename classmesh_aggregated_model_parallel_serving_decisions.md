# ClassMesh: Aggregated Model-Parallel Serving over a P2P Phone Pipeline

## 1. Project Summary

**ClassMesh** is an offline education-focused distributed inference system where a single student query is served by a pipeline of nearby mobile phones.

Each phone owns a shard of the model and runs that shard locally using **ExecuTorch**. Phones communicate over a peer-to-peer network and pass intermediate activations from one device to the next.

The goal is to demonstrate that a group of nearby phones can collectively run a model that may be too large or too expensive for one low-end device to run alone, without requiring cloud connectivity.

---

## 2. Final Direction Chosen

We are building:

> **Aggregated model-parallel serving over a peer-to-peer mobile phone pipeline using ExecuTorch.**

This means:

- A single student query is served by multiple phones.
- The model is partitioned by layers.
- Each phone runs one model shard.
- The same phone pipeline handles both prompt processing and token generation.
- We are not splitting prefill and decode onto separate workers.
- ExecuTorch is used as the on-device runtime for each model shard.
- The P2P layer handles communication, activation transfer, shard assignment, and orchestration.

---

## 3. Terminology Decisions

### 3.1 What This Is

This project is best described as:

- **Layer-wise model-parallel inference**
- **Pipeline-parallel inference across phones**
- **Aggregated model-parallel serving**
- **P2P distributed inference over mobile devices**

The preferred phrasing is:

> **Aggregated model-parallel serving over a P2P phone pipeline.**

### 3.2 What This Is Not

This is **not**:

- **Distributed RAG**
- **Multi-agent response aggregation**
- **Disaggregated prefill/decode serving**
- **Only task-parallel inference**
- **Only multiple phones generating independent answers**
- **Cloud-based distributed inference**

### 3.3 Aggregated vs Disaggregated Serving

In our context:

#### Aggregated Serving

The same worker setup handles the full inference lifecycle:

```text
Prefill:
  Phone A → Phone B → Phone C → Phone D

Decode:
  Phone A → Phone B → Phone C → Phone D
```

The same model shard placement is used for both phases.

#### Disaggregated Serving

Prefill and decode would be served by separate workers:

```text
Prefill:
  Phone A/B

Decode:
  Phone C/D
```

We are **not** doing this.

### 3.4 Model Parallelism

Our model is split by layers:

```text
Phone A: embedding + layers 0–2
Phone B: layers 3–5
Phone C: layers 6–8
Phone D: layers 9–11 + final norm + LM head
```

This is model-parallel inference, not RAG.

---

## 4. Product Context

The product use case is **offline education**.

A student asks a question such as:

```text
Explain photosynthesis in simple words.
```

Instead of sending the query to the cloud or running the entire model on one phone, ClassMesh distributes inference across nearby phones.

The final demo should show:

- No internet connection.
- Multiple phones connected locally.
- Each phone assigned a model shard.
- Student asks one educational question.
- Activations flow through the phone pipeline.
- A short educational answer is generated.
- UI displays which phone ran which shard and the latency per shard.

---

## 5. High-Level Architecture

```text
Student Query
    ↓
Coordinator Phone
    - tokenizes prompt
    - starts inference
    - samples next token
    ↓
Phone A
    - embedding + layers 0–2
    ↓ activation tensor
Phone B
    - layers 3–5
    ↓ activation tensor
Phone C
    - layers 6–8
    ↓ activation tensor
Phone D
    - layers 9–11 + final norm + LM head
    ↓ logits
Coordinator
    - samples next token
    - repeats pipeline
    ↓
Final educational answer
```

---

## 6. Core Components

### 6.1 Model Selection

We should start with a small decoder-only transformer.

Requirements:

- Small enough to run on Android.
- Easy to split by layers.
- Short-context friendly.
- Capable of generating simple educational answers.
- Exportable to ExecuTorch.
- Works with short outputs of roughly 10–30 tokens.

Important decision:

> The goal is not to maximize answer quality. The goal is to prove distributed model-parallel inference across phones.

---

### 6.2 Model Partitioner

This is an offline Python pipeline.

It takes a full PyTorch model and splits it into layer shards.

Example:

```text
Full model:
  token_embedding
  layer 0
  layer 1
  layer 2
  layer 3
  layer 4
  layer 5
  layer 6
  layer 7
  layer 8
  layer 9
  layer 10
  layer 11
  final_norm
  lm_head

Shard 0:
  token_embedding + layers 0–2

Shard 1:
  layers 3–5

Shard 2:
  layers 6–8

Shard 3:
  layers 9–11 + final_norm + lm_head
```

Outputs:

```text
shard_0.pte
shard_1.pte
shard_2.pte
shard_3.pte
model_config.json
tokenizer files
```

Before exporting to ExecuTorch, we must verify in Python:

```text
full_model(prompt) ≈ shard_3(shard_2(shard_1(shard_0(prompt))))
```

---

### 6.3 ExecuTorch Shard Runner

Each phone runs one model shard locally using ExecuTorch.

The Android app should expose a clean wrapper:

```kotlin
interface ShardRunner {
    fun loadShard(shardId: Int, modelPath: String)
    fun run(input: TensorPayload): TensorPayload
}
```

Every worker phone runs exactly one shard for the MVP:

```text
Phone A loads shard_0.pte
Phone B loads shard_1.pte
Phone C loads shard_2.pte
Phone D loads shard_3.pte
```

The rest of the distributed system should not depend directly on ExecuTorch internals.

---

### 6.4 Tensor Payload Format

Phones pass intermediate activations between each other.

Define a shared tensor payload format:

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

Expected shapes:

```text
Prefill:
  [batch, sequence_length, hidden_size]

Decode:
  [batch, 1, hidden_size]
```

Important decision:

> Tensor data should not be sent as JSON. Use a small JSON/control header plus a binary payload.

---

### 6.5 P2P Network Layer

The P2P layer handles communication between phones.

Responsibilities:

- Discover phones.
- Create classroom session.
- Join classroom session.
- Send control messages.
- Send binary tensor payloads.
- Detect disconnects.
- Handle errors and timeouts.

MVP topology:

```text
Coordinator → Phone A → Phone B → Phone C → Phone D → Coordinator
```

Important decision:

> Start with a linear chain pipeline. Do not build a full arbitrary mesh first.

Control message types:

```text
SESSION_CREATE
SESSION_JOIN
CAPABILITY_ADVERTISE
ASSIGN_SHARD
LOAD_SHARD
START_PREFILL
ACTIVATION
LOGITS
TOKEN_SELECTED
HEARTBEAT
ERROR
SESSION_END
```

---

### 6.6 Coordinator

The coordinator controls the distributed inference session.

Responsibilities:

- Receive student prompt.
- Tokenize prompt.
- Manage shard assignment.
- Start prefill.
- Run decode loop.
- Route inputs to the first shard.
- Receive final logits.
- Sample next token.
- Track generated tokens.
- Stop generation.
- Display final answer.
- Handle timeouts and failures.

Coordinator state:

```kotlin
data class InferenceSession(
    val requestId: String,
    val prompt: String,
    val shardPlan: List<ShardAssignment>,
    var step: Int,
    val generatedTokens: MutableList<Int>,
    val maxNewTokens: Int
)
```

---

### 6.7 Worker Runtime

Each worker phone runs a simple loop:

```text
1. Receive ASSIGN_SHARD.
2. Load shard_x.pte.
3. Wait for tensor input.
4. Run local ExecuTorch shard.
5. Send output tensor to next phone.
6. Repeat until SESSION_END.
```

Worker interface:

```kotlin
interface WorkerNode {
    fun assignShard(shardId: Int)
    fun onTensorReceived(payload: TensorPayload)
    fun forwardToNext(output: TensorPayload)
}
```

---

### 6.8 KV Cache Manager

For proper LLM decoding, each shard should maintain its own KV cache.

```text
Phone A stores KV cache for layers 0–2
Phone B stores KV cache for layers 3–5
Phone C stores KV cache for layers 6–8
Phone D stores KV cache for layers 9–11
```

Important decision:

> Do not send KV cache across phones every token.

Each phone should keep the cache for its own layers. The only tensor passed between phones during decode should be the current hidden-state activation.

MVP simplification:

- Start without KV cache.
- Recompute the full sequence for every generated token.
- Add KV cache only after the distributed pipeline works end to end.

---

### 6.9 Education UI and Demo Layer

The UI must make the distributed inference visible.

The demo should show:

```text
ClassMesh Session: Active
Internet: Off
Connected phones: 4

Shard assignment:
Phone A — Embedding + Layers 0–2
Phone B — Layers 3–5
Phone C — Layers 6–8
Phone D — Layers 9–11 + LM Head

Current token: 12/30

Latency:
Phone A: 41 ms
Phone B: 38 ms
Phone C: 44 ms
Phone D: 52 ms
Network transfer: 29 ms

Answer:
Photosynthesis is how plants use sunlight...
```

---

## 7. Sequential Implementation Plan

### Phase 0: Freeze the MVP

Decide:

```text
Number of phones: 3 or 4
Model: tiny decoder-only transformer
Prompt: one fixed educational prompt first
Output length: 10–30 tokens
Topology: linear chain
Transport: Nearby Connections or equivalent P2P transport
Runtime: ExecuTorch Android
Backend: CPU/XNNPACK first; Qualcomm/QNN later if time permits
```

MVP success condition:

> With internet off, multiple phones connect locally, each phone runs a model shard, and together they generate a short educational answer.

---

### Phase 1: Python-Only Sharded Model

Before Android, prove the model can be split.

Build:

```text
run_full_model.py
run_sharded_model.py
```

Verify:

```text
full_model(prompt) ≈ shard_3(shard_2(shard_1(shard_0(prompt))))
```

Deliverable:

```text
Prompt: Explain gravity simply.
Full model output: ...
Sharded model output: same or close enough.
```

---

### Phase 2: Export Shards to ExecuTorch

Export each shard to `.pte`.

Artifacts:

```text
/model_artifacts
  shard_0.pte
  shard_1.pte
  shard_2.pte
  shard_3.pte
  tokenizer.json
  model_config.json
```

Example `model_config.json`:

```json
{
  "num_shards": 4,
  "hidden_size": 768,
  "dtype": "fp16",
  "shards": [
    {"shard_id": 0, "layers": "embedding + 0-2"},
    {"shard_id": 1, "layers": "3-5"},
    {"shard_id": 2, "layers": "6-8"},
    {"shard_id": 3, "layers": "9-11 + lm_head"}
  ]
}
```

---

### Phase 3: Run Each Shard Locally on Android

Do not use P2P yet.

Build a debug screen:

```text
Load shard 0
Run shard 0 with test tensor
Show output shape and latency

Load shard 1
Run shard 1 with test tensor
Show output shape and latency
```

This proves:

```text
.pte file → Android app → ExecuTorch runtime → output tensor
```

---

### Phase 4: Build P2P Messaging Without Models

First send simple messages:

```text
Phone A → Phone B: hello
Phone B → Phone A: ack
```

Then send control messages:

```json
{
  "type": "CAPABILITY_ADVERTISE",
  "peerId": "phone_b",
  "battery": 0.83,
  "availableShards": [1]
}
```

Then send fake tensor bytes:

```text
shape: [1, 1, 768]
dtype: fp16
bytes: random tensor data
```

Deliverable:

```text
Phone A sends fake activation tensor.
Phone B receives it, reconstructs shape, prints checksum.
```

---

### Phase 5: One-Hop Remote Shard Execution

Combine P2P and ExecuTorch.

Flow:

```text
Phone A sends tensor to Phone B
Phone B runs shard_1.pte
Phone B sends output tensor back to Phone A
```

Deliverable:

```text
RemoteShardRunner works:
input tensor on Phone A
→ shard execution on Phone B
→ output tensor back on Phone A
```

---

### Phase 6: Multi-Phone Pipeline With Fake Model

Before using real model shards, build the chain with fake workers.

```text
Coordinator → Fake Shard 0 → Fake Shard 1 → Fake Shard 2 → Fake Shard 3 → Coordinator
```

Each fake shard can transform the tensor slightly:

```text
output = input + shard_id
```

Deliverable:

```text
Pipeline routing works across phones.
```

---

### Phase 7: Multi-Phone Pipeline With Real Shards

Replace fake shards with real ExecuTorch shard runners.

Flow:

```text
tokens / embeddings
    ↓
Phone A: shard 0
    ↓
Phone B: shard 1
    ↓
Phone C: shard 2
    ↓
Phone D: shard 3
    ↓
Coordinator receives logits
```

First goal:

```text
Generate one next token.
```

Deliverable:

```text
Prompt → distributed pipeline → logits → sampled token
```

---

### Phase 8: Full Decode Loop

Repeat the pipeline.

```text
for step in 1..max_new_tokens:
    send token / activation into pipeline
    receive logits
    sample next token
    append token
```

Start with:

```text
max_new_tokens = 5
```

Then increase to:

```text
max_new_tokens = 20 or 30
```

Deliverable:

```text
Student prompt → generated educational answer
```

---

### Phase 9: UI and Demo Polish

Make the distributed system obvious.

The final demo should show:

- Session creation.
- Peer discovery.
- Shard assignment.
- No internet.
- Per-phone execution.
- Per-shard latency.
- Network transfer latency.
- Generated educational answer.

---

## 8. Five-Person Team Split

### Person 1: Model Partitioning Lead

Owns:

- Choose tiny model.
- Split layers in Python.
- Verify sharded output.
- Export `.pte` files.
- Define tensor shapes.
- Create `model_config.json`.

Main deliverable:

```text
shard_0.pte ... shard_3.pte
```

---

### Person 2: ExecuTorch Android Lead

Owns:

- Android ExecuTorch setup.
- `ShardRunner` wrapper.
- Local shard loading.
- Local shard execution.
- Tensor conversion.
- Latency measurement.

Main deliverable:

```text
runShard(inputTensor) → outputTensor on Android
```

---

### Person 3: P2P Networking Lead

Owns:

- P2P session setup.
- Peer discovery.
- Control messages.
- Binary payload transfer.
- Heartbeats.
- Disconnect detection.

Main deliverable:

```text
send TensorPayload from one phone to another
```

---

### Person 4: Distributed Inference Coordinator Lead

Owns:

- Shard assignment.
- Inference state machine.
- Prefill flow.
- Decode loop.
- Activation routing.
- Logits collection.
- Token sampling.
- Timeout handling.

Main deliverable:

```text
prompt → distributed shards → generated tokens
```

---

### Person 5: UI / Demo / Integration Lead

Owns:

- Classroom UI.
- Connected devices screen.
- Shard pipeline visualization.
- Final answer screen.
- Metrics dashboard.
- Demo script.
- Fallback plan.

Main deliverable:

```text
Judge-friendly working demo
```

---

## 9. Interfaces to Freeze Early

### TensorPayload

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

### ControlMessage

```kotlin
data class ControlMessage(
    val type: String,
    val sessionId: String,
    val requestId: String?,
    val payload: Map<String, Any>
)
```

### ShardAssignment

```kotlin
data class ShardAssignment(
    val shardId: Int,
    val peerId: String,
    val layerRange: String,
    val modelPath: String
)
```

### ShardRunner

```kotlin
interface ShardRunner {
    fun loadShard(modelPath: String)
    fun run(input: TensorPayload): TensorPayload
}
```

### P2PTransport

```kotlin
interface P2PTransport {
    fun sendControl(peerId: String, message: ControlMessage)
    fun sendTensor(peerId: String, payload: TensorPayload)
    fun onControl(handler: (ControlMessage) -> Unit)
    fun onTensor(handler: (TensorPayload) -> Unit)
}
```

### InferenceCoordinator

```kotlin
interface InferenceCoordinator {
    fun createSession()
    fun assignShards(peers: List<PeerInfo>)
    fun generate(prompt: String): String
}
```

---

## 10. Build Sequence Summary

Recommended order:

```text
1. Python full model works.
2. Python sharded model works.
3. Export shards to .pte.
4. Android runs one shard locally.
5. P2P sends JSON messages.
6. P2P sends fake tensor bytes.
7. One phone remotely runs one shard.
8. Multi-phone fake pipeline works.
9. Multi-phone real pipeline produces logits.
10. Multi-phone real pipeline generates one token.
11. Multi-phone real pipeline generates short answer.
12. Add UI, metrics, and demo polish.
```

---

## 11. What to Avoid Initially

Do not build these first:

- Full arbitrary mesh routing.
- Dynamic repartitioning.
- Automatic leader election.
- Large model.
- KV-cache optimization.
- Qualcomm/QNN optimization.
- Voice input.
- RAG.
- Teacher dashboard.
- Multi-student classroom.

These can be added only after the core distributed inference works.

---

## 12. MVP Scope

### MVP Must Have

- 3 or 4 Android phones.
- Local P2P connection.
- Static shard assignment.
- ExecuTorch model shard on each phone.
- Linear phone pipeline.
- One educational query.
- Short generated response.
- UI showing shard ownership and latency.

### MVP Should Have

- Fake-model mode for debugging.
- Local single-device shard test screen.
- Error handling for missing peers.
- Basic metrics dashboard.

### MVP Stretch Goals

- KV cache per shard.
- Dynamic shard assignment.
- Qualcomm/QNN acceleration.
- More polished education prompts.
- Multiple question types.
- Fallback to single-device tiny model.
- Optional local RAG/curriculum grounding.

---

## 13. Known Technical Risks

### 13.1 Model Export Risk

Partial model shards may be difficult to export cleanly to ExecuTorch.

Mitigation:

- Start with the smallest possible model.
- Prove sharding in Python first.
- Export only after tensor interfaces are stable.

### 13.2 Tensor Shape Compatibility Risk

Shard outputs must exactly match the next shard’s expected inputs.

Mitigation:

- Define `model_config.json`.
- Print shapes at every boundary.
- Add local tests for every shard.

### 13.3 P2P Transfer Latency Risk

Activation tensors may be large.

Mitigation:

- Use small hidden size.
- Use fp16 where possible.
- Use binary payloads, not JSON.
- Keep output length short for MVP.

### 13.4 KV Cache Complexity Risk

Distributed KV cache management is hard.

Mitigation:

- Start without KV cache.
- Recompute full sequence for MVP.
- Add KV cache only after basic generation works.

### 13.5 Demo Reliability Risk

Multi-phone demos can fail due to networking issues.

Mitigation:

- Build fake pipeline mode.
- Build one-phone local simulation mode.
- Prepare a fallback video or logs.
- Keep topology static.

---

## 14. Fallback Plan

If full multi-token generation is too hard, demo the following in order of fallback strength:

### Fallback A: Full Short Answer

Best case:

```text
Student prompt → distributed phone pipeline → 20-token answer
```

### Fallback B: One-Token Generation

Still valid:

```text
Student prompt → distributed phone pipeline → logits → one sampled token
```

Explain that full generation repeats the same loop.

### Fallback C: Distributed Forward Pass

Minimum technical proof:

```text
Input tensor → Phone A shard → Phone B shard → Phone C shard → Phone D shard → final logits
```

### Fallback D: Fake Pipeline + Real Local Shard

If integration breaks:

```text
Real ExecuTorch shard locally
+
P2P fake tensor pipeline across phones
```

This still demonstrates the two hardest pieces separately.

---

## 15. Final Demo Script

A good demo flow:

```text
1. Open ClassMesh on 4 phones.
2. Turn off internet.
3. Create classroom session on coordinator phone.
4. Other phones join.
5. Coordinator assigns model shards.
6. Each phone loads its ExecuTorch shard.
7. Student asks: "Explain photosynthesis in simple words."
8. UI shows activations flowing through phones.
9. Each phone displays local shard latency.
10. Coordinator receives logits and generates tokens.
11. Final answer appears.
12. Metrics show no cloud access and per-phone contribution.
```

---

## 16. Final Pitch

> ClassMesh enables offline AI education by turning nearby phones into a local distributed inference cluster. Instead of relying on cloud LLMs or requiring a single powerful device, ClassMesh partitions a model across multiple phones. Each phone runs a model shard locally with ExecuTorch, while a peer-to-peer pipeline passes activations between devices. The same pipeline handles both prompt processing and token generation, making this an aggregated model-parallel serving system for low-connectivity classrooms.

---

## 17. Current Decision Snapshot

| Decision Area | Chosen Direction |
|---|---|
| Product domain | Offline education |
| System style | Aggregated model-parallel serving |
| Not doing | Disaggregated prefill/decode serving |
| Not doing | Distributed RAG as core |
| Model split | Layer-wise sharding |
| Runtime | ExecuTorch |
| Communication | P2P phone network |
| Topology | Linear chain for MVP |
| Number of phones | 3–4 for MVP, 5-person team |
| Coordinator | Teacher/strongest phone |
| First model | Tiny decoder-only transformer |
| First generation target | Short educational answer |
| KV cache | Defer until basic pipeline works |
| UI focus | Show distributed inference clearly |
| Main success criterion | Multiple phones jointly generate one answer without internet |

