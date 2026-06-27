# Beacon

**Distributed On-Device AI for Underserved Communities**

Qualcomm × Meta ExecuTorch Hackathon · June 27–28, 2026

Beacon runs a single 3-billion-parameter language model distributed across multiple Android phones connected on a local WiFi hotspot — no internet, no cloud, no server. Each phone runs its assigned layer group via ExecuTorch on the Snapdragon Hexagon NPU. Intermediate activations transit between devices over TCP on a phone-hosted hotspot.

The result is a capable AI assistant for first-aid guidance and language interpretation in refugee camps, rural clinics, and disaster zones where connectivity is unavailable.

---

## Problem Statement

Three barriers cut AI off from the people who need it most:

1. **No internet connectivity** — disaster zones and remote clinics have no reliable network
2. **Language barriers** — aid workers and patients often do not share a common language
3. **Hardware constraints** — modern LLMs require high-end hardware most communities cannot access

Beacon addresses all three: offline operation, multilingual voice pipeline, and distributed compute across low-cost devices.

---

## Technical Architecture

### Device Topology

| Device | Role | Responsibilities |
|--------|------|------------------|
| **Phone A** (S25 Ultra loaner) | Coordinator | Hosts WiFi hotspot, runs App UI, Whisper ASR on DSP, LLM layers 1–16 on NPU, TTS on GPU |
| **Phone B** (team device) | Worker | Connects to hotspot, runs LLM layers 17–32 on NPU, returns output tokens |

Architecture supports N worker devices; v1 demo uses 2.

### Inference Pipeline

```
Mic → [Whisper ASR, Hexagon DSP, Phone A] → tokens
  → [LLM layers 1–16, Hexagon NPU, Phone A] → activations (float16, ~3.3 MB)
  → [TCP socket over WiFi hotspot] → Phone B
  → [LLM layers 17–32, Hexagon NPU, Phone B] → output tokens
  → [TCP socket] → Phone A
  → [TTS, GPU, Phone A] → audio
```

### Model Stack

| Component | Model | Export | Backend |
|-----------|-------|--------|---------|
| ASR | Whisper-small (244M params) | `whisper.pte` | Hexagon DSP |
| LLM | Llama 3.2 3B (Q4_K_M, split at layer 16) | `llm_chunk_a.pte` + `llm_chunk_b.pte` | Hexagon NPU |
| TTS | MMS-TTS or VITS-small | `tts.pte` | GPU |

Activation quantization: float32 → float16 before WiFi transit (halves transfer size).

### Android Project Layout

```
BeaconApp/
├── app/src/main/java/com/beacon/
│   ├── ExecuTorchRunner.kt       ← P2 owns
│   ├── InferenceCoordinator.kt   ← P2 owns
│   ├── ActivationServer.kt       ← P3 owns
│   ├── ActivationClient.kt       ← P3 owns
│   └── HotspotManager.kt         ← P3 owns
├── app/src/main/assets/
│   ├── whisper.pte               ← P1 produces
│   ├── llm_chunk_a.pte           ← P1 produces
│   ├── llm_chunk_b.pte           ← P1 produces
│   └── tts.pte                   ← P1 produces
└── scripts/export_models.py      ← P1 owns
```

---

## Team Structure

| Person | Role | Branch | Owns |
|--------|------|--------|------|
| P1 | Model Engineer | `p1/model-export` | Model export, NPU optimization, performance profiling |
| P2 | Android Backend Engineer | `p2/android-backend` | ExecuTorch runtime, full inference pipeline |
| P3 | Networking Engineer | `p3/networking` | WiFi hotspot, TCP transport, activation passing |

### Responsibilities

**P1 — Model Engineer**
- Export Whisper + Llama 3.2 3B (split at layer 16) + TTS to `.pte`
- Validate Hexagon NPU backend
- Profile with Qualcomm AI Hub
- Produce latency/energy numbers
- Support P2 on ExecuTorch runner integration

**P2 — Android Backend Engineer**
- Build `ExecuTorchRunner.kt`
- Wire ASR → LLM-A → TTS on Phone A; LLM-B on Phone B
- Build `InferenceCoordinator.kt`
- Support P3 on ActivationClient integration and P1 on `.pte` loading validation

**P3 — Networking Engineer**
- Build `HotspotManager.kt`, `ActivationServer.kt`, `ActivationClient.kt`
- Implement float16 quantization of activations
- Measure and optimize transfer latency
- Handle reconnection
- Support P2 on end-to-end pipeline wiring

---

## Requirements

### Functional Requirements

| ID | Requirement | Owner | Acceptance Criteria |
|----|-------------|-------|---------------------|
| FR-01 | On-device ASR | P1, P2 | Whisper.pte loads on Phone A DSP; transcribes 10s audio clip to correct text in < 3s |
| FR-02 | LLM chunk A inference | P1, P2 | llm_chunk_a.pte runs on Phone A NPU; produces activation tensor of shape `[1, S, 3200]` |
| FR-03 | Activation transport | P3 | Float16 activation tensor transmitted from Phone A to Phone B in < 100ms |
| FR-04 | LLM chunk B inference | P1, P2 | llm_chunk_b.pte runs on Phone B NPU; produces output logits; decoded token matches reference |
| FR-05 | On-device TTS | P1, P2 | tts.pte produces intelligible speech audio from output tokens |
| FR-06 | WiFi hotspot creation | P3 | Phone A creates hotspot; Phone B connects and receives IP in < 30s |
| FR-07 | Full offline operation | P2, P3 | Complete inference cycle runs with airplane mode on (WiFi Direct only) |
| FR-08 | Pipeline stability | P2 | 10 consecutive inference calls complete without crash or OOM |

### Non-Functional Requirements

| ID | Requirement | Owner | Acceptance Criteria |
|----|-------------|-------|---------------------|
| NFR-01 | End-to-end latency | P2 | Mic release to first spoken word < 4s (target), < 2.5s (stretch) |
| NFR-02 | NPU utilization | P1 | Hexagon NPU > 60% utilization on both phones during LLM inference |
| NFR-03 | Activation transfer latency | P3 | < 100ms p50; < 200ms p95 over local WiFi hotspot |
| NFR-04 | Model RAM footprint | P1 | Each `.pte` chunk < 1.5 GB in device RAM; both chunks fit across two devices |
| NFR-05 | Thermal stability | P1 | No thermal throttle during 10-minute sustained inference session |
| NFR-06 | Activation data integrity | P3 | Max absolute error < 0.001 between sent and received activation values |

---

## Integration Checkpoints

| Time | Checkpoint | Participants | Success Criteria |
|------|------------|--------------|------------------|
| Day 1, 12:00 | Model chunk loading | P1 + P2 | `llm_chunk_a.pte` loads on S25 Ultra and produces activation tensor of correct shape |
| Day 1, 14:00 | Cross-phone inference | P2 + P3 | InferenceCoordinator calls ActivationClient and receives tokens back from Phone B |
| Day 1, 18:00 | Full offline pipeline | All three | Complete one request end-to-end with airplane mode on |
| Day 2, 11:00 | Stability run | All three | 10 consecutive requests complete without crash; latency logged |

---

## Implementation Timeline

### Pre-Hackathon

| Day | Task | Owner | Output |
|-----|------|-------|--------|
| Day 1 | Download Llama 3.2 3B; set up ExecuTorch Python env; baseline export | P1 | Baseline `.pte` on laptop |
| Day 1 | Implement ActivationServer + ActivationClient in Kotlin; test on two laptops | P3 | TCP transport verified |
| Day 2 | Set up Android project; add ExecuTorch gradle dependency; load Whisper sample `.pte` | P2 | ExecuTorch Android build working |
| Day 2 | Split model at layer 16; verify both chunks independently | P1 | Split validated; activation shape confirmed |
| Day 3 | Implement HotspotManager.kt; create hotspot from Phone A | P3 | Working hotspot + TCP on Android |
| Day 3 | Export final `.pte` files targeting Hexagon NPU backend | P1 | All 4 `.pte` files ready |
| Day 4 | Wire ExecuTorchRunner into ActivationClient on two physical phones | P2 + P3 | Distributed inference on 2 phones |
| Day 4 | Measure activation tensor size and WiFi transfer time | P1 + P3 | Transfer latency measurement logged |

### Hackathon Day 1 — June 27

| Time | Task | Owner |
|------|------|-------|
| 9:00–10:30 | Check in; receive S25 Ultra; install APK; confirm `.pte` files load | P2 |
| 10:30–12:00 | Run Qualcomm AI Hub profiler; capture NPU %, tok/s, mW baseline | P1 |
| 10:30–12:00 | Confirm single-device pipeline (ASR → LLM A → TTS) on loaner | P2 |
| 10:30–12:00 | Deploy ActivationServer on Phone B; confirm TCP connection | P3 |
| 12:00–13:00 | **Integration checkpoint 1**: verify `.pte` loading and activation shape | P1 + P2 |
| 13:00–16:00 | Full integration: InferenceCoordinator → ActivationClient → Phone B | P2 + P3 |
| 16:00–18:00 | Optimize NPU utilization; re-run profiler | P1 |
| 18:00–22:00 | Airplane mode test; float16 optimization; 10-request stability run | All |

### Hackathon Day 2 — June 28

| Time | Task | Owner |
|------|------|-------|
| 9:00–11:00 | Stress test transport: 20 requests, measure p50/p95 latency | P3 |
| 9:00–11:00 | Stability run: 10 consecutive full-pipeline requests | P2 |
| 9:00–11:00 | Final profiler run on both phones; produce performance slide | P1 |
| 11:00–13:00 | **Integration checkpoint 2**: full pipeline 3× with airplane mode | All |
| 13:00–15:00 | Submit GitHub repo; README complete; APK attached | P2 |
| 15:00–16:00 | Charge devices; rehearse hotspot connection flow | P3 |
| 16:00+ | Judging presentations | All |

---

## Performance Targets

### P1 — Model Engineer

| Deliverable | Target | Stretch |
|-------------|--------|---------|
| Whisper ASR latency (10s clip) | < 3,000 ms | < 2,000 ms |
| LLM chunk A tok/s | > 5 tok/s | > 8 tok/s |
| LLM chunk B tok/s | > 5 tok/s | > 8 tok/s |
| Each `.pte` RAM footprint | < 1.5 GB | < 1.2 GB |
| NPU utilization during inference | > 60% | > 75% |

### P2 — Android Backend

| Deliverable | Target | Stretch |
|-------------|--------|---------|
| Single-device e2e latency | < 6,000 ms | < 4,000 ms |
| Distributed e2e latency | < 4,000 ms | < 2,500 ms |
| Consecutive runs without crash | 10 / 10 | 10 / 10 |
| Heap growth over 10 runs | < 50 MB | < 20 MB |

### P3 — Networking

| Deliverable | Target | Stretch |
|-------------|--------|---------|
| TCP connection time | < 3,000 ms | < 1,000 ms |
| Transfer latency p50 | < 100 ms | < 50 ms |
| Transfer latency p95 | < 200 ms | < 100 ms |
| Throughput | > 10 MB/s | > 30 MB/s |
| Float16 max absolute error | < 0.001 | < 0.0001 |
| Reconnection recovery time | < 5,000 ms | < 2,000 ms |

---

## Integration Test — Full Pipeline

Run after all unit and component tests pass. Go/no-go gate for the demo.

| ID | Test | Pass Threshold |
|----|------|----------------|
| INT-01 | End-to-end distributed inference with airplane mode on | Response latency < 4,000 ms; audio intelligible |
| INT-02 | 10 consecutive requests without failure | 10/10 success; heap stable |
| INT-03 | NPU active on both phones during inference | NPU > 60% on Phone A; > 50% on Phone B |
| INT-04 | Hotspot drop and recovery mid-session | Session resumes; no app crash |
| INT-05 | Activation data integrity through full pipeline | Top-1 token matches for ≥ 8/10 prompts |

---

## Risk Register

| Risk | Mitigation | Fallback |
|------|------------|----------|
| WiFi hotspot drops mid-demo | Test 20+ connections; auto-reconnect in HotspotManager | Switch to single-device mode (P2 pre-builds fallback APK) |
| Activation tensor > 200ms transfer | Quantize to float16; reduce seq_len to 128 | Reduce seq_len to 32 for demo |
| `.pte` won't load on loaner S25 Ultra | P1 validates against AI Hub before hackathon | Fall back to llama.cpp RPC backend |
| ExecuTorch Android build fails on-site | P2 pre-builds and pushes AAR to GitHub | Use pre-built debug APK |

---

## Branch Strategy

```
main                          ← Project overview, README, integration docs
├── p1/model-export           ← Model export, NPU optimization, .pte files
├── p2/android-backend        ← ExecuTorch runtime, inference pipeline
└── p3/networking             ← WiFi hotspot, TCP transport, activation passing
```

Each engineer works on their branch and merges to `main` at integration checkpoints.

---

## Hardware

- **Samsung Galaxy S25 Ultra** (loaner, Snapdragon 8 Elite) — Phone A (coordinator)
- **Team Android devices** — Phone B (worker)

---

## Open Questions

- Will the hackathon provide more than one S25 Ultra per team?
- Does Qualcomm AI Hub support profiling of split `.pte` models?
- Which TTS model gives best Arabic/Swahili quality at ExecuTorch `.pte` size?
- GitHub Copilot usage logging — document all Copilot-assisted code sections for Copilot Build Award eligibility.

---

## License

Hackathon project — Qualcomm × Meta ExecuTorch Hackathon 2026.
