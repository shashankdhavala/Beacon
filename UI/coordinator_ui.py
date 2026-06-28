#!/usr/bin/env python3
"""
Local web UI for the Beacon coordinator.

Run from the repo root:
  python3 UI/coordinator_ui.py --route "1=10.154.197.31:9000,2=10.154.197.225:9000" --port 8081

Then open:
  http://127.0.0.1:8081
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from android_gpt2_coordinator import execute_gpt2_route
from mac_coordinator import execute_text_route, parse_route


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Beacon</title>
  <style>
    :root {
      --bg: #f4f6f5;
      --panel: #ffffff;
      --ink: #16211f;
      --muted: #66736f;
      --line: #d8e0dd;
      --accent: #0b6e4f;
      --accent-dark: #084f3a;
      --accent-soft: #dceee7;
      --ok: #067647;
      --danger: #b42318;
      --soft: #eef6f2;
      --warm: #f7f1e7;
      --shadow: 0 18px 46px rgba(20, 36, 32, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(11, 110, 79, 0.13), rgba(247, 241, 231, 0.72) 360px, rgba(244, 246, 245, 0) 620px),
        var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .shell {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 46px;
    }

    header {
      min-height: 172px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
      padding: 26px 28px;
      border: 1px solid rgba(216, 224, 221, 0.82);
      border-radius: 8px;
      background:
        linear-gradient(135deg, rgba(255,255,255,0.92), rgba(238,246,242,0.88)),
        var(--panel);
      box-shadow: var(--shadow);
    }

    h1 {
      margin: 0;
      font-size: 40px;
      line-height: 1.08;
      letter-spacing: 0;
    }

    .subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.45;
    }

    .eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 820;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .eyebrow::before {
      content: "";
      width: 24px;
      height: 2px;
      border-radius: 999px;
      background: var(--accent);
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 11px;
      background: rgba(255, 255, 255, 0.74);
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      box-shadow: 0 8px 18px rgba(20, 36, 32, 0.06);
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #b54708;
    }

    .dot.ok { background: var(--ok); }
    .dot.err { background: var(--danger); }

    .app {
      display: grid;
      grid-template-columns: minmax(340px, 410px) 1fr;
      gap: 18px;
      align-items: start;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      padding: 18px 19px 13px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
    }

    .panel-head h2 {
      margin: 0;
      font-size: 17px;
      line-height: 1.2;
    }

    .panel-head p {
      margin: 7px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    form {
      padding: 20px;
      display: grid;
      gap: 16px;
    }

    label {
      display: grid;
      gap: 8px;
      color: #34433f;
      font-size: 13px;
      font-weight: 700;
    }

    textarea {
      min-height: 184px;
      width: 100%;
      resize: vertical;
      border: 1px solid #c9d4d0;
      border-radius: 8px;
      padding: 15px 16px;
      color: var(--ink);
      background: #fffdf9;
      font: inherit;
      font-size: 16px;
      line-height: 1.48;
      outline: none;
    }

    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(11, 110, 79, 0.14);
    }

    button {
      height: 46px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      font-size: 15px;
      font-weight: 780;
      cursor: pointer;
      box-shadow: 0 10px 20px rgba(11, 110, 79, 0.18);
    }

    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: 0.58; cursor: wait; }

    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfd;
    }

    .result {
      min-height: 560px;
    }

    .empty {
      min-height: 420px;
      display: grid;
      place-items: center;
      padding: 44px 24px;
      color: var(--muted);
      text-align: center;
      line-height: 1.6;
    }

    .empty-card {
      max-width: 360px;
      border: 1px dashed #bfd0ca;
      border-radius: 8px;
      padding: 24px;
      background: #fbfcfd;
    }

    .empty-title {
      color: var(--ink);
      font-weight: 760;
      margin-bottom: 7px;
    }

    .answer {
      padding: 20px;
      background:
        linear-gradient(180deg, rgba(238,246,242,0.92), rgba(255,255,255,0.92)),
        var(--soft);
    }

    .answer .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      margin-bottom: 8px;
    }

    .answer .text {
      padding: 16px;
      border: 1px solid #cfe1da;
      border-radius: 8px;
      background: #ffffff;
      font-size: 24px;
      line-height: 1.28;
      font-weight: 780;
      overflow-wrap: anywhere;
    }

    .tabs {
      display: flex;
      gap: 6px;
      padding: 12px 14px 0;
      border-bottom: 1px solid var(--line);
      background: #fbfcfb;
    }

    .tab {
      height: 36px;
      border: 1px solid transparent;
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      padding: 0 14px;
      background: transparent;
      color: var(--muted);
      font-size: 13px;
      font-weight: 780;
      box-shadow: none;
    }

    .tab:hover {
      background: #f3f7f5;
      color: var(--accent-dark);
    }

    .tab.active {
      background: #fff;
      border-color: var(--line);
      color: var(--ink);
    }

    .tab-panel {
      display: none;
    }

    .tab-panel.active {
      display: block;
    }

    .metrics {
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px;
      background: #fbfcfd;
    }

    .metric .name {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 7px;
    }

    .metric .value {
      font-size: 19px;
      font-weight: 800;
      line-height: 1.1;
    }

    .route {
      padding: 0 18px 16px;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: #fff;
      color: #34433f;
      font-size: 12px;
      font-weight: 650;
    }

    .path {
      padding: 0 18px 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .node {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid #cfe1da;
      border-radius: 999px;
      padding: 7px 11px;
      background: #fff;
      color: #243b35;
      font-size: 12px;
      font-weight: 760;
    }

    .node strong {
      display: inline-grid;
      place-items: center;
      width: 22px;
      height: 22px;
      border-radius: 50%;
      background: var(--accent-soft);
      color: var(--accent-dark);
      font-size: 12px;
    }

    .arrow {
      color: #8c9b97;
      font-size: 13px;
      font-weight: 760;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    th, td {
      padding: 13px 15px;
      border-top: 1px solid var(--line);
      text-align: left;
      vertical-align: middle;
    }

    th {
      color: var(--muted);
      font-size: 12px;
      background: #fbfcfd;
      font-weight: 780;
    }

    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
    }

    .error {
      margin: 18px;
      border: 1px solid rgba(180, 35, 24, 0.28);
      background: #fff4f2;
      color: var(--danger);
      border-radius: 8px;
      padding: 14px;
      white-space: pre-wrap;
      line-height: 1.45;
    }

    .sample-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .sample {
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      color: #34433f;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }

    .sample:hover {
      border-color: var(--accent);
      color: var(--accent-dark);
    }

    @media (max-width: 860px) {
      header { align-items: flex-start; flex-direction: column; }
      .app { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .answer .text { font-size: 21px; }
      h1 { font-size: 34px; }
      .tabs { overflow-x: auto; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <div class="eyebrow">Offline Mesh</div>
        <h1>Beacon</h1>
        <div class="subtitle">A coordinator routes prompts through nearby worker shards and collects the result locally.</div>
      </div>
      <div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Idle</span></div>
    </header>

    <div class="app">
      <section>
        <div class="panel-head">
          <h2>Ask The Local Mesh</h2>
          <p>Enter a prompt and send it through the active shard topology.</p>
        </div>
        <form id="promptForm">
          <label>
            Prompt
            <textarea id="prompt" spellcheck="true">I am batman</textarea>
          </label>
          <div class="sample-row">
            <button class="sample" type="button" data-sample="I am batman">I am batman</button>
            <button class="sample" type="button" data-sample="Explain gravity simply">Explain gravity simply</button>
            <button class="sample" type="button" data-sample="Translate hello">Translate hello</button>
          </div>
          <button id="runButton" type="submit">Run Through Shards</button>
          <div class="hint">Launch with <code>--mode gpt2</code> to run one-token hidden states through the on-device shard runtime.</div>
        </form>
      </section>

      <section class="result">
        <div class="panel-head">
          <h2>Coordinator Result</h2>
          <p>Prompt output and per-token receipts from the shard route.</p>
        </div>
        <div id="output" class="empty">
          <div class="empty-card">
            <div class="empty-title">Ready for a local run</div>
            <div>Start both worker phones, enter a prompt, and the coordinator will collect the shard outputs here.</div>
          </div>
        </div>
      </section>
    </div>
  </main>

  <script>
    const form = document.getElementById("promptForm");
    const promptInput = document.getElementById("prompt");
    const button = document.getElementById("runButton");
    const output = document.getElementById("output");
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const samples = document.querySelectorAll(".sample");

    function setStatus(text, cls = "") {
      statusText.textContent = text;
      statusDot.className = `dot ${cls}`;
    }

    function fmtMs(value) {
      return `${Number(value).toFixed(2)} ms`;
    }

    function renderError(message) {
      output.className = "";
      output.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function renderResult(data) {
      const isGpt2 = data.mode === "gpt2";
      const rows = data.trials.map((trial) => {
        if (isGpt2) {
          return `
            <tr>
              <td>${trial.step}</td>
              <td>${escapeHtml(trial.phase)} ${trial.kept ? "kept" : "discarded"}</td>
              <td><code>${escapeHtml(trial.inputTokenText)}</code></td>
              <td><code>${escapeHtml(trial.predictedTokenText)}</code></td>
              <td>${fmtMs(trial.latencyMs)}</td>
              <td>${trial.requestBytes} / ${trial.responseBytes}</td>
            </tr>
          `;
        }
        return `
          <tr>
            <td>${trial.step}</td>
            <td>route</td>
            <td><code>${escapeHtml(trial.token)}</code></td>
            <td><code>${escapeHtml(trial.received)}</code></td>
            <td>${fmtMs(trial.latencyMs)}</td>
            <td>${trial.requestBytes} / ${trial.responseBytes}</td>
          </tr>
        `;
      }).join("");

      const route = data.routeHops.map((hop, index) => `
        ${index > 0 ? '<span class="arrow">to</span>' : ''}
        <span class="node"><strong>${hop.shardId}</strong>Shard ${hop.shardId}</span>
      `).join("");

      output.className = "";
      output.innerHTML = `
        <div class="tabs" role="tablist" aria-label="Coordinator result views">
          <button class="tab active" type="button" data-tab="output">Output</button>
          <button class="tab" type="button" data-tab="metrics">Metrics</button>
        </div>
        <div class="tab-panel active" data-panel="output">
          <div class="answer">
            <div class="label">Output</div>
            <div class="text">${escapeHtml(data.output)}</div>
          </div>
        </div>
        <div class="tab-panel" data-panel="metrics">
          <div class="metrics">
            <div class="metric"><div class="name">Tokens</div><div class="value">${isGpt2 ? data.generatedTokenIds.length : data.tokens.length}</div></div>
            <div class="metric"><div class="name">p50</div><div class="value">${fmtMs(data.summary.p50Ms)}</div></div>
            <div class="metric"><div class="name">p95</div><div class="value">${fmtMs(data.summary.p95Ms)}</div></div>
            <div class="metric"><div class="name">Mode</div><div class="value">${isGpt2 ? "GPT-2" : "TEXT"}</div></div>
          </div>
          <div class="path"><span class="node"><strong>C</strong>Coordinator</span><span class="arrow">to</span>${route}<span class="arrow">to</span><span class="node"><strong>C</strong>Coordinator</span></div>
          <table>
            <thead>
              <tr>
                <th>Step</th>
                <th>Phase</th>
                <th>Input</th>
                <th>${isGpt2 ? "Predicted" : "Coordinator Received"}</th>
                <th>Latency</th>
                <th>Bytes</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      `;
      bindTabs();
    }

    function bindTabs() {
      const tabs = output.querySelectorAll(".tab");
      const panels = output.querySelectorAll(".tab-panel");
      tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          const target = tab.dataset.tab;
          tabs.forEach((item) => item.classList.toggle("active", item === tab));
          panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === target));
        });
      });
    }

    samples.forEach((sample) => {
      sample.addEventListener("click", () => {
        promptInput.value = sample.dataset.sample;
        promptInput.focus();
      });
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      setStatus("Running", "");
      output.className = "empty";
      output.textContent = "Coordinator is sending tokens through the shard route...";

      try {
        const response = await fetch("/api/text-route", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: promptInput.value })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          throw new Error(data.error || `HTTP ${response.status}`);
        }
        renderResult(data);
        setStatus("Complete", "ok");
      } catch (error) {
        renderError(error.message);
        setStatus("Error", "err");
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class CoordinatorUiHandler(BaseHTTPRequestHandler):
    server_version = "BeaconCoordinatorUI/0.2"
    route: str = ""
    timeout: float = 10.0
    mode: str = "text"
    artifact_dir: str = "artifacts/tiny-gpt2"
    max_new_tokens: int = 8

    def do_GET(self) -> None:
        if self.path not in ("/", "/index.html"):
            self.send_json({"error": "not found"}, status=404)
            return
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/text-route":
            self.send_json({"error": "not found"}, status=404)
            return

        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            payload = json.loads(body.decode("utf-8"))
            message = str(payload.get("message", ""))
            if self.mode == "gpt2":
                result = execute_gpt2_route(
                    route_arg=self.route,
                    prompt=message,
                    artifact_dir=self.artifact_dir,
                    max_new_tokens=self.max_new_tokens,
                    checksum="none",
                    timeout=self.timeout,
                )
            else:
                result = execute_text_route(
                    route_arg=self.route,
                    message=message,
                    checksum="none",
                    timeout=self.timeout,
                )
            self.send_json(result)
        except Exception as error:
            self.send_json({"ok": False, "error": str(error)}, status=400)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Beacon coordinator UI.")
    parser.add_argument("--route", required=True, help='Shard route, e.g. "1=10.0.0.11:9000,2=10.0.0.12:9000"')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--mode", choices=["text", "gpt2"], default="text")
    parser.add_argument("--artifact-dir", default="artifacts/tiny-gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    parse_route(args.route)
    CoordinatorUiHandler.route = args.route
    CoordinatorUiHandler.timeout = args.timeout
    CoordinatorUiHandler.mode = args.mode
    CoordinatorUiHandler.artifact_dir = args.artifact_dir
    CoordinatorUiHandler.max_new_tokens = args.max_new_tokens

    server = ThreadingHTTPServer((args.host, args.port), CoordinatorUiHandler)
    print(f"Beacon Coordinator UI running at http://{args.host}:{args.port}")
    print(f"Route: {args.route}")
    print(f"Mode: {args.mode}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
