#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/native/out"
mkdir -p "$OUT_DIR"

if [[ -z "${ANDROID_NDK:-}" ]]; then
  if [[ -n "${ANDROID_NDK_HOME:-}" ]]; then
    ANDROID_NDK="$ANDROID_NDK_HOME"
  elif [[ -n "${ANDROID_NDK_ROOT:-}" ]]; then
    ANDROID_NDK="$ANDROID_NDK_ROOT"
  else
    echo "Set ANDROID_NDK to your Android NDK path." >&2
    exit 2
  fi
fi

HOST_TAG="darwin-x86_64"
if [[ "$(uname -m)" == "arm64" && -d "$ANDROID_NDK/toolchains/llvm/prebuilt/darwin-arm64" ]]; then
  HOST_TAG="darwin-arm64"
fi

CXX="$ANDROID_NDK/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android26-clang++"
if [[ ! -x "$CXX" ]]; then
  echo "Could not find Android clang++ at $CXX" >&2
  exit 2
fi

"$CXX" \
  -std=c++17 \
  -O2 \
  -Wall \
  -Wextra \
  -static-libstdc++ \
  "$ROOT_DIR/native/executor_bridge_worker.cpp" \
  -o "$OUT_DIR/beacon_executor_bridge_worker"

echo "$OUT_DIR/beacon_executor_bridge_worker"
