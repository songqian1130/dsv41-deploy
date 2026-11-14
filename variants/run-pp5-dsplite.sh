#!/usr/bin/env bash
# 新引擎 + dspark，但只保留 1 层 MTP（省下 mtp.1/mtp.2 的 4.92 GiB）
#
# 与 run-pp5-prod.sh 的差异只有 3 处：
#   1. 容器名
#   2. --hf-overrides '{"n_mtp_layers":1}'  → 少建 2 层 MTP
#   3. num_speculative_tokens 3 -> 1        （1 层 MTP 只对应 1 个 draft）
#
# 目的：hybrid runtime 是按实测余量自动决定 coverage 的，MTP 少吃 4.92 GiB
# 就有 4.92 GiB 的专家能回到卡上，步时应该从 66ms 明显下降。
set -euo pipefail
IMAGE=18gogogo/170hx4-256r-dsv41-1017-flash:sm80
NAME=dsv41-pp5-dsplite
MODEL=/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official
CACHE_BASE=/run/media/xpz/models/dsv41-cache

mkdir -p "$CACHE_BASE/vllm" "$CACHE_BASE/triton"

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --ipc=host \
  --cap-add=SYS_NICE \
  --network host \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
  -e DSV41_HOST_STAGING_CACHE=1 \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_PROTO=Simple \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e VLLM_CACHE_ROOT=/tmp/dsv41-cache \
  -e TRITON_CACHE_DIR=/tmp/dsv41-triton \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=36000 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=16 \
  -v "$MODEL":/model:ro \
  -v "$CACHE_BASE/vllm":/tmp/dsv41-cache \
  -v "$CACHE_BASE/triton":/tmp/dsv41-triton \
  --restart no \
  "$IMAGE" \
  /workspace/vllm-backport/.venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name deepseek-v4.1-flash-1017 DeepSeek-V4.1-Flash-1017 \
    --host 0.0.0.0 --port 9010 \
    --pipeline-parallel-size 5 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.99 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --limit-mm-per-prompt '{"image": 16}' \
    --max-model-len 262144 \
    --kv-cache-tokens 524288 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"reasoning_effort":50}' \
    --hf-overrides '{"n_mtp_layers":1}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":1,"enable_adaptive_verification":true}' \
    --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[5,5,5,4,21],"cpu_threads":[28,28],"cache_decode_interval":64,"engram_cache_gib":4,"host_cache_gib":8,"engram_storage":"ssd","route_tail_skip":0}}'
