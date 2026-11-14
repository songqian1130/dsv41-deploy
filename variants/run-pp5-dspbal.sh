#!/usr/bin/env bash
# 新引擎 + dspark k=3（与生产完全一致），唯一改动：KV 池从 524288 token 降到 131072
#
# 目的：验证"新 PP5 慢不是因为平台，而是因为 KV 池吃掉了本该留给专家的显存"。
# 生产配置里 --kv-cache-tokens 524288，而 max-num-seqs 只有 4、max-model-len 262144。
# 若把 KV 降到 131072，空出来的 HBM 应该能让更多专家回到卡上（coverage 从 93.54% 上升），
# 步时随之下降。
set -euo pipefail
IMAGE=18gogogo/170hx4-256r-dsv41-1017-flash:sm80
NAME=dsv41-pp5-dspbal
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
    --max-model-len 131072 \
    --kv-cache-tokens 131072 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"reasoning_effort":50}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}' \
    --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[8,8,8,8,8],"cpu_threads":[28,28],"cache_decode_interval":64,"engram_cache_gib":4,"host_cache_gib":8,"engram_storage":"ssd","route_tail_skip":0}}'
