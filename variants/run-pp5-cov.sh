#!/usr/bin/env bash
# fast 模式 + 100% 专家驻留实验（唯一改动：挂载修补后的 hybrid.py）
#
# 背景：plan_expert_cache() 把每层上限写成 bank_size - 1 = 383，按 8 对齐后恒为 376，
# 导致 layer20~37 每层固定有 8 个专家留在主机，每步要进 20 次 CPU MoE 调用。
# 修补内容（3 处）：
#   190: target = min(bank_size, ...)          # 原 bank_size - 1
#   202: maximum = bank_size // 8 * 8          # 原 (bank_size - 1) // 8 * 8
#   227: count + 8 <= bank_size                # 原 count + 8 < bank_size
# 其余配置与 fast 完全一致（KV 131072、cpu_threads [28,4]、dspark k=3）。
# 只挂一个只读文件，镜像不动；回滚 = 重新起 fast 容器。
set -euo pipefail
IMAGE=18gogogo/170hx4-256r-dsv41-1017-flash:sm80
NAME=dsv41-pp5-cov
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
  -v "$HOME/dsv41-repo-test/hybrid-cov.py":/workspace/vllm-backport/vllm/models/deepseek_v4_1/hybrid.py:ro \
  --restart no \
  "$IMAGE" \
  /workspace/vllm-backport/.venv/bin/python -m vllm.entrypoints.cli.main serve /model \
    --served-model-name deepseek-v4.1-flash-1017 DeepSeek-V4.1-Flash-1017 \
    --host 0.0.0.0 --port 9011 \
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
    --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[5,5,5,4,21],"cpu_threads":[28,4],"cache_decode_interval":64,"engram_cache_gib":4,"host_cache_gib":8,"engram_storage":"ssd","route_tail_skip":0}}'
