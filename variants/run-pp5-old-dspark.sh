#!/usr/bin/env bash
# 实验：旧引擎 + dspark（只放 1 层 MTP，不借用新引擎的专家下放）
#
# 与 run-pp5-ram.sh 的差异只有 5 处，其余逐行一致：
#   1. VLLM_PP5_LAZY_RECEIVE 1 -> 0      （dspark 必须用 V2 runner，两者互斥）
#   2. VLLM_USE_V2_MODEL_RUNNER 0 -> 1
#   3. gpu-memory-utilization 0.97 -> 0.99 （MTP 一层要 2.47 GiB，余量只有 1.93）
#   4. --speculative-config dspark k=1    （k=1 时每步最多 2 token）
#   5. --hf-overrides n_mtp_layers=1      （只建 mtp.0，省掉 mtp.1/mtp.2 的 4.92 GiB）
#
# 显存账：0.99*64=63.36 GiB，权重 58.2 GiB，余 5.16 GiB；MTP 一层 2.47 GiB 后
#       还剩 2.69 GiB 给 KV/workspace（原 ram 模式 KV 余 1.93 GiB）。
# engram 与 ram 模式一致（全主机内存），不引入 SSD 变量。
set -euo pipefail

IMAGE=local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102
NAME=dsv41-pp5-olddspark
MODEL=/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official
CACHE=/run/media/xpz/models/deploy-llm/isolated-aiultrayang-pp5-ssd-async-20261031/cache

docker rm -f "$NAME" 2>/dev/null || true

docker run -d --name "$NAME" \
  --gpus all \
  --network host \
  --shm-size 16g \
  --ulimit memlock=-1 \
  --ulimit nofile=1048576:1048576 \
  -e TMPDIR=/tmp \
  -e VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,4 \
  -e VLLM_PP5_LAZY_RECEIVE=0 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
  -e PYTHONUNBUFFERED=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8 \
  -e NCCL_IB_DISABLE=1 \
  -e VLLM_PP5_SHARED_MAPPING=1 \
  -e NCCL_PROTO=Simple \
  -e NCCL_ALGO=Ring \
  -e NCCL_P2P_DISABLE=1 \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e TRITON_CACHE_DIR=$CACHE/triton \
  -e XDG_CACHE_HOME=$CACHE \
  -e HF_HUB_OFFLINE=1 \
  -v "$MODEL":"$MODEL":ro \
  -v "$CACHE":"$CACHE":rw \
  -v /home/xpz/.cache/deepseek-v41-official:/home/xpz/.cache/deepseek-v41-official:ro \
  --restart no \
  "$IMAGE" \
  "$MODEL" \
    --host=0.0.0.0 \
    --port=9010 \
    --served-model-name=deepseek-v4.1-flash-1017 \
    --pipeline-parallel-size=5 \
    --tensor-parallel-size=1 \
    --max-model-len=262144 \
    --max-num-seqs=4 \
    --max-num-batched-tokens=2048 \
    --gpu-memory-utilization=0.99 \
    --kv-cache-memory-bytes=2147483648 \
    --kv-cache-dtype=fp8_ds_mla \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":16}' \
    --safetensors-load-strategy=lazy \
    --hf-overrides '{"n_mtp_layers":1}' \
    --speculative-config '{"method":"dspark","num_speculative_tokens":1}' \
    --engram-config '{"cpu_offload":true,"storage":"ram","engram_cache_gib":8}' \
    --tokenizer-mode=deepseek_v41 \
    --enable-auto-tool-choice \
    --tool-call-parser=deepseek_v41 \
    --reasoning-parser=deepseek_v41 \
    --enable-prompt-tokens-details \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'
