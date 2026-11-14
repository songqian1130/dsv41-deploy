#!/usr/bin/env bash
# ============================================================================
# 方案 A · PP5 fast —— 新引擎 V2 + dspark k=5 + cpu_threads [28,4] + engram SSD
#
# 这是 2026-11-14 的线上生产配置，单流四档均值 82.5 tok/s。
#
# 用法（一般不用直接调它，用同目录的 deploy.sh）：
#   ./run.sh                      # 起容器
#   ./run.sh --dry-run            # 只打印 docker 命令
#
# 可覆盖的环境变量：
#   DSV41_MODEL   模型目录（必须是 HF 格式，含 config.json + *.safetensors）
#   DSV41_IMAGE   镜像（默认 18gogogo/170hx4-256r-dsv41-1017-flash:sm80）
#   DSV41_PORT    端口（默认 9010）
#   DSV41_GPUS    卡号（默认 0,1,2,3,4，必须 5 张）
#   DSV41_NAME    容器名（默认 dsv41-pp5-fast）
#   DSV41_CACHE   vLLM/triton 编译缓存目录（默认 ~/.cache/dsv41）
# ============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../scripts/lib.sh
. "$HERE/../../scripts/lib.sh"   # 只借用 show_cmd / DRY_RUN / 日志函数
parse_args "$@"

IMAGE=${DSV41_IMAGE:-18gogogo/170hx4-256r-dsv41-1017-flash:sm80}
NAME=${DSV41_NAME:-dsv41-pp5-fast}
MODEL=${DSV41_MODEL:-/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official}
CACHE_BASE=${DSV41_CACHE:-$HOME/.cache/dsv41}
PORT=${DSV41_PORT:-9010}
GPUS=${DSV41_GPUS:-0,1,2,3,4}

[ "$DRY_RUN" = 1 ] || { mkdir -p "$CACHE_BASE/vllm" "$CACHE_BASE/triton"; }

CMD=(docker run -d --name "$NAME"
  --gpus all
  --ipc=host
  --cap-add=SYS_NICE
  --network host
  -e CUDA_VISIBLE_DEVICES="$GPUS"
  -e DSV41_HOST_STAGING_CACHE=1
  -e NCCL_P2P_DISABLE=1
  -e NCCL_PROTO=Simple
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
  -e VLLM_CACHE_ROOT=/tmp/dsv41-cache
  -e TRITON_CACHE_DIR=/tmp/dsv41-triton
  -e HF_HUB_OFFLINE=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=36000
  -e CUDA_DEVICE_MAX_CONNECTIONS=16
  -v "$MODEL":/model:ro
  -v "$CACHE_BASE/vllm":/tmp/dsv41-cache
  -v "$CACHE_BASE/triton":/tmp/dsv41-triton
  --restart no
  "$IMAGE"
  /workspace/vllm-backport/.venv/bin/python -m vllm.entrypoints.cli.main serve /model
    --served-model-name deepseek-v4.1-flash-1017 DeepSeek-V4.1-Flash-1017
    --host 0.0.0.0 --port "$PORT"
    --pipeline-parallel-size 5 --tensor-parallel-size 1
    --gpu-memory-utilization 0.99
    --max-num-seqs 4
    --max-num-batched-tokens 4096
    --limit-mm-per-prompt '{"image": 16}'
    # 2026-11-15: 131072 -> 1048576。用户要求总上下文 ≥512k。
    # 依据：上游 aiultrayang 的 PP4 参考配置就是 kv_cache_tokens=1048576 / max_model_len=1048576，
    # 且其启动日志显示 5 个 rank 的 KV 预算合计仅 ~1.8 GiB（hybrid_runtime.py "Hybrid cache budget" 行）。
    # ⚠️ 未上机验证：重启后必须看两处 —— ① 各 rank "Hybrid cache budget: KV=" 行；
    #    ② 专家覆盖率（93.96%）。若覆盖率明显下滑，先退回 262144 再试。
    --max-model-len 1048576
    --kv-cache-tokens 1048576
    # KV 格式：不传 --kv-cache-dtype，auto 默认即 DeepSeek 官方 fp8_ds_mla（日志
    #    "Using DeepSeek's fp8_ds_mla KV cache format"，1M 池仅 ~3.5 GiB）。⚠️ 千万别显式
    #    传 fp8_ds_mla——会触发共享 KV 组跨 rank 检查直接 NotImplementedError（实测，见
    #    docs/rejected.md 第 9 条）。
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --reasoning-parser deepseek_v41
    --tool-call-parser deepseek_v41
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"reasoning_effort":50}'
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}'
    --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[5,5,5,4,21],"cpu_threads":[28,4],"cache_decode_interval":64,"engram_cache_gib":4,"host_cache_gib":8,"engram_storage":"ssd","route_tail_skip":0}}'
)

if [ "$ACTION" = stop ]; then stop_container "$NAME"; exit 0; fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

if [ "$DRY_RUN" = 1 ]; then
  show_cmd "${CMD[@]}"
  exit 0
fi

log "启动容器 $NAME（5 卡 PP5，加载约 9 分钟）"
"${CMD[@]}"
