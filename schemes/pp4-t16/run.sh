#!/usr/bin/env bash
# ============================================================================
# 方案 B · PP4 t16（终稿 2026-11-16 19:10）—— 4 卡流水线 GPU1,2,3,4，GPU0 给 ComfyUI
#
# 由方案 A 派生，差异只有 4 处：
#   1) --pipeline-parallel-size 4，pipeline_layers [6,6,7,21]
#      （硬约束：19~39 层必须在最后一段，故前 3 段合计只能是 19 层）
#   2) CUDA_VISIBLE_DEVICES=1,2,3,4（2026-11-16 A/B 实测 GPU4 做 rank3 与 GPU0 无差；
#      GPU0 永远留给 ComfyUI，别改回 0,1,2,3）
#   3) cpu_threads decode 档 4 → 16
#   4) 端口与容器名（端口 2026-11-15 起统一为 9010）
#
# 终稿参数（2026-11-16 t1m-099 实测，后 3 轮均值）：wall 50.43 / decode 57.55
#   --gpu-memory-utilization 0.99（⚠️ 原值。0.92→槽位3320；0.97 两档证伪 30.8/46.23，勿用）
#   --max-model-len/--kv-cache-tokens 1048576（1M；131k→1M 上下文×8 只慢 3.5%）
#   KV 档位对比：131k=52.25 / 512k=51.78 / 1M=50.43（详见 bench/README.md 3.1b）
#
# 用法：./run.sh [--dry-run] [--stop] [--port N] [--model PATH] [--gpus 1,2,3,4]
# 环境变量：DSV41_MODEL / DSV41_IMAGE / DSV41_PORT / DSV41_GPUS / DSV41_NAME / DSV41_CACHE
# ============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../scripts/lib.sh
. "$HERE/../../scripts/lib.sh"
parse_args "$@"

IMAGE=${DSV41_IMAGE:-18gogogo/170hx4-256r-dsv41-1017-flash:sm80}
NAME=${DSV41_NAME:-dsv41-pp4-t16}
MODEL=${DSV41_MODEL:-/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official}
CACHE_BASE=${DSV41_CACHE:-${HOME:-/root}/.cache/dsv41}
# 服务规矩（2026-11-15 定）：所有方案统一 9010 端口、无密钥，客户端不改地址
PORT=${DSV41_PORT:-9010}
GPUS=${DSV41_GPUS:-1,2,3,4}

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
  --log-driver=journald
  "$IMAGE"
  /workspace/vllm-backport/.venv/bin/python -m vllm.entrypoints.cli.main serve /model
    --served-model-name deepseek-v4.1-flash-1017 DeepSeek-V4.1-Flash-1017
    --host 0.0.0.0 --port "$PORT"
    --pipeline-parallel-size 4 --tensor-parallel-size 1
    --gpu-memory-utilization 0.99
    --max-num-seqs 4
    --max-num-batched-tokens 4096
    --limit-mm-per-prompt '{"image": 128}'
    # 2026-11-15 08:58: 1048576 -> 524288（二改）。一改 1M 实测覆盖率 47.4%（3640/7680），
    # 比基线掉 15 个点：大头不是 KV（fp8 后仅 ~3.5 GiB），是 V2 workspace 随 max_model_len
    # 天花板预留（1M 时 2~4 GiB/卡，合计 ~10 GiB）。社区对照：aiultrayang PP4 同规格跑
    # 512k 覆盖也只剩 43.33%；LocalFrameLab 直接用 65536 保速度。512k 档预计 ~4030 槽（52.5%），
    # 满足用户 ≥512k 要求且比上游同档高 9 个点。
    --max-model-len 1048576
    --kv-cache-tokens 1048576
    # KV 格式：不传 --kv-cache-dtype，auto 默认即 fp8_ds_mla（实测 1M 池仅 3.47 GiB）。
    #    ⚠️ 千万别显式传 fp8_ds_mla——会触发共享 KV 组跨 rank 检查直接 NotImplementedError。
    --enable-prefix-caching
    --enable-prompt-tokens-details
    --reasoning-parser deepseek_v41
    --tool-call-parser deepseek_v41
    --enable-auto-tool-choice
    --default-chat-template-kwargs '{"reasoning_effort":50}'
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"enable_adaptive_verification":true}'
    --additional-config '{"deepseek_v41_hybrid":{"pipeline_layers":[6,6,7,21],"cpu_threads":[28,16],"cache_decode_interval":64,"engram_cache_gib":4,"host_cache_gib":8,"engram_storage":"ssd","route_tail_skip":0}}'
)

if [ "$ACTION" = stop ]; then stop_container "$NAME"; exit 0; fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

if [ "$DRY_RUN" = 1 ]; then
  show_cmd "${CMD[@]}"
  exit 0
fi

log "启动容器 $NAME（4 卡 PP4，GPU4 留空，加载约 9 分钟）"
"${CMD[@]}"
