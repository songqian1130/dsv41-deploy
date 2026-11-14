#!/usr/bin/env bash
# ============================================================================
# 方案 B · PP4 t16 —— 4 卡流水线（GPU0-3），GPU4 完全留给 ComfyUI
#
# 由方案 A 派生，差异只有 4 处：
#   1) --pipeline-parallel-size 4，pipeline_layers [6,6,7,21]
#      （硬约束：19~39 层必须在最后一段，故前 3 段合计只能是 19 层）
#   2) CUDA_VISIBLE_DEVICES=0,1,2,3
#   3) cpu_threads decode 档 4 → 16
#      少一张卡后专家驻留率 93.96% → 56.25%，CPU 路由 0.27% → 6.8%（25 倍），
#      4 线程会被打满成为瓶颈；16 与 20 做过配对复测，差异在噪声内，故选 16。
#   4) 端口与容器名（默认 9015，可和方案 A 并存调试）
#
# 实测（2026-11-14，cases_v1 三轮 nonce 101/102/103 均值）：
#   code 74.6 / json 64.4 / zh 47.7 / math 56.8 → 均值 60.9 tok/s
#   22k 冷 TTFT 13.71 s（方案 A 为 9.40 s）· GPU4 全程 0 MiB
#
# 用法：./run.sh [--dry-run] [--stop] [--port N] [--model PATH] [--gpus 0,1,2,3]
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
CACHE_BASE=${DSV41_CACHE:-$HOME/.cache/dsv41}
# 服务规矩（2026-11-15 定）：所有方案统一 9010 端口、无密钥，客户端不改地址
PORT=${DSV41_PORT:-9010}
GPUS=${DSV41_GPUS:-0,1,2,3}

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
    --pipeline-parallel-size 4 --tensor-parallel-size 1
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
    # ⚠️ fp8_ds_mla 不可用于 PP 分层：V4.1 的 compressed-KV 共享组不能跨 rank，
    #    会报 NotImplementedError: PP splits inside a v4.1 kv-sharing group are not
    #    supported（2026-11-15 01:45 实测，B 的 [6,6,7,21] 在 layer 20 炸）。除非
    #    重排分层对齐共享组边界，否则只能 bf16。
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
