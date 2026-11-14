#!/usr/bin/env bash
# ============================================================================
# 方案 C · 旧 PP5 —— V1 引擎（无 dspark）+ 专家 100% 在卡 + engram SSD 或 RAM
#
# 2026-11-14 12:00 之前的线上方案，单流 43.5 tok/s。现仅作兜底与对照。
#
# 两种模式只差 engram 的存放位置：
#   DSV41_ENGRAM=ssd（默认）  22k 冷 TTFT 11.9 s，主机内存 ~30 GiB
#   DSV41_ENGRAM=ram          22k 冷 TTFT 6.45 s，主机内存 **189 GiB pinned**
# decode 阶段完全不读 engram，所以两种模式的 tok/s 一样（43.5）。
#
# ⚠ 镜像 `local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102`
#   只存在于原机器，不在任何 registry。换机器必须先导出导入：
#     docker save <image> | gzip > dsv41-legacy.tar.gz
#     scp dsv41-legacy.tar.gz target: && ssh target 'gunzip -c dsv41-legacy.tar.gz | docker load'
#
# ⚠ 这个镜像的 ENTRYPOINT 已经是 [vllm, serve]，所以参数里不要再写 serve，直接跟模型路径。
#
# 用法：./run.sh [--dry-run] [--stop] [--port N] [--model PATH] [--engram ssd|ram]
# 环境变量：DSV41_MODEL / DSV41_IMAGE / DSV41_PORT / DSV41_GPUS / DSV41_NAME / DSV41_CACHE / DSV41_ENGRAM
# ============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../scripts/lib.sh
. "$HERE/../../scripts/lib.sh"

ENGRAM=${DSV41_ENGRAM:-ssd}
# 先把本方案独有的 --engram 挑出来，剩下的交给公共解析器
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --engram) ENGRAM=${2:?--engram 需要 ssd 或 ram}; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
parse_args ${ARGS[@]+"${ARGS[@]}"}
case "$ENGRAM" in ssd|ram) ;; *) die "--engram 只能是 ssd 或 ram（收到 $ENGRAM）" ;; esac

IMAGE=${DSV41_IMAGE:-local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102}
NAME=${DSV41_NAME:-dsv41-pp5-legacy}
MODEL=${DSV41_MODEL:-/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official}
CACHE=${DSV41_CACHE:-$HOME/.cache/dsv41-legacy}
PORT=${DSV41_PORT:-9010}
GPUS=${DSV41_GPUS:-0,1,2,3,4}

[ "$DRY_RUN" = 1 ] || { mkdir -p "$CACHE/triton"; }

# 旧镜像会去这个路径找 engram 的运行时缓存，有就挂，没有就跳过（不阻断部署）
OPT_MOUNT=()
if [ -d "$HOME/.cache/deepseek-v41-official" ]; then
  OPT_MOUNT=(-v "$HOME/.cache/deepseek-v41-official":"$HOME/.cache/deepseek-v41-official":ro)
fi

CMD=(docker run -d --name "$NAME"
  --gpus all
  --network host
  --shm-size 16g
  --ulimit memlock=-1
  --ulimit nofile=1048576:1048576
  -e TMPDIR=/tmp
  -e VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,0
  -e VLLM_PP5_LAZY_RECEIVE=1
  -e TRANSFORMERS_OFFLINE=1
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e VLLM_USE_V2_MODEL_RUNNER=0
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=0
  -e PYTHONUNBUFFERED=1
  -e NCCL_P2P_LEVEL=SYS
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1
  -e VLLM_PP_LAYER_PARTITION=8,8,8,8,8
  -e NCCL_IB_DISABLE=1
  -e VLLM_PP5_SHARED_MAPPING=1
  -e NCCL_PROTO=Simple
  -e NCCL_ALGO=Ring
  -e NCCL_P2P_DISABLE=1
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn
  -e TRITON_CACHE_DIR="$CACHE/triton"
  -e XDG_CACHE_HOME="$CACHE"
  -e HF_HUB_OFFLINE=1
  -e CUDA_VISIBLE_DEVICES="$GPUS"
  -v "$MODEL":"$MODEL":ro
  -v "$CACHE":"$CACHE":rw
  ${OPT_MOUNT[@]+"${OPT_MOUNT[@]}"}
  --restart no
  "$IMAGE"
  "$MODEL"
    --host=0.0.0.0
    --port="$PORT"
    --served-model-name=deepseek-v4.1-flash-1017
    --pipeline-parallel-size=5
    --tensor-parallel-size=1
    # 2026-11-15: 262144 -> 524288；并显式加 kv-cache-tokens（旧引擎支持，aiultrayang run.log 同款）。
    # ⚠️ 未上机验证：C 全部专家在卡、util 0.97，KV 多吃 ~1.8 GiB 是否放得下要看重启日志；
    #    起不来就把 kv-cache-tokens 删掉（回到自动分配）或退 262144。
    --max-model-len=524288
    --max-num-seqs=4
    --max-num-batched-tokens=2048
    --gpu-memory-utilization=0.97
    --kv-cache-dtype=fp8_ds_mla
    --kv-cache-tokens=524288
    --trust-remote-code
    --disable-custom-all-reduce
    --enable-prefix-caching
    --limit-mm-per-prompt '{"image":16}'
    --safetensors-load-strategy=lazy
    --engram-config "{\"cpu_offload\":true,\"storage\":\"$ENGRAM\",\"engram_cache_gib\":8}"
    --tokenizer-mode=deepseek_v41
    --enable-auto-tool-choice
    --tool-call-parser=deepseek_v41
    --reasoning-parser=deepseek_v41
    --enable-prompt-tokens-details
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'
)

if [ "$ACTION" = stop ]; then stop_container "$NAME"; exit 0; fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

if [ "$DRY_RUN" = 1 ]; then
  show_cmd "${CMD[@]}"
  exit 0
fi

log "启动容器 $NAME（V1 引擎，engram=$ENGRAM）"
"${CMD[@]}"
