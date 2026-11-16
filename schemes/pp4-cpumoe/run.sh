#!/usr/bin/env bash
# ============================================================================
# 方案 L · PP4 CPU-MoE（zebgop 版）—— 4 卡 PP4，9 层专家跑 CPU，engram 走 SSD
#
# 与方案 A/B 的根本差别：
#   1) V1 runner（VLLM_USE_V2_MODEL_RUNNER=0）—— CPU 专家 kernel 只支持 V1
#   2) 9 层路由专家不加载到 GPU（DSV41_SKIP_WEIGHT_RE 跳过 16-19 / 35-39 的 experts 权重），
#      由 overlay/hybrid/cpu_moe.cpp 在 CPU 上用 AVX2 直接算（16 线程）
#   3) engram 两表（~189 GiB）不进内存，走 engram_ssd 的 O_DIRECT pread 从 NVMe 读
#   4) 默认 GPU = 0,1,2,4：GPU3 空出来给 ComfyUI；GPU4 只承载 PP3 的小包流量，
#      避免它在"大块 H2D + kernel 并发"下把整机冻死（见 docs/gpu4-hang.md）
#
# 用法：./run.sh [--dry-run] [--stop] [--port N] [--model PATH] [--gpus 0,1,2,4]
# 环境变量：DSV41_MODEL / DSV41_IMAGE / DSV41_PORT / DSV41_GPUS / DSV41_NAME
#           DSV41_OVERLAY（overlay 目录，默认 ~/dsv41-run/repo/overlay）
#           DSV41_CPU_EXPERT_THREADS（默认 16）
# ============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../scripts/lib.sh
. "$HERE/../../scripts/lib.sh"
parse_args "$@"

IMAGE=${DSV41_IMAGE:-vllm/vllm-openai:deepseekv41-flash-0909}
NAME=${DSV41_NAME:-dsv41-pp4-cpumoe}
MODEL=${DSV41_MODEL:-/run/media/xpz/models/dsv41-hf}
# 模型目录结构：<MODEL>/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/local
SNAP_REL=${DSV41_SNAP_REL:-hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/local}
PORT=${DSV41_PORT:-9010}
GPUS=${DSV41_GPUS:-0,1,2,4}
OVERLAY=${DSV41_OVERLAY:-${HOME:-/root}/dsv41-run/repo/overlay}
THREADS=${DSV41_CPU_EXPERT_THREADS:-16}
PARTITION=${DSV41_PARTITION:-8,12,8,12}
CPU_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-16-19,35-39}
ENGRAM=${DSV41_ENGRAM_STORAGE:-ssd}
MAXLEN=${DSV41_MAXLEN:-131072}
BATCHED=${DSV41_BATCHED:-2048}
SEQS=${DSV41_SEQS:-4}
UTIL=${DSV41_UTIL:-0.97}
SPEC_N=${DSV41_SPEC:-3}

if [ ! -d "$OVERLAY/vllm" ] || [ ! -d "$OVERLAY/hybrid" ] || [ ! -d "$OVERLAY/engram_ssd" ]; then
  echo "overlay 不完整：$OVERLAY"
  echo "  取法：git clone https://github.com/zebgop-ops/dsv41flash-pp.git ~/dsv41-run/repo   # Apache-2.0"
  exit 1
fi
if [ ! -f "$MODEL/$SNAP_REL/model-00048-of-00048.safetensors" ]; then
  echo "checkpoint 不完整：$MODEL/$SNAP_REL 缺 shard 48（Engram 表所在）"; exit 1
fi

# 跳过两类权重：engram 表（ssd 模式由 SSD 后端读）+ CPU 层专家（CPU kernel 自己读）
CPU_RE=$(python3 - "$CPU_LAYERS" <<'PY'
import sys
out=set()
for part in sys.argv[1].split(','):
    part=part.strip()
    if not part or part.lower()=='none': continue
    if '-' in part:
        a,b=part.split('-'); out.update(range(int(a),int(b)+1))
    else: out.add(int(part))
print('|'.join(str(x) for x in sorted(out)))
PY
)
SKIP_RE='layers\.(1|14)\.engram\.embed\.'
[ -n "$CPU_RE" ] && SKIP_RE="$SKIP_RE|layers\.($CPU_RE)\.ffn\.experts\."

MOUNTS=()
while IFS= read -r f; do
  rel=${f#"$OVERLAY/vllm"/}
  MOUNTS+=(-v "$f:/usr/local/lib/python3.12/dist-packages/vllm/$rel:ro")
done < <(find "$OVERLAY/vllm" -type f -name '*.py')
MOUNTS+=(-v "$OVERLAY/hybrid:/opt/dsv41/hybrid:ro" -v "$OVERLAY/engram_ssd:/opt/dsv41/engram_ssd:ro")

CMD=(docker run -d --name "$NAME" --runtime=nvidia
  --ipc=host --shm-size 32g
  -p "$PORT:$PORT"
  -e NVIDIA_VISIBLE_DEVICES="$GPUS"
  -v "$MODEL:/hf" "${MOUNTS[@]}"
  -v /home/xpz/dsv41-run/dump:/dump
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1
  -e PYTHONPATH=/opt/dsv41
  -e PYTHONFAULTHANDLER=1
  -e VLLM_PP_LAYER_PARTITION="$PARTITION"
  -e VLLM_USE_V2_MODEL_RUNNER=0
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
  -e DSV41_MODEL_DIR="/hf/$SNAP_REL"
  -e DSV41_CPU_EXPERT_LAYERS="$CPU_LAYERS"
  -e DSV41_CPU_EXPERT_THREADS="$THREADS"
  -e DSV41_CPU_MOE=native
  -e DSV41_ENGRAM_STORAGE="$ENGRAM"
  -e DSV41_SKIP_WEIGHT_RE="$SKIP_RE"
  -e DSV41_MEM_CAP_FRACTION=0.965
  -e DSV41_FULL_Q1=1
  -e DSV41_PREFILL_EAGER=1
  -e DSV41_PP_SPREAD=1
  -e DSV41_DEFER_DRAFT=1
  -e DSV41_DRAFT_SKIP_PREFILL=1
  --restart no
  "$IMAGE"
    "/hf/$SNAP_REL"
    --served-model-name deepseek-v4.1-flash
    --port "$PORT"
    --pipeline-parallel-size 4
    --gpu-memory-utilization "$UTIL"
    --max-model-len "$MAXLEN"
    --max-num-seqs "$SEQS"
    --max-num-batched-tokens "$BATCHED"
    --attention-backend TRITON_MLA_SPARSE_DSV41
    --kv-cache-dtype fp8
    --attention-config '{"indexer_kv_dtype":"fp8"}'
    -cc.cudagraph_mode=FULL_AND_PIECEWISE
    --no-enable-prefix-caching
    -cc.cudagraph_capture_sizes=[1,2,3,4,6,8,12,16]
    --speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":$SPEC_N,\"prompt_lookup_max\":8,\"prompt_lookup_min\":5}"
    --tool-call-parser deepseek_v41
    --reasoning-parser deepseek_v41
    --enable-auto-tool-choice
)

if [ "$ACTION" = stop ]; then stop_container "$NAME"; exit 0; fi

if [ "$DRY_RUN" = 1 ]; then show_cmd "${CMD[@]}"; exit 0; fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

log "启动容器 $NAME（4 卡 PP4 $PARTITION，GPU3 留空，CPU 专家 $CPU_LAYERS × $THREADS 线程）"
"${CMD[@]}"
echo "docker logs -f $NAME  （就绪标志：Application startup complete.）"
