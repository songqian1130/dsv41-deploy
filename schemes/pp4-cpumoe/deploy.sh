#!/usr/bin/env bash
# ============================================================================
# 方案 L · PP4 CPU-MoE（zebgop 版）—— 一键部署
#
#   ./deploy.sh                  # 预检 → 起容器 → 等 /health
#   ./deploy.sh --dry-run        # 只打印 docker 命令
#   ./deploy.sh --stop
#
# 前置：4 张 GPU（GPU0-2 + GPU4；GPU3 留给 ComfyUI）· 约 110 GiB 主机内存
#       overlay 目录（git clone https://github.com/zebgop-ops/dsv41flash-pp.git）
# 与方案 A/B 不可同时运行（同一批卡）
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../scripts/lib.sh
. "$HERE/../../scripts/lib.sh"

usage() {
  cat <<EOF
方案 L · PP4 CPU-MoE（4 卡，单流 zh 26.4 tok/s 初测，GPU3 空出，不崩优先）
用法：$(basename "$0") [选项]
  --dry-run        只打印 docker 命令
  --stop           停止并删除容器
  --port N         端口（默认 9010，全方案统一）
  --model PATH     HF cache 目录（默认 /run/media/xpz/models/dsv41-hf）
  --gpus 0,1,2,4   卡号
环境变量：DSV41_MODEL / DSV41_IMAGE / DSV41_PORT / DSV41_GPUS / DSV41_NAME / DSV41_OVERLAY
EOF
}

parse_args "$@"

NAME=${DSV41_NAME:-dsv41-pp4-cpumoe}
PORT=${DSV41_PORT:-9010}
MODEL=${DSV41_MODEL:-/run/media/xpz/models/dsv41-hf}
DEFAULT_IMAGE=vllm/vllm-openai:deepseekv41-flash-0909
DEFAULT_ID=sha256:00d577a6a632
IMAGE=${DSV41_IMAGE:-$DEFAULT_IMAGE}
EXPECT_ID=$([ "$IMAGE" = "$DEFAULT_IMAGE" ] && echo "$DEFAULT_ID" || echo "")
OVERLAY=${DSV41_OVERLAY:-${HOME:-/root}/dsv41-run/repo/overlay}
GPUS=${DSV41_GPUS:-0,1,2,4}
PARTITION=${DSV41_PARTITION:-8,12,8,12}
CPU_LAYERS=${DSV41_CPU_EXPERT_LAYERS:-16-19,35-39}

if [ "$ACTION" = stop ]; then
  if [ "$DRY_RUN" = 1 ]; then log "将执行：docker rm -f $NAME"; exit 0; fi
  require_docker
  stop_container "$NAME"
  exit 0
fi

log "方案 L · PP4 CPU-MoE —— 4 卡 [$GPUS] $PARTITION + CPU 专家 $CPU_LAYERS"
if [ "$DRY_RUN" = 1 ]; then "$HERE/run.sh" "$@"; exit $?; fi

require_docker
require_gpus 4
require_model "$MODEL"
require_image "$IMAGE" "$EXPECT_ID"
# 主机内存：权重不进内存，但 CPU 专家 bank + 运行时页缓存仍需 ~110 GiB
require_mem_gib 120

[ -d "$OVERLAY/vllm" ] || die "overlay 缺失：$OVERLAY（git clone https://github.com/zebgop-ops/dsv41flash-pp.git ~/dsv41-run/repo）"

"$HERE/run.sh" "$@"
rc=$?
[ $rc -eq 0 ] || die "启动失败（run.sh 返回 $rc）"

# V1 runner + CPU 专家冷启动慢，给 20 分钟
wait_health "$PORT" "$NAME" 1200
print_summary "$PORT" "$NAME"
