#!/usr/bin/env bash
# ============================================================================
# 方案 A · PP5 fast —— 一键部署
#
#   ./deploy.sh                  # 预检 → 起容器 → 等 /health → 打印端点
#   ./deploy.sh --dry-run        # 只打印 docker 命令
#   ./deploy.sh --stop           # 停掉
#   ./deploy.sh --model /data/xxx --port 9011
#
# 前置：5 张 GPU（每张 ≥64 GB）· 约 190 GiB 主机内存 · Docker + nvidia-container-toolkit
# 首次启动约 9 分钟（权重 4 分钟 + engine init 5 分钟，其中每层专家缓存校准各 5 秒）
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../scripts/lib.sh
. "$HERE/../../scripts/lib.sh"

usage() {
  cat <<EOF
方案 A · PP5 fast（5 卡，单流 82.5 tok/s）
用法：$(basename "$0") [选项]
  --dry-run        只打印 docker 命令
  --stop           停止并删除容器
  --port N         端口（默认 9010）
  --model PATH     模型目录（默认 \$DSV41_MODEL 或脚本内置路径）
  --gpus 0,1,2,3,4 卡号
环境变量：DSV41_MODEL / DSV41_IMAGE / DSV41_PORT / DSV41_GPUS / DSV41_NAME / DSV41_CACHE
EOF
}

parse_args "$@"

NAME=${DSV41_NAME:-dsv41-pp5-fast}
PORT=${DSV41_PORT:-9010}
MODEL=${DSV41_MODEL:-/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official}
DEFAULT_IMAGE=18gogogo/170hx4-256r-dsv41-1017-flash:sm80
DEFAULT_ID=sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157
IMAGE=${DSV41_IMAGE:-$DEFAULT_IMAGE}
# 只有用默认镜像时才校验 ID；用户主动换镜像就不该拦
EXPECT_ID=$([ "$IMAGE" = "$DEFAULT_IMAGE" ] && echo "$DEFAULT_ID" || echo "")

if [ "$ACTION" = stop ]; then
  if [ "$DRY_RUN" = 1 ]; then log "将执行：docker rm -f $NAME"; exit 0; fi
  require_docker
  stop_container "$NAME"
  exit 0
fi

log "方案 A · PP5 fast —— 5 卡 PP5 + dspark k=5 + cpu_threads [28,4]"
if [ "$DRY_RUN" = 1 ]; then "$HERE/run.sh" "$@"; exit $?; fi

require_docker
require_gpus 5
require_model "$MODEL"
require_image "$IMAGE" "$EXPECT_ID"
require_mem_gib 200

"$HERE/run.sh" "$@"
rc=$?
[ $rc -eq 0 ] || die "启动失败（run.sh 返回 $rc）"

wait_health "$PORT" "$NAME" 1800
print_summary "$PORT" "$NAME"
