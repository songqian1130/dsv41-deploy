#!/usr/bin/env bash
# ============================================================================
# 方案 C · 旧 PP5（V1 引擎）—— 一键部署
#
#   ./deploy.sh                       # engram SSD（默认）
#   ./deploy.sh --engram ram          # engram 全放 pinned 内存（TTFT 快 46%，吃 189 GiB）
#   ./deploy.sh --dry-run
#   ./deploy.sh --stop
#
# 前置：5 张 GPU（每张 ≥64 GB）· SSD 模式 40 GiB 内存 / RAM 模式 190 GiB
# ⚠ 镜像是原机本地构建的（local/ 前缀），换机器要先 docker save / load，见 README
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../scripts/lib.sh
. "$HERE/../../scripts/lib.sh"

usage() {
  cat <<EOF
方案 C · 旧 PP5（V1 引擎，单流 43.5 tok/s，启动 4:30）
用法：$(basename "$0") [选项]
  --dry-run        只打印 docker 命令
  --stop           停止并删除容器
  --engram ssd|ram engram 存放位置（默认 ssd；ram 需 189 GiB 内存，TTFT 6.45 s）
  --port N         端口（默认 9010）
  --model PATH     模型目录
  --gpus 0,1,2,3,4 卡号
环境变量：DSV41_MODEL / DSV41_IMAGE / DSV41_PORT / DSV41_GPUS / DSV41_NAME / DSV41_CACHE / DSV41_ENGRAM
EOF
}

# --engram 是本方案独有的，先挑出来，其余交给通用解析
ENGRAM=${DSV41_ENGRAM:-ssd}
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --engram) ENGRAM=${2:?--engram 需要 ssd 或 ram}; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
parse_args ${ARGS[@]+"${ARGS[@]}"}
case "$ENGRAM" in ssd|ram) ;; *) die "--engram 只能是 ssd 或 ram（收到 $ENGRAM）" ;; esac

NAME=${DSV41_NAME:-dsv41-pp5-legacy}
PORT=${DSV41_PORT:-9010}
MODEL=${DSV41_MODEL:-/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official}
DEFAULT_IMAGE=local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102
DEFAULT_ID=sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1
IMAGE=${DSV41_IMAGE:-$DEFAULT_IMAGE}
# 只有用默认镜像时才校验 ID；用户主动换镜像就不该拦
EXPECT_ID=$([ "$IMAGE" = "$DEFAULT_IMAGE" ] && echo "$DEFAULT_ID" || echo "")

if [ "$ACTION" = stop ]; then
  if [ "$DRY_RUN" = 1 ]; then log "将执行：docker rm -f $NAME"; exit 0; fi
  require_docker
  stop_container "$NAME"
  exit 0
fi

log "方案 C · 旧 PP5（V1 引擎，无 dspark）· engram=$ENGRAM"
if [ "$DRY_RUN" = 1 ]; then "$HERE/run.sh" ${ARGS[@]+"${ARGS[@]}"} --engram "$ENGRAM"; exit $?; fi

require_docker
require_gpus 5
require_model "$MODEL"
require_image "$IMAGE" "$EXPECT_ID"
if [ "$ENGRAM" = ram ]; then require_mem_gib 190; else require_mem_gib 40; fi

"$HERE/run.sh" ${ARGS[@]+"${ARGS[@]}"} --engram "$ENGRAM"
rc=$?
[ $rc -eq 0 ] || die "启动失败（run.sh 返回 $rc）"

if [ "$ENGRAM" = ram ]; then wait_health "$PORT" "$NAME" 1500; else wait_health "$PORT" "$NAME" 900; fi
print_summary "$PORT" "$NAME"
