#!/usr/bin/env bash
# ============================================================================
# 一键部署入口：在目标机器上拉下本仓库后，只跑这一条
#
#   ./deploy.sh pp5-fast          # 方案 A：5 卡，82.5 tok/s（生产）
#   ./deploy.sh pp4-t16           # 方案 B：4 卡，60.9 tok/s，GPU4 留给 ComfyUI
#   ./deploy.sh pp5-legacy        # 方案 C：V1 旧引擎，43.5 tok/s，启动最快
#   ./deploy.sh pp4-cpumoe        # 方案 L：V1 + 9 层专家跑 CPU，GPU3 留给 ComfyUI（不崩优先）
#
# 通用选项（透传给具体方案）：
#   --dry-run        只打印 docker 命令，不启动
#   --stop           停止并删除容器
#   --model PATH     模型目录（默认 /run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official）
#   --port N         服务端口
#   --gpus 0,1,2,3   指定卡号
#
# 没有参数时只列方案，不动系统。
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

list_schemes() {
  cat <<'EOF'
可部署的方案：
  pp5-fast      方案 A · 5 卡 · 单流 82.5 tok/s · 启动 8:57 · 生产基线
  pp4-t16       方案 B · 4 卡 · 单流 60.9 tok/s · GPU4 空出给 ComfyUI
  pp5-legacy    方案 C · 5 卡 · 单流 43.5 tok/s · 启动 4:30 · V1 旧引擎（可 --engram ram）
  pp4-cpumoe    方案 L · 4 卡 · 9 层专家跑 CPU · GPU3 空出 · GPU4 在场也不冻机的兜底

用法：./deploy.sh <方案名> [--dry-run|--stop] [--model PATH] [--port N] [--gpus LIST]
对比页：index.html      参数与实测明细：SCHEMES.md
EOF
}

if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "--list" ]; then
  list_schemes
  [ $# -eq 0 ] || exit 0
  exit 2
fi

SCHEME=$1; shift
DIR="$HERE/schemes/$SCHEME"

if [ ! -d "$DIR" ]; then
  printf '没有这个方案：%s\n\n' "$SCHEME" >&2
  list_schemes >&2
  exit 2
fi
if [ ! -x "$DIR/deploy.sh" ]; then
  chmod +x "$DIR/deploy.sh" "$DIR"/*.sh 2>/dev/null || true
fi

exec "$DIR/deploy.sh" "$@"
