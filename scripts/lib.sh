#!/usr/bin/env bash
# 共用部署库。被 schemes/*/deploy.sh source，不单独执行。
#
# 设计原则：
#   1) 只依赖 docker + nvidia-smi + curl，不引入 python/ yq 之类额外依赖
#   2) 所有路径都能用环境变量覆盖，脚本里不出现"只能在这台机器跑"的硬编码
#   3) --dry-run 只打印将要执行的 docker 命令，不碰系统
set -uo pipefail

if [ -t 1 ]; then
  RST=$'\033[0m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'
else
  RST=''; DIM=''; RED=''; GRN=''; YEL=''
fi

log()  { printf '%s[deploy]%s %s\n' "$DIM" "$RST" "$*"; }
ok()   { printf '%s[  ok  ]%s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '%s[ warn ]%s %s\n' "$YEL" "$RST" "$*"; }
die()  { printf '%s[ fail ]%s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

DRY_RUN=${DRY_RUN:-0}
ACTION=${ACTION:-up}

# 各方案可以覆盖这个函数，输出自己的说明
usage() {
  cat <<EOF
用法：$(basename "$0") [选项]
  --dry-run        只打印将要执行的 docker 命令，不启动
  --stop           停止并删除容器
  --port N         覆盖端口
  --model PATH     覆盖模型目录
  --gpus 0,1,2     覆盖卡号
  --name NAME      覆盖容器名
环境变量：DSV41_MODEL / DSV41_IMAGE / DSV41_PORT / DSV41_GPUS / DSV41_NAME / DSV41_CACHE
EOF
}

# 解析通用参数：--dry-run / --stop / --port N / --model PATH / --gpus 0,1,2,3 / --name NAME
parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY_RUN=1 ;;
      --stop)    ACTION=stop ;;
      --port)    DSV41_PORT=${2:?--port 需要参数}; shift ;;
      --model)   DSV41_MODEL=${2:?--model 需要参数}; shift ;;
      --gpus)    DSV41_GPUS=${2:?--gpus 需要参数}; shift ;;
      --name)    DSV41_NAME=${2:?--name 需要参数}; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "未知参数：$1（用 --help 看用法）" ;;
    esac
    shift
  done
}

# 打印将要执行的命令（带换行缩进，方便复制）
show_cmd() {
  local -a cmd=("$@")
  printf '\n%s$%s ' "$DIM" "$RST"
  local first=1
  for a in "${cmd[@]}"; do
    if [ $first = 1 ]; then printf '%q' "$a"; first=0
    elif [[ "$a" == -* && "$a" != "-d" ]]; then printf ' \\\n    %q' "$a"
    else printf ' %q' "$a"; fi
  done
  printf '\n\n'
}

require_docker() {
  command -v docker >/dev/null 2>&1 || die "找不到 docker，请先安装 Docker Engine"
  docker info >/dev/null 2>&1 || die "docker 守护进程不可用（没启动，或当前用户不在 docker 组）"
}

require_gpus() {
  local need="$1"
  command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi，请先装 NVIDIA 驱动"
  local have
  have=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
  [ "$have" -ge "$need" ] || die "需要 $need 张 GPU，本机只有 $have 张"
  docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"' \
    || die "docker 未注册 nvidia runtime，请装 nvidia-container-toolkit 后重启 docker"
  ok "$have 张 GPU，nvidia runtime 可用"
}

require_model() {
  local m="$1"
  [ -d "$m" ] || die "模型目录不存在：$m（用 --model 或 DSV41_MODEL 指定）"
  [ -f "$m/config.json" ] || die "模型目录不像 HF 权重（缺 config.json）：$m"
  local n
  n=$(find "$m" -maxdepth 1 -name '*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -gt 0 ] || die "模型目录里没有 *.safetensors：$m"
  ok "模型：$m（$n 个 safetensors）"
}

require_image() {
  local img="$1" expect="${2:-}"
  if docker image inspect "$img" >/dev/null 2>&1; then
    ok "镜像已在本机：$img"
    check_image_id "$img" "$expect"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    warn "--dry-run：镜像不在本机，正式执行时会尝试 docker pull"
    return 0
  fi
  case "$img" in
    local/*)
      die "这是本地构建的镜像，不在任何 registry：$img
     三条路，按有没有外网选：
       1) 重建（主路径，需外网，40~90 分钟）
            ./engine/v1-legacy/build.sh          # 或 engine/v2-dspark/build.sh
       2) 离线整包（仅无外网时，带 sha256 与镜像 ID 校验）
            源机器  ./tools/engine-bundle.sh save <引擎> --out /big/disk
            目标机  ./tools/engine-bundle.sh load /big/disk/<引擎>-*.tar.zst
       3) 已有别的镜像：用 DSV41_IMAGE=<镜像名> 覆盖" ;;
  esac
  log "本机没有 $img，尝试拉取……"
  docker pull "$img" || die "拉取失败：$img
     要么网络/仓库权限问题，要么这个 tag 已经不在。
     三条后路：
       1) 从源码重建（主路径）：engine/ 下对应引擎的 build.sh
       2) 离线整包：./tools/engine-bundle.sh load <包.tar.zst>
       3) 换镜像：DSV41_IMAGE=<你自己的镜像>
     来源与补丁：engine/README.md"
  ok "已拉取：$img"
  check_image_id "$img" "$expect"
}

# 镜像 ID 是我们唯一能证明「跑的是测过那一版」的东西。
# 不一致不阻断（允许用户用 DSV41_IMAGE 主动换引擎），但必须喊出来。
check_image_id() {
  local img="$1" expect="$2"
  [ -n "$expect" ] || return 0
  local actual
  actual=$(docker image inspect "$img" --format '{{.Id}}' 2>/dev/null || echo "")
  if [ "$actual" = "$expect" ]; then
    ok "镜像 ID 与记录一致（${expect:0:19}…）"
  else
    warn "镜像 ID 与记录不一致！"
    warn "  记录 $expect"
    warn "  实际 $actual"
    warn "  —— 这不是测出基准数字的那个镜像，速度可能完全不同。"
    warn "     来源与校验见 engine/README.md"
  fi
}

require_mem_gib() {
  local need="$1"
  local avail
  avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)
  if [ "$avail" -lt "$need" ]; then
    warn "可用内存 ${avail} GiB < 建议 ${need} GiB，可能 OOM 或无法分配 pinned 内存"
  else
    ok "可用内存 ${avail} GiB（建议 ≥${need}）"
  fi
}

tail_logs() {
  local name="$1"
  printf '\n%s---- docker logs --tail 40 %s ----%s\n' "$DIM" "$name" "$RST" >&2
  docker logs --tail 40 "$name" 2>&1 | sed 's/^/    /' >&2 || true
  printf '%s--------------------------------%s\n\n' "$DIM" "$RST" >&2
}

wait_health() {
  local port="$1" name="$2" timeout="${3:-1800}"
  if [ "$DRY_RUN" = 1 ]; then
    log "--dry-run：跳过健康等待（正式执行会等到 /health 200）"
    return 0
  fi
  local start=$SECONDS
  log "等待 http://127.0.0.1:${port}/health，最长 ${timeout}s（首次启动通常 5~12 分钟）"
  while :; do
    local elapsed=$(( SECONDS - start ))
    if curl -fsS -m 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      ok "服务就绪，用时 ${elapsed}s"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$name"; then
      tail_logs "$name"
      die "容器 $name 已退出（不是还在加载）"
    fi
    if [ "$elapsed" -ge "$timeout" ]; then
      tail_logs "$name"
      die "超过 ${timeout}s 仍未就绪"
    fi
    if [ $(( elapsed % 30 )) -lt 3 ]; then
      log "  ... ${elapsed}s，容器仍在加载"
    fi
    sleep 3
  done
}

print_summary() {
  local port="$1" name="$2"
  cat <<EOF

$(ok "部署完成")
  容器    $name
  端点    http://127.0.0.1:${port}/v1
  模型名  deepseek-v4.1-flash-1017
  日志    docker logs -f $name
  停止    docker rm -f $name

  试一下：
    curl -s http://127.0.0.1:${port}/v1/chat/completions \\
      -H 'Content-Type: application/json' \\
      -d '{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"用一句话说明 MoE 是什么"}],"max_tokens":128}'

EOF
}

stop_container() {
  local name="$1"
  if [ "$DRY_RUN" = 1 ]; then
    log "将执行：docker rm -f $name"
    return 0
  fi
  docker rm -f "$name" >/dev/null 2>&1 && ok "已停止并删除容器 $name" || log "容器 $name 本来就不存在"
}
