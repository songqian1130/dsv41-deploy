#!/usr/bin/env bash
# dsv41-pp5-ctl.sh — DeepSeek V4.1 Flash PP5 模式化启停 / 开机自启 / 健康自愈
#
# ⚠ 这是**原机专属**脚本，不是通用部署入口。它依赖：
#     /etc/dsv41-pp5/mode 模式文件、systemd 单元 dsv41-pp5.service + dsv41-pp5-health.timer、
#     以及服务器上 ~/dsv41-repo-test/run-pp5-*.sh 这套脚本名。
#   换机器部署请用仓库根目录的 ./deploy.sh <方案>。
#   留在这里是因为它管着生产机的自愈，删了会在重启后没人拉起服务。
#
#   up            : 按当前模式确保服务在跑（已健康则跳过）
#   heal          : 看门狗；挂了就重建，加载期内不打扰
#   check         : 只探活
#   mode          : 打印当前模式
#   use <mode>    : 切换模式（停掉其它模式容器 -> 写模式文件 -> 启动）
#   status        : 打印模式、容器、健康、内存、显存
#
# 模式定义见下方 MODES 表；模式文件 /etc/dsv41-pp5/mode 记录当前选择，
# 因此 systemd 看门狗自愈时拉起的也是同一模式。
set -uo pipefail

MODE_DIR=/etc/dsv41-pp5
MODE_FILE=$MODE_DIR/mode
MODEL_DIR=/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official
REPO=/home/xpz/dsv41-repo-test
HEALTH_URL=${HEALTH_URL:-http://127.0.0.1:9010/health}
LOG=${CTL_LOG:-/var/log/dsv41-pp5-ctl.log}
DRY_RUN=${DRY_RUN:-0}

# 非 root 调用时（人工诊断）退化为只打 stderr，避免 "Permission denied" 噪声
if [ ! -w "$(dirname "$LOG")" ]; then LOG=/dev/null; fi

log() { echo "$(date '+%F %T') [$1] $2" | tee -a "$LOG" >&2; }

# ---- 模式表：name|container|runner|预热秒|启动前最少可用内存GB|说明 ----
MODES=$(cat <<'EOF'
legacy|dsv41-pp5-legacy|run-pp5-legacy.sh|1500|190|旧引擎 V1+惰性接收，engram 走 SSD，专家100%在卡
ram|dsv41-pp5-ram|run-pp5-ram.sh|2100|190|旧引擎 V1+惰性接收，engram 全放 pinned 内存（旧生产，已被 fast 取代）
dspark|dsv41-pp5-dspark|run-pp5-dspark.sh|2100|190|新引擎 V2+dspark k=3+20层专家在CPU，engram 走 SSD（实测比 ram 慢，留作对照）
fast|dsv41-pp5-dspthr|run-pp5-dspthr.sh|2100|100|新引擎 V2+dspark k=3+decode CPU线程4+KV131072（实测最快，生产首选）
pp4|dsv41-pp4|run-pp4.sh|2100|100|4卡PP4（GPU4 留给 ComfyUI），专家驻留56.25%，decode CPU线程16
EOF
)

mode_conf() # $1=mode -> 设置 NAME RUNNER WARMUP_SEC MIN_AVAIL_GB DESC
{
  local line
  line=$(echo "$MODES" | awk -F'|' -v m="$1" '$1==m')
  if [ -z "$line" ]; then
    log error "未知模式：$1（可用：$(echo "$MODES" | cut -d'|' -f1 | tr '\n' ' '))"
    return 1
  fi
  IFS='|' read -r _ NAME _r WARMUP_SEC MIN_AVAIL_GB DESC <<<"$line"
  RUNNER=$REPO/$_r
  return 0
}

cur_mode() { cat "$MODE_FILE" 2>/dev/null | tr -d '[:space:]' || true; }

health() { curl -sf -m 4 -o /dev/null "$HEALTH_URL"; }
state()  { docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing; }
free_gb(){ awk '/^MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo; }

started_ago() {
  local ts
  ts=$(docker inspect -f '{{.State.StartedAt}}' "$NAME" 2>/dev/null) || { echo 999999; return; }
  python3 - "$ts" <<'PY' 2>/dev/null || echo 999999
import sys, datetime
s = sys.argv[1].split('.')[0].replace('T', ' ')
t = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=datetime.timezone.utc)
print(int((datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()))
PY
}

drop_others() {
  # 停掉所有其它模式的容器：5 张卡 + 190GB 内存只能给一个引擎
  local c
  for c in $(echo "$MODES" | cut -d'|' -f2); do
    [ "$c" = "$NAME" ] && continue
    if [ "$(docker inspect -f '{{.State.Status}}' "$c" 2>/dev/null)" = "running" ]; then
      log warn "停掉其它模式容器 $c（会抢占 5 卡与主机内存）"
      [ "$DRY_RUN" = "1" ] && { log info "[dry-run] 将停 $c"; continue; }
      docker stop "$c" >/dev/null 2>&1 || docker kill "$c" >/dev/null 2>&1 || true
    fi
  done
  sleep 4
}

remove_own() {
  local st; st=$(state)
  [ "$st" = "missing" ] && return 0
  log info "清理旧容器 $NAME（当前状态=$st）"
  docker rm -f "$NAME" >/dev/null 2>&1 && return 0
  sleep 5
  docker rm -f "$NAME" >/dev/null 2>&1 && return 0
  log error "无法删除容器 $NAME，放弃启动"
  return 1
}

preflight() {
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    log error "模型目录未挂载（缺 $MODEL_DIR/config.json），放弃启动。请检查 /run/media/xpz/models 是否挂上"
    return 1
  fi
  local i used gpu_scope
  # pp4 是唯一 4 卡模式：GPU4 归 ComfyUI 常驻，不参与显存预检（2026-11-15 修）
  if [ "$CUR_MODE" = "pp4" ]; then gpu_scope="--id=0,1,2,3"; else gpu_scope="--id=0,1,2,3,4"; fi
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader $gpu_scope 2>/dev/null | awk '{print $1}' | sort -nr | head -1)
    if [ -z "$used" ]; then log warn "nvidia-smi 无输出，等 10s（$i/30）"; sleep 10; continue; fi
    if [ "$used" -lt 1024 ]; then break; fi
    log warn "GPU 仍被占用（$CUR_MODE 范围内最大 ${used}MiB），等 10s（$i/30）"
    sleep 10
  done
  if [ "${used:-0}" -ge 1024 ]; then
    log error "显存未释放（$CUR_MODE 范围内最大 ${used}MiB），放弃启动"
    return 1
  fi
  # ram/dspark 模式退出后 pinned 页需要时间归还，必须等到够用再起
  local avail
  for i in $(seq 1 24); do
    avail=$(free_gb)
    if [ "$avail" -ge "$MIN_AVAIL_GB" ]; then break; fi
    log warn "可用内存 ${avail}GB < ${MIN_AVAIL_GB}GB（上一步的 pinned 页未归还？），等 5s（$i/24）"
    sleep 5
  done
  if [ "$avail" -lt "$MIN_AVAIL_GB" ]; then
    log error "可用内存不足：${avail}GB < ${MIN_AVAIL_GB}GB"
    return 1
  fi
  log info "预检通过：显存已释放，可用内存 ${avail}GB，模式=$CUR_MODE($DESC)"
  return 0
}

launch() {
  if [ "$DRY_RUN" = "1" ]; then
    log info "[dry-run] 将清理 $NAME、预检后执行 $RUNNER"
    return 0
  fi
  bash /usr/local/bin/dsv41-salvage.sh || true
  drop_others
  remove_own || return 1
  preflight || return 1
  log info "启动容器 $NAME（加载 5-12 分钟，期间 /health 不通属正常）"
  bash "$RUNNER" >>"$LOG" 2>&1
  log info "容器已创建，状态=$(state)"
}

do_status() {
  local m; m=$(cur_mode)
  echo "模式文件: ${m:-未设置}"
  mode_conf "${m:-legacy}" || return 1
  printf '当前模式: %-8s 容器: %-22s 说明: %s\n' "$m" "$NAME" "$DESC"
  echo "容器状态: $(state)   健康: $(health && echo OK || echo FAIL)"
  echo "可用内存: $(free_gb) GB"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null
  echo "--- 三种模式容器 ---"
  docker ps -a --format '{{.Names}}\t{{.Status}}' | grep -E 'dsv41-pp5' || true
}

case "${1:-up}" in
  mode) cur_mode; echo ;;
  status) do_status; exit $? ;;
  use)
    new=${2:-}
    [ -z "$new" ] && { echo "usage: $0 use {legacy|ram|dspark}" >&2; exit 2; }
    mode_conf "$new" || exit 2
    CUR_MODE=$new
    mkdir -p "$MODE_DIR"
    echo "$new" > "$MODE_FILE"
    log info "模式切换 -> $new（容器 $NAME）"
    launch
    ;;
  up|heal)
    CUR_MODE=$(cur_mode)
    if [ -z "$CUR_MODE" ]; then
      CUR_MODE=legacy
      mkdir -p "$MODE_DIR"; echo "$CUR_MODE" > "$MODE_FILE"
      log warn "模式文件缺失，回落到 $CUR_MODE"
    fi
    mode_conf "$CUR_MODE" || exit 1
    if [ "$1" = "up" ]; then
      if [ "$(state)" = "running" ] && health; then log info "已在运行且健康（$CUR_MODE），跳过"; exit 0; fi
      launch
    else
      st=$(state)
      if [ "$st" = "running" ]; then
        if health; then exit 0; fi
        age=$(started_ago)
        if [ "$age" -lt "$WARMUP_SEC" ]; then log info "启动加载中（${age}s < ${WARMUP_SEC}s 预热期），暂不处理"; exit 0; fi
        log warn "容器已运行 ${age}s 但 /health 不通，判定卡死，重建"
      else
        log warn "容器状态=${st}，需要重建"
      fi
      launch
    fi
    ;;
  check)
    CUR_MODE=$(cur_mode); mode_conf "${CUR_MODE:-legacy}" || exit 1
    if [ "$(state)" = "running" ] && health; then exit 0; fi
    exit 1
    ;;
  *) echo "usage: $0 {up|heal|check|mode|status|use <mode>}" >&2; exit 2 ;;
esac
