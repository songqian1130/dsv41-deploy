#!/usr/bin/env bash
# ab_rank3.sh —— 卡位 A/B 对比：固定前三个 PP rank，只换 rank3 的卡
#
# 为什么这么设计（2026-11-16）：
#   PP4 的层切分是 [6,6,7,21]，最后一段吃 21 层，是全流水线最重的一跳。
#   原来拿 GPU"0,1,2,3" 和 "1,2,3,4" 对比，四张卡全换了，分不清是卡的问题还是
#   21 层压在哪张卡上的问题。本脚本固定 rank0/1/2 = 前三张，只换 rank3。
#
# 固定条件（不要改，改了就不是同一把尺子）：
#   - util 0.99 / KV 131072：用 ~/dsv41-repo-test/run-pp4.sh 的现值，本脚本不修改它
#   - 五卡功耗墙统一 150W（GPU0 平时是 250W 给 ComfyUI，这里临时压平）
#   - ComfyUI 必须停（引擎启动要求 MemAvailable ≥ 226 GiB）
#   - 每组先跑，再连跑 N 轮（默认 6，前 3 轮是热身，只取后 3 轮）
#
# 用法：
#   ./ab_rank3.sh                 # 默认：1,2,3,4 vs 1,2,3,0，各 6 轮
#   ./ab_rank3.sh 8               # 各 8 轮
#   ./ab_rank3.sh 6 1,2,3,4 1,2,3,0
#
# 判读：
#   A ≈ B          → GPU4 作为 rank3 没有问题，之前 A0 vs A1 的 6% 是热身
#   A 明显低于 B    → GPU4 承载 21 层有代价；再查 lspci AER 错误计数
#   A ≈ B 但都低    → 问题在"GPU0 不在流水线里"本身，另查
set -uo pipefail

ROUNDS=${1:-6}
GPUS_A=${2:-1,2,3,4}
GPUS_B=${3:-1,2,3,0}
PREFIX=${LABEL_PREFIX:-abrank3}

ENGINE="${HOME:-/home/xpz}/dsv41-repo-test"
URL="http://127.0.0.1:9010"
CONTAINER="dsv41-pp4"
STAMP=$(date +%m%d_%H%M)
LOGDIR="${HOME:-/home/xpz}/manju/ab_rank3_${STAMP}"
RUNLOG="$LOGDIR/run.log"

mkdir -p "$LOGDIR"
log()  { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }
tagg() { echo "g$(echo "$1" | tr ',' '_')"; }

# ---------- 前置检查 ----------
preflight() {
  log "=== preflight ==="
  if [ ! -x "$ENGINE/run-pp4.sh" ] && [ ! -f "$ENGINE/run-pp4.sh" ]; then
    log "FATAL: 找不到 $ENGINE/run-pp4.sh"; exit 1
  fi
  if [ ! -f "$ENGINE/bench/run_cases.py" ]; then
    log "FATAL: 找不到 $ENGINE/bench/run_cases.py"; exit 1
  fi
  # 必须确认 run-pp4.sh 的当前参数是基准配置，否则两组都在错的前提下跑
  local util kv
  util=$(grep -oE '\-\-gpu-memory-utilization [0-9.]+' "$ENGINE/run-pp4.sh" | awk '{print $2}')
  kv=$(grep -oE '\-\-max-model-len [0-9]+' "$ENGINE/run-pp4.sh" | awk '{print $2}')
  log "run-pp4.sh: util=$util  max-model-len=$kv"
  if [ "$util" != "0.99" ]; then
    log "⚠️  util 不是 0.99（原值）。0.92/0.97 会让槽位 4320→3320/3769，数字不可比。"
    log "    确认要这样对比就继续，否则先恢复：sed -i 's/--gpu-memory-utilization [0-9.]*/--gpu-memory-utilization 0.99/' $ENGINE/run-pp4.sh"
  fi
  if [ "$kv" != "131072" ]; then
    log "⚠️  KV 不是 131072（基准档），当前=$kv"
  fi

  # 统一功耗墙：GPU0 平时 250W（ComfyUI 用），这里压平到 150W 消除变量
  log "统一五卡功耗墙 → 150W"
  sudo nvidia-smi -i 0,1,2,3,4 -pl 150 2>&1 | tail -1 | tee -a "$RUNLOG"
  nvidia-smi --query-gpu=index,power.limit --format=csv,noheader | tee -a "$RUNLOG"

  log "停 ComfyUI（引擎启动要 MemAvailable ≥ 226 GiB）"
  sudo systemctl stop comfyui 2>/dev/null
  sleep 3
  free -g | head -2 | tee -a "$RUNLOG"
}

# ---------- 等引擎就绪 ----------
wait_ready() {
  local i
  for i in $(seq 1 45); do          # 45 × 20s = 15 分钟上限
    if curl -s -m 3 -o /dev/null -w "%{http_code}" "$URL/health" 2>/dev/null | grep -q 200; then
      log "READY after $((i*20))s"
      return 0
    fi
    if ! sudo docker ps --format "{{.Names}}" | grep -qx "$CONTAINER"; then
      log "DIED —— 容器已退出："
      sudo docker logs --tail 25 "$CONTAINER" 2>&1 | cut -c1-180 | tee -a "$RUNLOG"
      return 1
    fi
    sleep 20
  done
  log "TIMEOUT（>15 分钟）"
  return 1
}

# ---------- 记录这一组的实际形态 ----------
snapshot() {
  local tag=$1
  log "--- [$tag] 预算 ---"
  sudo docker logs "$CONTAINER" 2>&1 | grep "Hybrid cache budget" \
    | sed 's/.*free=/free=/' | cut -c1-130 | tee -a "$RUNLOG"
  log "--- [$tag] 槽位 ---"
  sudo docker logs "$CONTAINER" 2>&1 | grep "final GPU cache plan" \
    | grep -o "total=[0-9]* slots" | tee -a "$RUNLOG"
  log "--- [$tag] 实际显存分布（最高那张 = rank3，吃 21 层）---"
  nvidia-smi --query-gpu=index,power.limit,memory.used --format=csv,noheader | tee -a "$RUNLOG"
  log "--- [$tag] 容器 GPU 顺序 ---"
  sudo docker inspect "$CONTAINER" \
    --format '{{range .HostConfig.DeviceRequests}}{{.DeviceIDs}}{{end}}' 2>/dev/null | tee -a "$RUNLOG"
}

# ---------- 跑一组 ----------
run_group() {
  local gpus=$1
  local tag
  tag=$(tagg "$gpus")
  log ""
  log "================================================================"
  log "组 $tag   gpus=$gpus   轮数=$ROUNDS"
  log "================================================================"

  sudo docker stop -t 30 "$CONTAINER" >/dev/null 2>&1
  sleep 6
  ( cd "$ENGINE" && bash run-pp4.sh --gpus "$gpus" ) 2>&1 | tail -3 | tee -a "$RUNLOG"

  wait_ready || { log "组 $tag 启动失败，跳过"; return 1; }
  snapshot "$tag"

  local i nonce
  for i in $(seq 1 "$ROUNDS"); do
    nonce=$(( (RANDOM % 8000) + 1000 ))
    log "--- [$tag] round $i/$ROUNDS  nonce=$nonce ---"
    ( cd "$ENGINE" && python3 bench/run_cases.py "${PREFIX}_${tag}" "$nonce" --set real ) 2>&1 \
      | grep -E "^\[config\]|^(code|json|zh|math)" | tee -a "$RUNLOG"
  done
  log "组 $tag 完成"
}

# ---------- 汇总 ----------
summary() {
  log ""
  log "================ 汇总 ================"
  ( cd "$ENGINE" && python3 - "$PREFIX" <<'PY'
import json, sys, statistics
prefix = sys.argv[1]
rows = [json.loads(l) for l in open("bench/results.jsonl", encoding="utf-8") if l.strip()]
groups = {}
for r in rows:
    lab = r.get("label", "")
    if not lab.startswith(prefix + "_"): continue
    groups.setdefault(lab, []).append(r)

ORDER = ["code_python", "json_struct", "zh_tech", "math_steps"]
for lab in sorted(groups):
    rs = groups[lab]
    print("\n%s  (%d 轮)" % (lab, len(rs)))
    print("  %-4s %-10s %-10s %-10s" % ("轮", "wall均值", "decode", "ttft"))
    walls = []
    for i, r in enumerate(rs, 1):
        w = [c["tps_wall"] for c in r["cases"] if c.get("tps_wall")]
        d = [c["tps_decode"] for c in r["cases"] if c.get("tps_decode")]
        t = [c["ttft_s"] for c in r["cases"] if c.get("ttft_s")]
        if not w: continue
        walls.append(statistics.mean(w))
        print("  %-4d %-10.2f %-10.2f %-10.3f" % (i, statistics.mean(w),
              statistics.mean(d) if d else 0, statistics.mean(t) if t else 0))
    if walls:
        warm = walls[3:] if len(walls) > 4 else walls
        print("  → 全部均值 %.2f | 排除前 3 轮 %.2f (n=%d)"
              % (statistics.mean(walls), statistics.mean(warm), len(warm)))
PY
  ) 2>&1 | tee -a "$RUNLOG"
  log "日志目录：$LOGDIR"
}

# ---------- main ----------
log "ab_rank3 开始  轮数=$ROUNDS  A=$GPUS_A  B=$GPUS_B  prefix=$PREFIX"
preflight
run_group "$GPUS_A" || log "⚠️ A 组失败，继续 B 组"
run_group "$GPUS_B" || log "⚠️ B 组失败"
summary

log ""
log "注意：功耗墙已被压平到 150W。要让 ComfyUI 的卡恢复 250W 就"
log "      sudo systemctl restart nvidia-170hx-power-limit.service"
log "      然后 sudo systemctl start comfyui"
