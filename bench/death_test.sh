#!/bin/bash
# 固定死机用例：引擎持续满载 + ComfyUI H3 视频（原生音频）
# 用法: death_test.sh [引擎负载秒数] [并发数] [帧数]
# 每次都走同一条路径，不再临场拼命令。
# 默认 180s = 3 分钟（H3 视频本身约 128s，3 分钟足够覆盖死亡窗口）
set -u

DUR=${1:-180}
W=${2:-6}
FRAMES=${3:-124}
# 引擎端口可用环境变量覆盖：ENGINE_BASE=http://127.0.0.1:9010 death_test.sh
E=${ENGINE_BASE:-http://127.0.0.1:18000}
C=http://127.0.0.1:8188
D=${D:-/home/xpz/manju}          # 放家目录，重启不丢（/tmp 是 tmpfs）
L=$D/death_test.log
B=$D/engine_busy.log

mkdir -p "$D"
: > "$L"
: > "$B"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

say "=== 0 预检"
MODEL=$(curl -s -m 6 "$E/v1/models" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin)["data"][0]["id"])
except Exception:
    print("")')
if [ -z "$MODEL" ]; then say "FAIL 引擎不可用"; exit 2; fi
say "engine model = $MODEL"
if ! curl -s -m 6 -o /dev/null "$C/system_stats"; then say "FAIL ComfyUI 不可用"; exit 2; fi
say "comfy ok"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tee -a "$L"

say "=== 1 引擎满载 ${W}路 x ${DUR}s"
ENGINE=$E nohup python3 "$D/engine_busy.py" "$DUR" "$W" > "$B" 2>&1 &
EPID=$!

for _ in $(seq 1 60); do
  sleep 2
  grep -q "tok/s" "$B" 2>/dev/null && break
done
SAMPLES=$(grep -c "tok/s" "$B" 2>/dev/null || true)
say "engine busy samples=$SAMPLES"
if [ "${SAMPLES:-0}" -eq 0 ]; then
  say "FAIL 引擎没有产出 token，终止"
  tail -5 "$B" | tee -a "$L"
  kill "$EPID" 2>/dev/null
  exit 3
fi
tail -4 "$B" | tee -a "$L"
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | tee -a "$L"

say "=== 2 提交 H3 视频 ${FRAMES} 帧（此时引擎在算）"
TAG="dual_$(date +%H%M%S)"
COMFY_BASE=$C nohup python3 "$D/scripts/bench.py" video \
  --tag "$TAG" --width 1344 --height 768 --length "$FRAMES" --steps 8 \
  --runs 1 --deadline $((DUR + 180)) > "$D/bench_dual.log" 2>&1 &
BPID=$!

say "=== 3 计时观察（每 10s 一行）"
for i in $(seq 1 120); do
  sleep 10
  T=$((i * 10))
  MEM4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 4 2>/dev/null | tr -dc '0-9')
  UT=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  HH=$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$C/system_stats")
  QH=$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$E/v1/models")
  say "+${T}s gpu4=${MEM4}MiB util=[${UT}] comfy=${HH} engine=${QH}"
  if ! kill -0 "$BPID" 2>/dev/null; then say "bench 进程结束"; break; fi
done

wait "$BPID" 2>/dev/null
say "=== 4 bench 输出"
tail -8 "$D/bench_dual.log" | tee -a "$L"
kill "$EPID" 2>/dev/null
say "=== 结束：机器存活，用例未触发死机"
