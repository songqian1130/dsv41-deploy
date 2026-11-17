#!/bin/bash
# 合成负载 + ComfyUI H3 视频：不需要加载大模型，3 分钟一轮。
# 用法:
#   synthetic_test.sh "<合成负载参数>" [帧数]
# 例:
#   synthetic_test.sh "--dram 44 --threads 16"            # 只打 DRAM 带宽
#   synthetic_test.sh "--nvme 2"                          # 只打 NVMe 读
#   synthetic_test.sh "--h2d 1.5"                         # 只打 host->device
#   synthetic_test.sh "--dram 44 --nvme 2 --h2d 1.5 --threads 16"
# 环境变量:
#   DUR=180        合成负载持续秒数（默认 3 分钟）
#   SKIP_COMFY=1   只打负载，不提交 H3（做对照）
set -u

SYN_ARGS=${1:---dram 44 --threads 16}
FRAMES=${2:-124}
DUR=${DUR:-180}
PY=${PY:-/home/xpz/venvs/comfy/bin/python}   # 系统 python3 没有 torch
C=http://127.0.0.1:8188
D=${D:-/home/xpz/manju}          # 放家目录，重启不丢（/tmp 是 tmpfs）
L=$D/syn_test.log
SL=$D/syn_load.log

mkdir -p "$D"
: > "$L"
: > "$SL"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

say "=== 0 预检"
if [ "${SKIP_COMFY:-0}" = "1" ]; then
  say "SKIP_COMFY=1（只打合成负载）"
else
  if ! curl -s -m 6 -o /dev/null "$C/system_stats"; then say "FAIL ComfyUI 不可用"; exit 2; fi
  say "comfy ok"
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tee -a "$L"

say "=== 1 合成负载: $SYN_ARGS  持续 ${DUR}s"
nohup "$PY" "$D/synthetic_load.py" $SYN_ARGS --duration "$DUR" > "$SL" 2>&1 &
SPID=$!

for _ in $(seq 1 40); do
  sleep 1
  grep -q "GB/s" "$SL" 2>/dev/null && break
done
SAMPLES=$(grep -c "GB/s" "$SL" 2>/dev/null || true)
say "load samples=$SAMPLES"
if [ "${SAMPLES:-0}" -eq 0 ]; then
  say "FAIL 合成负载没跑起来"
  tail -8 "$SL" | tee -a "$L"
  kill "$SPID" 2>/dev/null
  exit 3
fi
tail -3 "$SL" | tee -a "$L"

if [ "${SKIP_COMFY:-0}" = "1" ]; then
  say "=== 2 跳过 ComfyUI，只观察负载本身"
  BPID=""
else
  say "=== 2 提交 H3 视频 ${FRAMES} 帧（合成负载运行中）"
  TAG="syn_$(date +%H%M%S)"
  COMFY_BASE=$C nohup python3 "$D/scripts/bench.py" video \
    --tag "$TAG" --width 1344 --height 768 --length "$FRAMES" --steps 8 \
    --runs 1 --deadline $((DUR + 180)) > "$D/bench_syn.log" 2>&1 &
  BPID=$!
fi

say "=== 3 计时观察（每 10s 一行）"
for i in $(seq 1 120); do
  sleep 10
  T=$((i * 10))
  MEM4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 4 2>/dev/null | tr -dc '0-9')
  UT=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ' ')
  HH=$(curl -s -m 4 -o /dev/null -w '%{http_code}' "$C/system_stats")
  LOADLINE=$(tail -1 "$SL" 2>/dev/null | tr -d '\n')
  say "+${T}s gpu4=${MEM4}MiB util=[${UT}] comfy=${HH} | ${LOADLINE}"
  if [ -n "${BPID:-}" ] && ! kill -0 "$BPID" 2>/dev/null; then say "bench 进程结束"; break; fi
  if ! kill -0 "$SPID" 2>/dev/null; then say "合成负载结束"; break; fi
done

if [ -n "${BPID:-}" ]; then
  wait "$BPID" 2>/dev/null
  say "=== 4 bench 输出"
  tail -6 "$D/bench_syn.log" | tee -a "$L"
fi
kill "$SPID" 2>/dev/null
say "=== 结束：机器存活"
