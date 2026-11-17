#!/bin/bash
# 逐卡扫描：对指定的每张卡跑同一套致命配置，记录是否冻结。
# 用法: sweep_cards.sh [卡列表，逗号分隔] [每张持续秒数]
#   例: sweep_cards.sh 0,1,2,3,4 150
# 说明: 冻结会中断整轮（机器要断电），日志里最后一行就是出问题的那张卡。
set -u

LIST=${1:-1,3}
DUR=${2:-150}
D=${D:-/home/xpz/manju}
PY=${PY:-/home/xpz/venvs/comfy/bin/python}
FILE=${FILE:-/run/media/xpz/models/AIGC/ComfyUI260520/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors}
LOG=$D/sweep.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 扫描开始 cards=$LIST dur=${DUR}s file=$(basename "$FILE")"
if ! sudo systemctl stop comfyui 2>/dev/null; then :; fi
sleep 2

for g in ${LIST//,/ }; do
  say "--- GPU$g 开始"
  "$PY" "$D/synthetic_load.py" \
    --dram 10 --threads 16 \
    --stage 1.0 --stage-file "$FILE" --stage-gpu "$g" --stage-reserve-mb 32768 \
    --sm --gpus "$g" --duration "$DUR" > "$D/load_gpu$g.log" 2>&1
  RC=$?
  LAST=$(grep -E "^\[[0-9:]+\]" "$D/load_gpu$g.log" | tail -1)
  if [ "$RC" -eq 0 ]; then
    say "--- GPU$g 存活（rc=0） | $LAST"
  else
    say "--- GPU$g 异常退出（rc=$RC） | $LAST"
  fi
done

say "=== 扫描结束：以上各卡均存活"
