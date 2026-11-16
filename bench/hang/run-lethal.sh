#!/bin/bash
# GPU 平台冻结复现用例（单卡片、无需大模型、无需 ComfyUI）
#
# 作用：在指定的一张卡上同时做三件事，复现"整机冻结"：
#   1) 主机侧持续读内存（模拟 CPU 侧专家计算，默认 10GB/s，16 线程）
#   2) 从文件读数据 -> pinned 内存 -> 拷进指定 GPU（模拟推理/出图的权重搬运）
#   3) 同一张卡上跑 matmul kernel（模拟 GPU 正在计算）
#   另外在显存里常驻 32GB（模拟模型已占用的显存）
#
# 用法：
#   bash run-lethal.sh [GPU号] [文件路径]
#   GPU号默认 0；文件默认自动生成 4GiB 的临时文件
#
# 环境变量：
#   DRAM_GBPS   主机读带宽，默认 10；设为 0 则只测"搬运+计算"
#   STAGE_GBPS  搬运带宽，默认 1.0
#   RESERVE_MB  显存常驻，默认 32768（卡小的话降到 16384 或 8192）
#   DURATION    持续秒数，默认 180
#   PYTHON      解释器，默认 python3（必须能 import torch 且 CUDA 可用）
#
# 预期结果（在我们这台机器上）：
#   30~120 秒内整机失联：SSH 断、ping 丢包或偶通、必须断电重启。日志停在最后一行。
#   如果是脚本报错退出、或跑满 DURATION 秒仍正常，则说明没复现。

set -u

GPU=${1:-0}
FILE=${2:-}
DRAM_GBPS=${DRAM_GBPS:-10}
STAGE_GBPS=${STAGE_GBPS:-1.0}
RESERVE_MB=${RESERVE_MB:-32768}
DURATION=${DURATION:-180}
PYTHON=${PYTHON:-python3}

HERE=$(cd "$(dirname "$0")" && pwd)
LOGDIR=$HERE/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%m%d-%H%M%S)
LOG=$LOGDIR/repro_gpu${GPU}_${STAMP}.log

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 环境检查"
if ! "$PYTHON" -c "import torch, sys; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'devices', torch.cuda.device_count())" 2>>"$LOG"; then
  say "FAIL: $PYTHON 里没有可用的 torch+cuda"
  exit 2
fi
command -v nvidia-smi >/dev/null || { say "FAIL: 没有 nvidia-smi"; exit 2; }

COUNT=$("$PYTHON" -c "import torch;print(torch.cuda.device_count())")
if [ "$GPU" -ge "$COUNT" ]; then
  say "FAIL: 指定的 GPU $GPU 超出范围（共 $COUNT 张）"
  exit 2
fi

if [ -z "$FILE" ]; then
  FILE=/tmp/gpu-hang-repro-stage.bin
  if [ ! -f "$FILE" ] || [ "$(stat -c %s "$FILE" 2>/dev/null || echo 0)" -lt 2147483648 ]; then
    say "=== 生成 4GiB 测试文件: $FILE"
    dd if=/dev/urandom of="$FILE" bs=1M count=4096 status=none || { say "FAIL: 生成文件失败"; exit 2; }
  fi
fi
[ -f "$FILE" ] || { say "FAIL: 文件不存在 $FILE"; exit 2; }
say "staging 文件: $FILE ($(stat -c %s "$FILE") 字节)"

say "=== 初始状态"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | tee -a "$LOG"

say "=== 开始：gpu=$GPU dram=${DRAM_GBPS}GB/s stage=${STAGE_GBPS}GB/s reserve=${RESERVE_MB}MiB dur=${DURATION}s"
say "    （如果整机在 30~180 秒内失联，就是复现了）"

# 背景监视：每 5 秒记录一次，便于事后看最后状态
(
  while true; do
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU" 2>/dev/null | tr -dc '0-9')
    UT=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -i "$GPU" 2>/dev/null | tr -dc '0-9')
    echo "[$(date +%H:%M:%S)] gpu${GPU}_mem=${MEM}MiB util=${UT}%" >> "$LOG"
    sleep 5
  done
) &
MON=$!

cleanup() { kill "$MON" 2>/dev/null; }
trap cleanup EXIT

"$PYTHON" "$HERE/synthetic_load.py" \
  --dram "$DRAM_GBPS" --threads 16 \
  --stage "$STAGE_GBPS" --stage-file "$FILE" --stage-gpu "$GPU" \
  --stage-reserve-mb "$RESERVE_MB" --stage-chunk-mb 64 \
  --sm --gpus "$GPU" \
  --duration "$DURATION" 2>&1 | tee -a "$LOG"

say "=== 跑满 ${DURATION}s 未冻结（未复现）"
say "日志: $LOG"
