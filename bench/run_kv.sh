#!/usr/bin/env bash
# run_kv.sh —— KV 档位单组标准测试（2026-11-16 固化）
#
# 用途：测某个 KV 档位（512k / 1M / 131k）在指定卡位上的标准四档速度。
# 这就是把"用例恒定下来"落地的脚本：备份 → 改 KV → 起引擎 → 等就绪 → 记槽位
# → 固定 6 轮 cases_v1 → 汇总（排除前 3 轮热身）。结果自动入 results.jsonl。
#
# 用法：
#   ./run_kv.sh 524288                  # 512k，默认卡位 1,2,3,4，6 轮
#   ./run_kv.sh 1048576                 # 1M
#   ./run_kv.sh 131072 0,1,2,3          # 131k 基线复现（11-14 的原始配置）
#   ./run_kv.sh 524288 1,2,3,4 8        # 8 轮
#
# 前置（脚本自己会做，但要知道）：
#   - 停 ComfyUI（引擎启动要求 MemAvailable ≥ 226 GiB）
#   - label 固定为 kv<k>-u099-<gpus 压缩>，nonce 随机，配置快照自动记录
#
# 口径（不要改）：
#   - util 可指定（默认 0.99 原值）；512k/1M 档 0.97 的对照见 bench/README.md 3.1b
#   - 端到端看 tps_wall，decode 仅参考；前 3 轮热身不计入结论
set -uo pipefail

KV=${1:?用法: run_kv.sh <kv_tokens> [gpus=1,2,3,4] [rounds=6] [util=0.99]}
GPUS=${2:-1,2,3,4}
ROUNDS=${3:-6}
UTIL=${4:-0.99}
PREFIX="kv${KV}-u$(echo "$UTIL" | tr -d '.')-$(echo "$GPUS" | tr ',' '_')"

ENGINE="${HOME:-/home/xpz}/dsv41-repo-test"
URL="http://127.0.0.1:9010"
CONTAINER="dsv41-pp4"
STAMP=$(date +%m%d_%H%M)
LOGDIR="${HOME:-/home/xpz}/manju/run_kv_${STAMP}"
RUNLOG="$LOGDIR/run.log"
mkdir -p "$LOGDIR"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$RUNLOG"; }

# ---------- preflight ----------
log "=== run_kv: KV=$KV  gpus=$GPUS  rounds=$ROUNDS  util=$UTIL ==="

# 备份 → 改 KV 与 util
cp "$ENGINE/run-pp4.sh" "$ENGINE/run-pp4.sh.bak-kv$KV-u$(echo "$UTIL"|tr -d '.')-$STAMP"
sed -i "s/--max-model-len [0-9]*/--max-model-len $KV/; s/--kv-cache-tokens [0-9]*/--kv-cache-tokens $KV/; s/--gpu-memory-utilization [0-9.]*/--gpu-memory-utilization $UTIL/" "$ENGINE/run-pp4.sh"
log "已改 KV=$KV util=$UTIL（备份: run-pp4.sh.bak-kv$KV-u$(echo "$UTIL"|tr -d '.')-$STAMP）"
grep -E "max-model-len|kv-cache-tokens|gpu-memory-utilization" "$ENGINE/run-pp4.sh" | sed 's/^/    /' | tee -a "$RUNLOG"

# 停 ComfyUI（内存前置） + 起引擎
sudo systemctl stop comfyui 2>/dev/null
free -g | head -2 | tee -a "$RUNLOG"
sudo docker stop -t 30 "$CONTAINER" >/dev/null 2>&1; sleep 6
( cd "$ENGINE" && bash run-pp4.sh --gpus "$GPUS" ) 2>&1 | tail -2 | tee -a "$RUNLOG"

# 等就绪
ok=0
for i in $(seq 1 45); do
  if curl -s -m 3 -o /dev/null -w "%{http_code}" "$URL/health" 2>/dev/null | grep -q 200; then
    log "READY after $((i*20))s"; ok=1; break
  fi
  sudo docker ps --format "{{.Names}}" | grep -qx "$CONTAINER" || {
    log "DIED ——"; sudo docker logs --tail 12 "$CONTAINER" 2>&1 | cut -c1-175 | tee -a "$RUNLOG"; exit 1; }
  sleep 20
done
[ "$ok" = 1 ] || { log "TIMEOUT(15min)"; exit 1; }

# 槽位与预算（档位对比的核心数字）
log "--- 槽位 ---"
sudo docker logs "$CONTAINER" 2>&1 | grep "final GPU cache plan" | grep -o "total=[0-9]* slots" | tee -a "$RUNLOG"
sudo docker logs "$CONTAINER" 2>&1 | grep "Hybrid cache budget" | sed 's/.*free=/free=/' | cut -c1-130 | tee -a "$RUNLOG"
nvidia-smi --query-gpu=index,power.limit,memory.used --format=csv,noheader | tee -a "$RUNLOG"

# 固定 N 轮（前 3 轮热身）
for i in $(seq 1 "$ROUNDS"); do
  n=$((RANDOM%8000+1000))
  log "round $i/$ROUNDS nonce=$n"
  ( cd "$ENGINE" && python3 bench/run_cases.py "$PREFIX" "$n" --set real ) 2>&1 \
    | grep -E "^\[config\]|^(code|json|zh|math)" | tee -a "$RUNLOG"
done

# ---------- 汇总 ----------
log "=== 汇总 $PREFIX ==="
( cd "$ENGINE" && python3 - "$PREFIX" <<'PY'
import json, sys, statistics
prefix = sys.argv[1]
rows = [json.loads(l) for l in open("bench/results.jsonl", encoding="utf-8") if l.strip()]
rs = [r for r in rows if r.get("label") == prefix]
if not rs:
    print("无数据"); sys.exit(0)
walls = [statistics.mean([c["tps_wall"] for c in r["cases"] if c.get("tps_wall")]) for r in rs]
decs  = [statistics.mean([c["tps_decode"] for c in r["cases"] if c.get("tps_decode")]) for r in rs]
ttfts = [statistics.mean([c["ttft_s"] for c in r["cases"] if c.get("ttft_s")]) for r in rs]
slots = next((r["config"].get("kv-cache-tokens") for r in rs if r.get("config")), "?")
print("轮数=%d  KV=%s" % (len(rs), slots))
print("  每轮 wall: %s" % " ".join("%.1f" % w for w in walls))
print("  每轮 dec : %s" % " ".join("%.1f" % d for d in decs))
w3 = statistics.mean(walls[-3:]) if len(walls) >= 4 else statistics.mean(walls)
d3 = statistics.mean(decs[-3:]) if len(decs) >= 4 else statistics.mean(decs)
print("  → 后 3 轮: wall=%.2f  decode=%.2f  ttft=%.3f" % (w3, d3, statistics.mean(ttfts[-3:])))
PY
) 2>&1 | tee -a "$RUNLOG"
log "日志: $RUNLOG"
log "恢复别的档位: ./run_kv.sh <kv> ；恢复 ComfyUI: 先引擎后 ComfyUI（内存顺序）"
