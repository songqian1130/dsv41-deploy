#!/usr/bin/env python3
"""ComfyUI 任务完成卸载的驱动级模拟：分配 N GB 显存 → DMA 回 host → del → empty_cache（真 cudaFree）。
循环执行，模拟"大模型任务结束后整体 offload"的事件。
用法: gpu0_offload_storm.py [GB] [循环间隔秒] [持续秒]
默认: 8 GB, 间隔 2s, 300s
"""
import sys
import time

import torch

GB = int(sys.argv[1]) if len(sys.argv) > 1 else 8
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0

dev = "cuda:0"
torch.cuda.set_device(0)
print(f"[storm] dev={dev} block={GB}GB interval={INTERVAL}s duration={DURATION}s", flush=True)

host = torch.empty(GB * 2**30, dtype=torch.uint8, pin_memory=True)
n = 0
t0 = time.time()
while time.time() - t0 < DURATION:
    x = torch.empty(GB * 2**30, dtype=torch.uint8, device=dev)
    x.fill_(1)                                   # 逼真落地
    host.copy_(x)                                # VRAM -> host DMA（offload）
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()                     # 驱动级释放（真 cudaFree 大块）
    n += 1
    print(f"[storm] cycle {n} done @ {time.strftime('%H:%M:%S')} (freed {GB}GB)", flush=True)
    try:
        import os
        os.fsync(1)
    except OSError:
        pass
    time.sleep(INTERVAL)
print(f"[storm] DONE cycles={n}", flush=True)
