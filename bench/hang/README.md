# GPU 平台冻结复现包

在一张 NVIDIA 卡上复现「整机冻结（hard hang）」的最小用例：**不需要大模型，不需要 ComfyUI，不需要容器**。
3 分钟内出结果，成功复现时机器会完全失去响应，只能断电重启。

## 1. 它到底做什么

三个负载同时压在同一张卡 / 同一台主机上：

| 负载 | 参数 | 模拟的现实场景 |
|---|---|---|
| 主机侧持续读内存 | `DRAM_GBPS`（默认 10GB/s，16 线程） | CPU 侧专家计算（MoE 放内存、AVX2 算） |
| 文件 → pinned → 显存 的批量拷贝 | `STAGE_GBPS`（默认 1.0GB/s） | 推理/出图把权重搬进显存（PCIe DMA） |
| 同一张卡跑 matmul | 固定开启 | GPU 正在执行的 kernel |
| 显存常驻 | `RESERVE_MB`（默认 32768） | 模型已经占着的显存 |

## 2. 前提

- Linux + NVIDIA 驱动（`nvidia-smi` 可用）
- Python 3 + 能 `import torch` 且 `torch.cuda.is_available()` 为真
- 磁盘有 4GiB 空闲（会自动生成测试文件；也可用第二个参数指定任意 >1GB 的文件）
- **一张可被独占的卡**（脚本会往该卡申请常驻显存 + 跑 matmul）

## 3. 怎么跑

```bash
cd gpu-hang-repro
bash run-lethal.sh 0                  # 用 0 号卡，默认致命配置
bash run-lethal.sh 0 /path/big.bin    # 指定 staging 文件（任意 >1GB 文件）
```

可选环境变量（放在命令前面）：

```bash
DRAM_GBPS=44 bash run-lethal.sh 0     # 把主机读带宽提到 44GB/s
DRAM_GBPS=0  bash run-lethal.sh 0     # 对照组：不压主机带宽
RESERVE_MB=8192 bash run-lethal.sh 0  # 小显存卡：把常驻降到 8GB
DURATION=300 bash run-lethal.sh 0     # 跑更久
```

如果机器缺 torch，可先 `PYTHON=/path/to/venv/bin/python bash run-lethal.sh 0`。

## 4. 怎么判断结果

| 现象 | 结论 |
|---|---|
| 30–180 秒内 SSH/控制台完全失联、ping 丢包或偶通、必须断电 | **复现成功**（和我们的现象一致） |
| 脚本报错退出（OOM、CUDA error 等） | 没复现，是配置问题，看 `logs/` 里的报错 |
| 跑满 `DURATION` 秒正常结束 | **未复现** |

日志在 `logs/repro_gpu*.log`，每 5 秒记一行显存/利用率，冻结时最后一行就是死亡瞬间的状态。

## 5. 我们这台机器上的对照数据

| 主机带宽 | 搬运 | 显存常驻 | GPU 计算 | 结果 |
|---|---|---|---|---|
| 44GB/s | 1.0GB/s | 32GB | 无 | ✅ 跑满 180s |
| 44GB/s | 1.0GB/s | 32GB | 有 | ❌ 30–60s 冻结 |
| 10GB/s | 1.0GB/s | 32GB | 有 | ❌ 冻结 |
| 10GB/s | 1.0GB/s | 32GB | 有，但数据已在内存（mlock） | ❌ 冻结 |
| 10GB/s | 1.0GB/s，块 4MB | 32GB | 有 | ❌ 冻结 |
| 无（Qwen 全驻留） | 出图自身的搬运 | — | 有 | ✅ 正常出片 |

由此得到的判据：**PCIe 搬运与 GPU kernel 在时间上重叠，就会冻结**；数据来自磁盘还是内存、搬运块大小、主机带宽高低都不改变结论。只去掉 kernel，就不冻结。

## 6. 建议的验证顺序（对方机器）

1. `bash run-lethal.sh <GPU>` —— 看能不能复现
2. 若能复现，再跑 `DRAM_GBPS=0 bash run-lethal.sh <GPU>` —— 判断主机带宽是否有份
3. 再跑 `RESERVE_MB=8192 ...` —— 判断显存占用量是否关键
4. 换一张卡重复第 1 步 —— 判断是不是某张卡 / 某个插槽的问题

第 4 步很值得做：我们这边 GPU4 走独立根端口，其余四张走另一个根端口，如果只有某一张卡能复现，那就是硬件/链路问题而不是平台通病。

## 7. 安全提示

- 复现会**硬冻结整机**，需要手动断电。不要在有重要任务的机器上跑
- 冻结时没有 Xid、没有内核栈，属于硬件/固件级挂死，别指望 dmesg 里有线索
- 脚本不会写系统配置、不会改驱动、不会动别的卡；测试文件默认放 `/tmp`

---

## 4. 补充：逐卡扫描（`sweep_cards.sh`）

本目录另有一份 `sweep_cards.sh`，对指定卡逐张跑同一套"搬运 + 计算"负载，每张 150 秒。
冻结会中断整轮（机器要断电），**日志最后一行就是出问题的那张卡**。

```bash
./sweep_cards.sh 0,1,2,3,4 150
```

2026-11-16 用它定位到：只有 GPU4（`0000:09:00.0`）冻结，GPU0/1/2/3 各 150 s 全部存活。
结论见 [`docs/GPU4-HANG.md`](../../docs/GPU4-HANG.md)。
