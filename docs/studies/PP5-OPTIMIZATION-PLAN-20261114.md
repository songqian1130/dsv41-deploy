# PP5 / PP4 优化计划（2026-11-14）

## 0. 三条结论

1. **新 PP5 更快的原因已经可以定量拆开：主要是 dspark 投机解码，不是 CPU 专家。**
   新引擎每步 69ms、每步产出 2.80 token → 24.6ms/token（40.6 tok/s）；旧引擎单 token 25~33ms（30~40 tok/s）。**单 token 成本两者几乎相同，差距全在"每步吐几个 token"**。若把 dspark 关掉，新引擎按 69ms/token 只有 14.5 tok/s。0.99 util 带来的专家覆盖率（7168/7680 = 93.3%）是第二个因素，旧引擎的专家 100% 在卡上，没有这方面短板。

2. **"新 PP5 死机"现在有对照组了，而且指向引擎路径而不是纯平台。**
   | 引擎 | 连续运行 | 请求数 | 结局 |
   |---|---:|---:|---|
   | 旧 PP5（aiultrayang，8,8,8,8,8，全 GPU 专家） | **45.2h** | 6,837（151/h） | 正常关机（人工） |
   | 新 PP5（18gogogo，[5,5,5,4,21]，CPU 专家） | 16.7h / 25.4h / 5.7h / 8.2h | 2,223 / 3,352 / … | **4 次无痕冻结** |
   旧引擎负载还更高（151 vs 132 req/h）。按新引擎 MTBF≈14h 的指数模型，旧引擎 45.2h 不崩的概率只有 **e^-3.23 ≈ 4%**。
   剩下要定位的是"新引擎里的哪一部分"：末级 21 层的功率阶跃、host 专家 LRU（仅 7.7 GiB，见 §1.5）、dspark、0.99 只剩 1% 余量。

3. **PP4 慢不是调参问题，是专家驻留悬崖；落 CPU 的专家实际读的是 NVMe（3.56 GB/s），不是内存。**
   4 卡 248GB 装不下 fp4 240GB + KV/激活 → 约 10% 专家落 CPU → 每步 ~39 条路由 × 6ms ≈ 237ms → 9.5 tok/s；5 卡 310GB 命中 99.9% → 18.96ms/token → 52.7 tok/s。**这 6ms 已被 E7 测实**：同样 18MiB 块，NVMe 单线程 8.16ms、内存单线程 3.29ms。所以 PP4 的出路是"让那 10% 专家进内存"或"加第二张盘"，不是补内存通道。
4. **全机真正的瓶颈是那块 SanDisk NVMe：实测 3.56 GB/s，14 线程即饱和。** 内存有 38.74 GB/s（10.9 倍余量）、DRAM 通道只有 4 条但无关紧要。所有"CPU 专家/engram 慢"的现象都要往盘上归因。
5. **engram 在 prefill 阶段被 4 KB 页放大了 19~24 倍，这是 TTFT 的主要成本（§3.7 实测）。**
   17.5k token 的 prefill 要从盘上读 **4.96 GB**，而有用数据只有 207 MB；37k token 读 8.40 GB。随机 4 KB 撞 IOPS 上限，盘只有 **550~570 MB/s** 可用。外推 20 万 token ≈ **85~95s 纯 I/O**。**decode 阶段则完全无感**（34 KB/步，占 69ms 的 0.05%）。所以：改 engram 只会动 TTFT，不会让 40 tok/s 变快；而 TTFT 的优化空间有 20~30 倍。

## 1. 关键量化基线

| 指标 | 本机 | 参考机（README） | 影响 |
|---|---:|---:|---|
| 内存带宽（自测 56T） | 读 55.0 GB/s / Triad(2R+1W) 25.2 GB/s | ~99 GB/s | CPU 专家、host 专家回填 |
| **NVMe 读（O_DIRECT 18MiB 块）** | **1 线程 2.31 GB/s（8.16ms/块）/ 14 线程起饱和 3.56 GB/s** | — | **CPU 专家、engram 的真实上限** |
| **内存 mmap 同尺寸块** | 1 线程 5.74 GB/s（3.29ms）/ 28 线程 **38.74 GB/s** | — | 比 NVMe 快 **10.9 倍** |
| 内存通道 | **8×32GB=256GB，但只占 4/8 条通道（2DPC）**，A/B/E/F 空 | 8 通道 | 带宽上限只有一半 |
| CPU | 2× E5-2680v4，28C/56T，2133 MT/s | EPYC 7K62 48C/96T | cpu_threads 上限 28 |
| PCIe | Gen2 x4（BIOS 拆分），H2D 1.61 GB/s | PCIe3 x16，3.7 GB/s | 专家回填、PP 边界 |
| CPU 专家单条 | 17.94 MiB / ~6ms ≈ 2.9 GB/s | — | 有效带宽仅为实测 DRAM 的 15% |
| 步时（新 PP5） | 69ms，2.80 tok/步 | — | dspark 接受率 60% |
| 崩溃前最后 200ms | GPU4 69.3W→316.5W，util 18%→65%，70ms 后整机停摆 | — | 末级突发是唯一时间相关量 |

## 1.5 新引擎的内存账（2026-11-14 核对启动日志，修正此前"134GiB host 专家"的错误说法）

**134.47 GiB 不是 host 需求**，它是 `20 个 MoE 层 × 384 专家 × 17.94 MiB = 134.1 GiB`，即**全部专家的总体积**（出自 `aiultrayang-pp4-20261103/progress.md`，原文就是"20层专家估算134.47GiB"）。此前把它当成"新 PP5 额外需要的 host 内存"是误读。

11-08 04:41 的实测日志（`/var/log/dsv41-crash/dockerjson-2026-11-08_124228.log`）：

```
Calibrating GPU expert cache coverage=93.54% (7184/7680 routed experts, 125.788 GiB);
host LRU 7.704 GiB
Hybrid final GPU cache plan=[...], total=7184 slots
```

| 项 | 数值 | 说明 |
|---|---:|---|
| 检查点总大小 | 475.25 GiB | 含 engram 189 GiB（在 SSD 上，不占内存） |
| 全部专家 | 134.1 GiB | 7680 个 × 17.94 MiB |
| **放上 5 张卡** | **125.788 GiB（93.54%，7184 个）** | 按 util 0.99 尽力塞 |
| **host LRU** | **7.704 GiB** | 496 个专家兜底（`host_cache_gib: 8` 的上限） |
| engram 行缓存 | 2 GiB × 2 rank = 4 GiB | `engram_cache_gib: 4` |
| 各 rank 加载后显存 | 37.5 / 34.89 / 35.12 / 27.96 / 21.5 GiB | `[5,5,5,4,21]` |

**结论：新 PP5 的主机侧额外开销是 ~12~15 GiB 量级，不是 134 GiB。**
- 旧引擎 `[8,8,8,8,8]` + `VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,0` → 专家 100% 在卡上，host 无专家副本。
- 新引擎专家覆盖率反而更低（93.54%），因为它把空间让给了 dspark 与 524288 的 KV 池。
- **两边"显存都占满"是设计使然**：util 0.97/0.99 是目标占用率，剩余空间被 KV 池吃干净（日志：`Available KV cache memory: 22.38 GiB`、`Reserving [0.441,0.8,0.8,0.359,1.241] GiB KV per worker`）。占满不等于有东西被 offload 到主机。
- **dspark 的权重是同一个 checkpoint 里的 `vocab-sharded Markov head`**（`speculator.py:364`），量级几百 MB~2 GiB；它主要多占的是验证用的 draft KV 槽（`num_speculative_tokens=3` × 4 并发），已含在上面那 0.3~1.2 GiB 的 KV 预留里。**dspark 不可能多出上百 GB。**

**对 E9 的影响**：engram 全放内存（189 GiB）与新引擎并不冲突——新引擎只要 ~14 GiB，合计 203 GiB，`available` 从 218 掉到 ~15~29 GiB（取决于引擎自身占用）。仍然低于 64 GiB 安全线，但不像"352 > 247"那样根本不可能。

## 2. 实验清单（按性价比排序）

| # | 假设 | 改动 | 预期 | 判据 | 耗时 |
|---|---|---|---|---|---|
| E1 | 冻结由末级功率阶跃触发 | GPU4 功率上限 200→150W，与其余四卡一致 | 峰值削 25%，速度 −8~15% | 跑 72h（5×MTBF）不崩 | 3 天，零成本 |
| E2 | 症状来自 host 专家路径的内存风暴 | 同引擎同切分，`host_cache_gib 8→4`、`engram_cache_gib 4→2`、`cpu_threads 28→24` | 主机侧抖动下降 | 72h 不崩且 decode 掉幅 <10% | 3 天 |
| E3 | 21 层末级是爆发点 | 试 `pipeline_layers [6,6,6,6,16]`；若引擎允许再试 `[8,8,8,8,8]` + 关 CPU offload | 峰值再降，速度接近 legacy+spec | 启动不报错 + 72h | 1 天试启动 |
| E4 | dspark 是速度主因 | 新引擎删 `--speculative-config` 跑一轮 | 应回落到 ~14 tok/s | 与预测误差 <20% 即证实 | 半天（1 次重启） |
| E5 | ~~补通道~~ **已作废** | 科脑 X99-D4 双路是"八槽双通道"：每 CPU 只引出 2 条通道 × 2DPC，8 槽插满 = 4 通道，**换顺序/换槽位都无效**。要 8 通道只能换 16 槽板（X10DRI / Z10PE-D16） | 无改动方案 | EDAC 已证实 mc0/mc1 = 0 MB | 0 |
| E7 | ~~并发度不足~~ **已执行，假设被证伪** | 见 §3.5：盘 14 线程即饱和 3.56 GB/s，28/56 线程无增益 | 不是并发度问题，是盘的顺序读上限 | 实测 | 已完成 |
| E8 | engram 缓存太小 | `engram_cache_gib` 8→64（legacy）/ 4→64（新引擎） | 随机 n-gram 下命中率≈容量比 34% → prefill I/O −30%，**TTFT −25~35%**；decode 不变 | `diskstats` 读量降幅 + 长 prompt TTFT | 半天 |
| **E9** | **4 KB 页放大是 TTFT 主因** | `engram storage: ssd → ram`（默认值本来就是 ram），独立容器 `dsv41-pp5-ram` 先试 | 20 万 token 的 engram I/O 从 85~95s 降到 **2~4s**；单流 TTFT −16~31%，**并发 −60~70%**；decode 不变 | 长 prompt TTFT + 并发 TTFT + `available` 不低于 20GB | 半天（含 1 次重启） |
| E10 | 双盘分片 | 第二块 NVMe 挂到另一路 PCIe x4，engram 按 rank 分片 | I/O 减半 → TTFT −50% | 两块盘各自的 `diskstats` | 需硬件 |

| E6 | PP4 只在 E5 后才有意义 | E5 完成后再评估 4 卡 | 39×1.5ms≈59ms → ~17 tok/s | 仍远低于 PP5 → 放弃 | 1 天 |

**建议执行顺序：E9（半天，TTFT 收益最大且可独立容器验证）→ E4（半天，把速度归因钉死）→ E1（3 天）→ E2/E3 择一。**
E5 已作废（主板只有 4 通道可引），E6 随之作废（PP4 的 CPU 专家成本没有下降空间）。
E8 与 E9 是互斥关系：E9 直接把 189 GiB 钉死后，E8 的 64GiB 缓存就没有意义了；所以先做 E9，失败再退回 E8。

## 3. 内存通道：已查清，是主板硬限制（无改动空间）

- 主板是 **科脑 X99-D4 双路（ZX-DU99D4）**，官方说明书：`DDR4×4（CPU1 侧）+ DDR4×4（CPU2 侧）`，最大 256GB（32GB×8）。**物理只有 8 个槽，已全部插满**，没有空槽可换。
- **EDAC 硬证据**（`/sys/devices/system/edac/mc/`）：`mc0`(SrcID#1_Ha#0)、`mc1`(SrcID#0_Ha#0) 的 `size_mb` 都是 **0**；`mc2`(SrcID#1_Ha#1)、`mc3`(SrcID#0_Ha#1) 各 **131072 MB**，且都只有 `Chan#0_DIMM#0/#1` + `Chan#1_DIMM#0/#1`。
- 翻译：E5-2680v4 每颗 CPU 有 4 条通道（2 个 iMC × 2 ch），**这块板每颗只引出 2 条通道，每条通道 2 个槽（2DPC）**。8 槽插满 = 4 通道 × 2DPC，不是 8 通道 × 1DPC。
- **所以换插槽顺序没有任何用**：CPU1 侧 4 槽只对应 2 条通道，无论怎么排列都是 2 通道 × 2DPC。要 8 通道必须换 16 槽板（超微 X10DRI、华硕 Z10PE-D16 之类），不是调参数能解决的。
- 4 通道 DDR4-2133 理论 68 GB/s，实测读 55 GB/s（**利用率 81%，已经榨干**），Triad 25.2 GB/s。带宽从此封顶。
- 2DPC 4-rank 32GB 条锁 2133 是 JEDEC 行为，拆成 1DPC 才能争 2400，但那样容量要减半（128GB），对 LLM 不划算。
- 唯一可能的小收益：BIOS 里确认内存频率是否被锁 2133；若条子额定 2400 可试。当前 `Speed` 与 `Configured Memory Speed` 都是 2133，多半是 4R 条本身的额定值。
- 板子暴露单 NUMA（0-55 全在 node0），BIOS 里若有 Node Interleaving 开关可以试着关掉，看能否分出 2 个 node；这能改善绑核，但不增加带宽。
- **真正的问题转移了**：CPU 专家路径只跑出 2.9 GB/s，而机器能给 55 GB/s——**用了 5%**。瓶颈是访存并行度/同步，不是带宽。见 E7。

## 3.5 E7 实测：瓶颈是 NVMe，不是内存（2026-11-14 09:30）

测试程序 `.workbuddy/tmp/expert_bw.c`（O_DIRECT 与 mmap 两种模式，18MiB 随机块，md5 sink 防优化），
在 95GB 的 `model-00047`（engram 表，冷数据）与预热后的 `model-00010`（7.4GB）上跑：

| 路径 | 1 线程 | 4 线程 | 14 线程 | 28 线程 | 56 线程 |
|---|---:|---:|---:|---:|---:|
| **NVMe O_DIRECT** | 2.31 GB/s / 8.16ms | 3.25 | 3.52 | 3.55 | **3.56** |
| **内存 mmap** | 5.74 GB/s / 3.29ms | — | — | **38.74** | 35.84 |

**三条硬结论**

1. **CPU 专家的 2.99 GB/s ≈ NVMe 上限 3.56 GB/s 的 84%**，单次 6ms 与实测单块 8.16ms 同量级（略快说明有部分缓存命中）。**它已经跑在盘的满速上，没有任何浪费。**
2. **并发度不是瓶颈**：盘在 14 线程就到 3.52 GB/s，28/56 线程只多 0.04 GB/s。E7 原假设"6ms 是并发度不足"**被证伪**。
3. **内存侧有 10.9 倍余量**：28 线程 38.74 GB/s。只要数据在 page cache 里，同一次读取只要 3.29ms（单线程）/ 更快（并发）。

**这同时解释了"新 PP5 为什么快"**：dspark 让**同一次 SSD 读取产出 2.8 个 token**。步时由 SSD 主导（69ms），所以每 token 成本 = 69/2.8 = 24.6ms；旧引擎 26~30ms 一步一个 token。单 token 成本相近，差距全在"一次盘读能摊薄到几个 token"。

**推论——改进方向彻底变了**

| 原方向 | 状态 | 原因 |
|---|---|---|
| 补内存通道（E5） | ❌ 作废 | 内存有 13 倍余量，补了也没用 |
| 提高 CPU 专家并发度（E7） | ❌ 证伪 | 盘 14 线程就饱和 |
| **加内存缓存命中率** | ✅ 首选 | 命中一次等于省一次盘读，直接砍步时 |
| **加第二张 NVMe 分片** | ✅ 次选 | X99 走 PCIe 3.0 x4，单盘 3.56 GB/s 已满；双盘理论 7 GB/s |
| 换 PCIe 4.0 平台 | ⚠️ 大改 | 上限 7 GB/s，但要换 CPU/主板/内存 |

## 3.6 立刻可做的参数调整

现在两个引擎的 engram 缓存都远低于可用内存（247GB 总、218GB available）：

| 引擎 | 当前 engram_cache | 建议 | 依据 |
|---|---:|---:|---|
| legacy（在跑） | 8 GiB | **48~64 GiB** | 189GiB 表 → 命中率 4%→25~34% |
| 新引擎 | 4 GiB | **48~64 GiB** | 同上，且新引擎受益更大 |

风险控制：先上 48GiB，盯 `available` 不低于 80GB 再往上加；`vm.overcommit_memory` 保持默认，出现 OOM 就退回 32GiB。

## 3.7 E9 实测：engram 的 SSD 读放大 19~24×，prefill 被它拖死（2026-11-14 11:00）

对运行中的 legacy（`storage='ssd'`，16384 行/rank 缓存）发**唯一随机内容**的 prompt，同时读 `/proc/diskstats` 的 `nvme0n1` sector 增量：

| prompt tokens | 物理读 | 有用数据 | 放大 | 实测 TTFT | B/行 |
|---:|---:|---:|---:|---:|---:|
| 17,561 | **4.96 GB** | 207 MB | **23.9×** | 8.71s | 6,171 |
| 37,255 | **8.40 GB** | 440 MB | **19.1×** | 15.32s | 4,925 |

- 每 token 固定 **48 行**（24 hash 列 × 2 层，layer1/layer14 各 1 个 rank），有用量仅 **258 B/行**。
- 但路径是 `engram_ssd.py` → `libdsv41_cuda.so` 的 **O_DIRECT + Linux AIO**，最小粒度 4 KB 页，权重要一页、scales 又要一页 → **每行实际读 ~5~6 KB，放大 19~24 倍**（`stats` 里的 `weight_pages`/`scale_pages` 计数器就是这个）。
- 盘的实际速率只有 **550~570 MB/s**，不是 3.56 GB/s——随机 4 KB 撞的是 IOPS 上限（≈14 万 IOPS），顺序带宽用不上。
- 复现脚本：`.workbuddy/tmp/engram_probe2.py`、`engram_probe3/4.py`（`python3 probe4.py <words> <seed>`，**必须换 seed**，否则 n-gram 行命中缓存，实测会掉到 11×）。

**外推**：20 万 token 的 prefill ≈ **48~56 GB 物理读 → 85~95 s 纯 I/O**。这解释了历史截图里 140~208s 的 TTFT 的很大一部分——此前把它全归给"并发排队"是不完整的。

**必须同时记住计算下限**：用同一段文字重复 26 遍（n-gram 全命中缓存，只读 38 MB）测到 16.4k token 只要 5.60s → **prefill 计算约 2.9k tok/s**，20 万 token 光算就要 **≈68s**。所以：

| 场景 | 现在 | 换成 ram 模式后 | 收益 |
|---|---:|---:|---|
| 17.5k 单流 TTFT | 8.71s（I/O 8.7s / 算 6.0s） | ~6.0s | −31% |
| 37k 单流 TTFT | 15.3s（I/O 14.7s / 算 12.8s） | ~12.8s | −16% |
| 200k 单流 TTFT | ~90s（I/O 88s / 算 68s） | ~68s | −25% |
| **3 路 200k 并发** | 盘要 260s+ → 全部排队到 200s 以上 | 盘不再是瓶颈 | **−60~70%** |

关键在最后一行：**一块盘的 engram 读把整个 prefill 聚合吞吐钉在 ~2.3k tok/s**（0.57 GB/s ÷ 0.25 MB/token），并发再多也只是排队——这正是最初那次"三路一起卡住、TTFT 140~208s"的真正机理。

**结论：prefill 阶段唯一值得动的是"别走 4 KB 页"。**

| 方案 | engram 侧 I/O（20 万 token） | 效果 | 代价 |
|---|---:|---|---|
| 现状 ssd + 8GiB | 48~56 GB @570MB/s ≈ 85~95s | 基准，且并发时全局限流 | — |
| ssd + cache 64GiB | 命中率≈容量比 34% → ≈ 60 GB ≈ 100s→63s | TTFT −10~20% | 64 GiB 内存 |
| **`storage="ram"`（UVA 直读）** | **2.4 GB @0.6~1.0GB/s ≈ 2.4~4s** | **I/O 从瓶颈变噪声；单流 −16~31%，并发 −60~70%** | 189 GiB pinned |
| 双 NVMe 分片 | 减半 → 45s | 单流 −3~10%，并发 −50% | 空 PCIe 槽 + 改分片 |


**`ram` 模式的账要算清**：`EngramConfig.storage` 默认值本来就是 `"ram"`，走 `cudaHostAlloc(Mapped)` 的 pinned host memory，Triton kernel 直接在 UVA 上按 **256 B 粒度** gather，没有页放大。代价是每 rank 一个 94.42 GiB 的 pinned 块，两层合计 **189 GiB**；当前 `total 247 / used 29 / available 218`，放完只剩 ~58 GiB。**能放下，但没有余量**，且 pinned 不可回收——分配失败会直接起不来（代码里有 `_exact_pinned_tensor` 的降级路径，但降级到 torch 分配器可能超尺寸）。



## 4. 验收与监控

- 已具备：`dsv41-blackbox`（1Hz 全指标）、`dsv41-fastwatch`（5Hz 六字段）、`dsv41-postmortem`（每次开机自动验尸）、`dsv41-salvage`（重启前抢救容器日志）、`panic=20`。
- 建议新增一条判据：**末级功率阶跃事件**——GPU 功率在 100ms 内变化 >150W 时写事件行，用来验证 E1/E3 是否真的削掉了爆发点。
- 针对 E8 再加两条：`engram 命中率` 与 `NVMe 读吞吐（/proc/diskstats nvme0n1 每秒增量）`。判据是改 cache 后 NVMe 读降幅与 tok/s 升幅是否对得上（1 GB/s ≈ 0.3ms/token 的量级）。

## 5. 风险与回滚

- 所有实验都用独立容器名 + 端口，`dsv41-pp5-ctl.sh` 保持指向**当前稳定配置**；实验容器只做短时切换。
- 当前状态（2026-11-14 09:06 开机）：`dsv41-pp5-legacy`（旧引擎，45.2h 对照方案）已启动加载，这是唯一有长跑证据的配置。
- **E5/E6 已作废**（内存通道是主板硬限制，补了也不增加有效带宽，因为瓶颈在 NVMe），不再需要停机拆机。
- E8 只改一个 env 变量，回滚 = 改回 8/4 重启；主要风险是内存压力，用 `available ≥ 80GB` 作为红线。

## 6. 需要拍板的三件事

1. 是否允许把 GPU4 功率从 200W 降到 150W（E1 的前提，可随时改回）。
2. 是否允许**加装第二张 NVMe**：X99 的 M.2/PCIe 走 3.0 x4，单盘 3.56 GB/s 已跑满，双盘分片理论翻倍；需要确认还有空闲 PCIe 槽和供电。
3. 这次要"稳"（旧引擎 30-40，已证 45h）还是"快"（新引擎 40-65，MTBF 14h）——决定 E1~E3 的取舍力度。
4. 是否愿意把 `engram_cache_gib` 直接上到 48GiB 试跑（E8，改一个数字，半小时见结果）。

## 7. 外部优化项移植评估（2026-11-14 11:45，代码实证，非推理）

针对此前从 Volta-sPLIt / Volta Max / Delta-Xolotl 调研里摘出的 7 条，逐条进镜像 `grep` 源码确认是否存在。
**两个引擎是两个不同镜像，必须分开查**：

- 旧引擎（ram / legacy / dspark 之外）：`local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102`，代码在 `/usr/local/lib/python3.12/dist-packages/vllm`
- 新引擎（= 新 PP5 原版）：`18gogogo/170hx4-256r-dsv41-1017-flash:sm80`，代码在 `/workspace/vllm-backport/vllm`

| 优化项 | 调研声称收益 | 本机是否存在 | 判定 |
|---|---|---|---|
| MRV2 Runner | +10%（120.5→134.2） | **有**，就是 `VLLM_USE_V2_MODEL_RUNNER`（12 处注释均指 Model Runner V2） | **唯一真正的空白点**；但 V2 与 `VLLM_PP5_LAZY_RECEIVE` 互斥，无法直接叠加在 ram 模式上 |
| 路由偏置 route bias 0.28 / `STRATA_BIAS_REFRESH` | offload 场景 +25% | **无**（`route_bias` 0 命中；`route_tail_skip` 是 CPU MoE 跳尾，语义不同） | 不可开。且旧引擎专家 100% 在卡（`VLLM_CPU_OFFLOAD_GB_PER_RANK=0`），没有"卡上/卡下"可分，偏置无优化对象 |
| DSpark k=5 + 耦合采样 | k=5 接受 1.85→2.25 | dspark 有（63 处），**coupled 采样无**（13 处全为 decoupled 等无关词） | k=5 需更多 draft KV 槽，显存已无余量（k=3 时 KV 仅剩 1.93 GiB）；且实测 k=3 每步 62 ms 已净亏 |
| PP_KEEP_BATCH | 128K 下 +24.8% | **无**（`keep_batch` 唯一命中是 `isaac.py` 的局部变量 `keep_batch_dim`） | 不可开。且 3~4 并发产生不了大量空 pipeline step，收益前提不成立 |
| packed KV + KV 热环 `KV_HOT_POSITIONS` | 300K 下每步 −1.09 ms | **无**（`hot_positions` 0 命中） | 不可开。KV 池只剩 1.93 GiB，也没有额外 HBM 做热环 |
| 前缀缓存分槽 | 首 token 20.6~49.2s → 0.57~0.88s | **已具备**：`--enable-prefix-caching` 一直在开，实测命中 94.6% | 已享受。TTFT 140~208s 的真实成因是 engram SSD 19~24 倍读放大 + 排队，已由 ram 模式解决 |
| engram 1G hugetlb 池 / NUMA 绑定 / 热集常驻 / SSD read-ahead | 冷启动差一倍、+22% | hugetlb **0**、read_ahead **0**；`numa_bind` 有（`config/parallel.py:291`） | ram 模式已把 engram I/O 归零（TTFT 11.92→6.45 s），这几条都是给 **SSD 路径**打的补丁，没有作用对象。`numa_bind` 在本机是单 NUMA（BIOS Node Interleaving），绑了等于没绑 |
| chunked prefill 8192 | 配合 PP_KEEP_BATCH +24.8% | **有**：`--max-num-batched-tokens`（新引擎 4096、旧引擎 2048） | 可试但要权衡：chunk 越大，**首个 chunk 越久 → TTFT 反而变差**，改善的是总 prefill 时长/吞吐 |

**小结：7 条里 6 条要么在本机镜像里根本没有实现（需移植代码，不是开开关），要么受益前提不成立；1 条（前缀缓存）早已在享受。**

### 7.1 为什么 dspark "学不到"新 PP5 的优点——不是配置问题

用 `diff -u run-pp5-prod.sh run-pp5-dspark.sh` 验证过：实验脚本与新 PP5 原版**逐行一致**，
同镜像、同 `pipeline_layers [5,5,5,4,21]`、同 `cpu_threads [28,28]`、同
`{"method":"dspark","num_speculative_tokens":3,"enable_adaptive_verification":true}`、
同 `engram_storage ssd / engram_cache_gib 4 / host_cache_gib 8`、同 util 0.99。
差异只有容器名、两行 ulimit 和引号风格。

所以 36.47 tok/s **就是新 PP5 在本机的真实水平**，不是没配好：

- 11-08 生产日志：69 ms/步 × 2.80 tok/步 = **24.6 ms/token**
- 11-14 实测：62 ms/步 × 2.48 tok/步 = **25.0 ms/token**

两者一致，复现成立。

**结构性原因（关键）**：dspark 的收益来自"一次 engram 4 KB 页读摊薄到约 2.8 个 token"——
这是 **SSD 路径**的红利。而 ram 模式把 engram I/O 直接归零（TTFT 11.92 s → 6.45 s），
这块红利已经被更彻底地吃掉了。此时再上 dspark，只剩成本：
`+7.39 GiB 显存 → 496 个专家被迫下放主机 → 每步 62 ms`。

一句话：**不是"没学到"，是"学不了"——收益项已被更强的方案覆盖，成本项照旧要付。**

### 7.2 还没试过的组合（按性价比）

1. **V2 runner 单独开、不带 dspark**（关 `VLLM_PP5_LAZY_RECEIVE`、开 `VLLM_USE_V2_MODEL_RUNNER=1`，其余同 ram）。
   这是唯一没测过的组合，用来量 MRV2 的调度收益能否补回失去惰性接收的损失。一次启动 + 一个 C1/C2 即可判定。
2. `--max-num-batched-tokens` 2048 → 4096 → 8192，只看 22k/200k 的 TTFT 与总时长曲线，找拐点。
3. 其余 5 项要改引擎代码移植，属于"另一个项目"，不建议在稳定性未定时做。

## 8. 新 PP5 到底快在哪（2026-11-14 11:58 定稿结论）

### 8.1 结论：提速来源只有 dspark 一个，而且它补的是自己挖的坑

把"每步耗时 × 每步产出"拆开，一切就清楚了：

| 配置 | 每步耗时 | 每步产出 | 单 token 成本 | 等效速度 |
|---|---:|---:|---:|---:|
| 旧引擎 V1 + 惰性接收（ram） | 23.2 ms | 1.00 token | **23.2 ms** | **43.18 tok/s** |
| 新引擎 V2 + dspark k=3 | 62.0 ms | 2.48 token | **25.0 ms** | 36.47 tok/s |
| 新引擎 V2 + dspark（11-08 生产） | 69.0 ms | 2.80 token | **24.6 ms** | ~40 tok/s |
| 新引擎**不开** dspark（按同一步时推算） | ~62 ms | 1.00 token | 62 ms | **~16 tok/s** |

**关键**：新引擎平台本身（V2 + 专家下放 + 无投机）只有 ~16 tok/s。
dspark 把它拉到 36（+125%），但只是追回到旧引擎的水平线下方——
因为旧引擎不用付那 62 ms 的步时，直接就是 23 ms。

所以"新 PP5 更快"这个印象的来源是**内部对比**（16 → 36，看起来翻倍），
不是**跨配置对比**（36 vs 43，实际更慢）。
历史 132 请求/小时 对 旧引擎 151 请求/小时，也是旧引擎更高。

### 8.2 62 ms 是谁吃掉的（三个嫌疑，尚未拆分）

1. **496 个专家下放主机**（8.68 GiB，coverage 93.54%）——每步路由命中就要跨 PCIe 读，且有 host LRU 7.7 GiB 兜底，miss 部分才读盘。
2. **MTP 3 层 draft forward**——固定 7.39 GiB 显存 + 每步 3 次小 forward。
3. **verify 时主模型一次 forward 4 个 token**——权重读取不变但 compute 变 4 倍。

旧引擎这 3 项全为零，所以 23 ms。这是"开投机必须交的税"，不是"平台更先进"。

### 8.3 被忽略的真事实：瓶颈是每步固定延迟，不是带宽、不是 engram

| 模式 | 单流 decode | 三路并发聚合 | 聚合 / 单流 |
|---|---:|---:|---:|
| ram | 43.18 | 43.92 | **1.02×** |
| dspark | 36.47 | 32.01~42.02 | **0.88~1.15×** |

**并发三路，聚合吞吐几乎不涨**（ram 只涨 2%，dspark 甚至下跌）。
如果瓶颈是权重读取带宽，3 路并发会把同一批权重摊给 3 个 token，聚合应该接近 3×。
实际不涨 → 瓶颈是**每步的固定时间**：PP5 五段串行，每段约 4.6 ms，合计 23 ms，
与 batch 里放几个请求基本无关。

这条推论同时解释了最初截图里"多路并发互相卡住、4.9 tok/s"：
并发不是把带宽分薄了，而是**排队等同一个串行流水线**，客户端看到的 tok/s 被摊成 1/N。

**下一步最大的机会不在移植别人的开关，而在让 PP 流水真正并行**（microbatch / 1F1B）。
如果 5 段能重叠，单步 23 ms 有希望压到 8~12 ms，那才是 80~120 tok/s 的量级。

## 9. 移植路线（用户已同意改引擎，前提：模型必须是原版满血 1017）

约束：不动 checkpoint、不动量化精度、不动权重；只改调度/kernel/内存布局。

| 优先级 | 项目 | 预期收益 | 工作量 | 验证方式 |
|---|---|---|---:|---|
| **P0** | 新引擎 + 专家 100% 在卡 + **关 dspark** | 量出平台本底（16 还是 28 tok/s），决定后续方向 | 半天（一次启动） | C1 单流 decode |
| **P1** | **PP microbatch / 流水重叠**（自己改 `gpu_worker` 调度） | 最大：步时 23→10 ms 量级，但难度最高 | 1~2 周 | C1 + C3 扩展比 |
| **P2** | PP_KEEP_BATCH 移植（空步不踢请求） | 长上下文 +10~25%，配合 chunked prefill | 3~5 天 | C2 TTFT、C3 聚合 |
| **P3** | route bias / 专家亲和路由 | 仅在"必须下放专家"时有 +25%；专家全在卡则无效 | 1 周 | 下放模式的 C1 |
| **P4** | KV FP4 + 热环（省 KV 显存换并发/上下文） | 每步 −1 ms 量级，主要是省显存 | 1~2 周 | 524K 上下文 + C3 |
| ✗ | hugetlb / SSD read-ahead | ram 模式 engram I/O 已归零，无对象（**此前判断需修正**：大页对 189 GiB pinned 表的 TLB 有理论收益，但 decode 每步只读 34 KB，实测空间很小） | — | — |
| ✗ | MRV2 单独开 | 已含在 P0 里一起量 | — | — |
