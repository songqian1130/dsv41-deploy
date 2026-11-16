# GPU4 整机冻结：定位结论与绕行方案

> 2026-11-16 定稿。这是本仓库所有方案的前提条件 —— 不解决它，任何"5 卡满载"都可能在 30–90 秒内断电重启。

## 1. 现象

- ComfyUI（H3 视频）+ 引擎同时跑，**整机冻结**：无响应、SSH 断、无 Xid、无 kernel panic，只能断电。
- 单独跑任何一个都不冻。纯 ComfyUI 跑 8 小时没事；引擎单独跑 5 卡也没事。
- 冻结点不固定：ComfyUI 加载权重时、第一个采样步、或引擎 prefill 时都可能触发，通常 30–90 秒内。

### 1.1 「无日志」是错的 —— 2026-11-16 17:00 事件首次拿到内核栈

之前所有死机都记录为"无 Xid、无日志"。**错**。journal 是异步落盘的，死锁如果持续几分钟，
journald 来得及写盘；死机后重启，`sudo journalctl -b -1` 就能拿到完整内核栈。
之前"无日志"只是断电太快没刷出来。

> **新规矩：每次死机重启后，第一条命令是 `sudo journalctl -b -1 | tail -200`，再干别的。**

死锁链条（`-b -1` 16:59:50 内核原文）：

```
INFO: task reboot:827 blocked on an rw-semaphore likely owned by task nv_open_q:735
task:nv_open_q  state:R running  ppid:2(kthreadd)
RIP: osDevReadReg032+0x3c/0x70 [nvidia]        ← 死循环读寄存器
  _regRead → kflcnWaitForHalt_TU102            ← 等 GSP falcon 停机，等不到
  kgspExecuteHsFalcon → s_executeBooterUcode
  kgspExecuteBooterLoad_TU102 → kgspInitRm_IMPL
  RmInitAdapter → rm_init_adapter              ← 某张卡正在（重新）open
  nv_start_device → nv_open_device → nvidia_open_deferred
```

**「整机失联」的真面目不是 CPU 卡死，而是 nvidia 驱动全局 rwsem 死锁**：一个设备 open
卡在 GSP falcon 引导等待里持锁不放，`reboot`、`nvidia-smi`、一切 CUDA 调用全部在锁上排队。
网络栈还活着（所以 SSH 短暂 `Connection refused`），随后彻底无响应。

### 1.2 2026-11-16 17:00 事件完整时间线

| 时刻 | 事件 | 来源 |
|---|---|---|
| 16:52:50 | 8192³ fp16 算力测试（四卡交错两遍）成功返回 | `-b -2` sudo 日志 |
| **16:53:34** | **卡间 P2P 测试启动（32MB×40 次/对），日志戛然而止 = 第一次死机** | `-b -2` 末尾 |
| 16:54:08 | 重新开机（`-b -1`） | journal |
| 16:54:33 | **开机仅 25 秒**：`nvidia-smi` 被 `(udev-worker)` 持有的 rwsem 卡住，每 10s 报一次 | `-b -1` hung task |
| 16:59:50 | `nv_open_q` 卡死在 `kflcnWaitForHalt`（见上），`reboot` 进程被锁阻塞 | `-b -1` |
| ~17:00 | 彻底失联（ping 100% 丢包） | — |
| 17:02:39 | 用户手动重启（`-b 0`），五卡各重试 53 次 Booter 后正常 | journal |

要点：

- 第一次死机（P2P）是**猝死**：无 hung task 预警、无 Xid、无任何内核消息，日志直接断。
- `-b -1` 开机即病态（25 秒就 hung），是**上次死机残留的 GPU 固件状态**所致，与
  「硬冻机重启后 NCCL CUDA 800 需 `rmmod nvidia_uvm`」是同一类残留。
- `-b -1` 里一条 `Booter failed` 都没有——它卡在 booter **执行中**（`kflcnWaitForHalt`
  死循环），连"失败重试"分支都没走到，比"失败"更糟。

### 1.3 GSP Booter 重试 53 次是这台机器的常态

| boot | 开机时刻 | Booter failed | 分布 | 结果 |
|---|---|---:|---|---|
| `-b -2` | 11:08:31 | 265 次 | 开机 20 秒内集中出现 | 五卡各 53 次重试后全部成功，之后 5h45m 运行稳定 |
| `-b -1` | 16:54:08 | **0 次** | — | 卡死在执行中，25 秒后 hung |
| `-b 0` | 17:02:39 | 265 次 | 开机 20 秒内集中出现 | 五卡各 53 次重试后成功，正常 |

**五张卡都有**（GPU0-4 各 53 次），不是 GPU4 独有。看日志时别把开机重试当成故障信号。

### 1.4 AER 干净 —— 排除 PCIe 链路质量问题

17:03 干净启动后查 `lspci -vvv`（GPU4 `09:00.0` vs 对照 GPU3 `05:00.0`）：
DevSta/UESta/CESta 的 Correctable、Uncorrectable、BadTLP、BadDLLP、Replay 计数
**全部为 0**。链路层没有累积错误 →「GPU4 链路有重传/纠错开销」的假说不成立，
4KB 延迟 +3.8% 的来源另有其因（或为测量噪声）。

### 1.5 判据修正：引擎后台也算 kernel，P2P 双向 DMA 更烈

P2P 测试脚本本身只有 `copy_`（32MB device→device × 40 次/对）+ 瞬时 `fill_`，
**没有持续 matmul**，按旧判据"不该冻"——但它死了。修正：

1. **机器不是空载**：引擎容器（GPU1-4 各 61GB、4 个 Worker_PP 常驻自旋 11.6 核）
   一直在跑，V2 runner idle 也有后台活动。P2P 的 DMA 与引擎后台活动重叠，
   仍然构成"搬运 + 计算"。
2. **P2P 比单卡 H2D 更烈**：这台机器五卡全是 PHB（无 switch），device→device 的
   DMA 要经 root complex + IOMMU（`intel_iommu=on`）转发 = **两张卡同时双向大流量**，
   每秒数 GB 穿根复合体。之前单卡 H2D 1.5 GB/s 单向能存活 150s，P2P 30 秒内猝死。
3. 结论修正为：**"跨卡大块 P2P DMA"（即使无显式 kernel）与引擎共存即致命**。
   禁止项从「GPU4 上大块 H2D + kernel」扩大为「GPU4 在场时任何卡间大块 P2P」。

## 2. 根因判据（合成负载复现，不依赖模型）

复现包在本仓库 [`bench/hang/`](../bench/hang/)（三个文件，`run-lethal.sh` 一把梭；服务器上另有一份在 `~/manju/`）。
它只做两件事：造"PCIe 大块搬运"和"GPU 计算"，然后把两者叠在同一张卡上。

单个开关的含义：

| 开关 | 行为 |
|---|---|
| `--stage GB/s` | 文件 → pinned 内存 → 显存，连续 H2D 大块搬运 + `cudaMalloc` 抖动 + 显存常驻 |
| `--sm` | GPU matmul，持续占用 SM |
| `--dram GB/s` | N 线程 float32 求和，压主机内存带宽 |
| `--stage-chunk-mb` / `--stage-reserve-mb` | 搬运粒度 / 显存常驻量 |

**结论：只要"PCIe host→device 大块搬运"与"GPU kernel 执行"在时间上重叠，GPU4 必冻。**

变量逐个排除过，都不改变结论：

| 变量 | 试过 | 结果 |
|---|---|---|
| 数据来源 | 磁盘读 / 内存生成 | 都冻 |
| 搬运粒度 | 64 MB / 4 MB | 都冻 |
| 主机带宽 | 10 GB/s / 44 GB/s（接近打满 55 GB/s） | 都冻 |
| 主机内存压力 | `--dram` 加到满载 | 不影响（不是内存问题） |
| 驱动层 | VFIO 直通给虚拟机（客户机内 64 GB 解锁完好、官方 `nvidia-driver-610-open`） | **客户机内同样冻死** → 排除驱动/软件层 |

## 3. 逐卡定位：只有 GPU4

[`bench/hang/sweep_cards.sh`](../bench/hang/sweep_cards.sh) 对每张卡单独跑同一套"搬运 + 计算"并发负载，各 150 秒：

| 卡 | PCI 地址 | 根端口 | 结果 |
|---|---|---|---|
| GPU0 | `0000:02:00.0` | 共享 | ✅ 存活 |
| GPU1 | `0000:03:00.0` | 共享 | ✅ 存活 |
| GPU2 | `0000:04:00.0` | 共享 | ✅ 存活 |
| GPU3 | `0000:05:00.0` | 共享 | ✅ 存活 |
| **GPU4** | **`0000:09:00.0`** | **独立 `00:03.3`** | ❌ 30–90 s 冻结，复现 5 次 |

GPU4 是唯一挂在独立根端口上的卡，其余四张共享同一个 PCIe switch。
**这是硬件/拓扑层面的问题，不是配置问题。**

## 4. 绕行方案（当前生产配置）

| 组件 | 配置 |
|---|---|
| ComfyUI | `CUDA_VISIBLE_DEVICES=3`（systemd drop-in `comfyui.service.d/zz-gpu3.conf`） |
| 引擎 | `DSV41_GPUS=0,1,2,4`（方案 L），GPU4 只承载 PP3 的小包流量 |
| GPU3 功耗墙 | **250 W**（drop-in `nvidia-170hx-power-limit.service.d/gpu3-250w.conf`） |

验证：引擎 6 路满载 + H3 124 帧，全程 22 次采样服务全 200，机器存活，视频 130.2 s。

### 功耗墙这个坑（2026-11-16 踩过）

`nvidia-170hx-power-limit.service` 把所有卡统一设 150 W，只给 GPU4 豁免 250 W。
ComfyUI 从 GPU4 搬到 GPU3 后直接掉进 150 W 的坑：

| 卡 | 功耗墙 | H3 124 帧 | SM 频率中位数 |
|---|---:|---:|---:|
| GPU4 | 250 W | 125.4 s | — |
| GPU3 | 150 W | 184.6 s | 1035 MHz |
| GPU3 | **250 W** | **128.3 s** | 1260 MHz |

**GPU 利用率 100% 但变慢 = 降频，不是排队。** 以后再看到"并发把视频拖慢了"，先查 `nvidia-smi --query-gpu=power.limit,clocks.sm`。

## 5. 待办

- [ ] **物理对调 GPU4 与 GPU0 的插槽**，确认问题跟卡走还是跟插槽/根端口走。
      跟卡走 → 换卡根治；跟插槽走 → 该槽永久禁用或换主板。
- [ ] 在确认前，任何方案都不得让 GPU4 承担"大块 H2D + kernel 并发"（即不要让 ComfyUI 回到 GPU4）。
      **且禁止在引擎运行时做任何卡间大块 P2P**（见 1.5，2026-11-16 已实证）。
- [x] ~~查 AER 可纠正错误计数~~（2026-11-16 已查，全 0，见 1.4，链路假说排除）
- [x] ~~拿到死机内核栈~~（2026-11-16 首次拿到，见 1.1——死锁在 GSP falcon 等待 + 全局 rwsem）
- [ ] **每次死机重启后先 `sudo journalctl -b -1 | tail -200`**，把栈抄进本文档，积累样本。
      关注点：`nv_open_q`/`RmInitAdapter` 持锁的是哪张卡的 open（日志里未必写卡号，
      可配合 `dmesg` 时间戳前最近的 `NVRM` 消息定位）。
- [ ] GPU3 250 W 下满载温度 86 °C（单跑）/ 68 °C（双跑），夏天前复测一次散热。
