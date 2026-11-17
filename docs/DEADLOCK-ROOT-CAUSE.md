# 死机根因：NVIDIA open 驱动 #968 族死锁（2026-11-17 锁定）

> 历史文档 [GPU4-HANG.md](GPU4-HANG.md) 记录了取证过程与"GPU4 嫌疑"；本文档是最终归因。
> 结论：**死机不是卡、不是槽、不是 PCIe 链路，是 nvidia-open 内核模块的已知死锁 bug 族**
> （上游 issue #968 / #922 / #979，截至 2026-11 open 未修）。

## 1. 结论

**机理**：进程 A open GPU 设备（`nv_open_q`，blocked at `rwsem_down_write_slowpath`）与
进程 B close/shutdown GPU（需同一把 `ldata_lock`）互等 → 双方进入不可中断睡眠（D 状态）
→ 整机冻结，无 panic、无 Xid、journal 戛然而止，唯一恢复手段 = 断电。

上游报告原文（#968）与我们 11-16 17:00 的内核栈逐字吻合：

```
Process 1: nv_open_q (trying to OPEN/initialize GPU)
  → blocked at rwsem_down_write_slowpath (write lock)
Process 2: python3 (trying to CLOSE/shutdown GPU)
Both in state D — machine completely stalls, not even keyboard input.
```

驱动 535.x–595.x+ 全线中招（我们是 610.43.03 open + cmpunlocker 解锁补丁）。

## 2. 证据链

| # | 证据 | 吻合点 |
|---|---|---|
| 1 | 11-16 17:00 死机栈：`nv_open_q` + `rwsem` + `kflcnWaitForHalt` | #968 的直接现场 |
| 2 | 猝死型死机：零内核输出、journal 戛然而止、断电恢复 | #968 表型 |
| 3 | 死机跟随"事件"（ComfyUI 大块显存释放 × 引擎请求并发），不跟随卡也不跟随槽（对调实验 11-17：嫌疑卡/09 槽双双洗清后仍死） | 驱动级死锁与硬件无关 |
| 4 | 死机时序：两次都在 ComfyUI 任务**完成卸载后 2-6 秒**（blackbox：GPU0 功率 242W→77W 卸载瞬间） | 卸载（RM free / close）与引擎活动并发的窗口 |
| 5 | **竞态燃料**：blackbox 1Hz + fastwatch 5Hz 的 nvidia-smi 子进程循环 = 每秒 1-5 次完整设备 **open→close**（`-lms 200` 循环内重走 NVML init/deinit），自 11-15 起常驻生产 | 高频 open/close 把 #968 的概率窗口放大到 ~6h 中招一次 |
| 6 | 引擎在 **Docker 容器**（open 模块对容器 O_NONBLOCK open 有 deferred 路径） | #968 报告的触发场景 |
| 7 | 今早 10:00：journald 比 blackbox 早断 4 秒 | rwsem 死锁后内核日志通道先卡，用户态文件写多活几秒 |

**反证排除**：
- 对调实验（02:00.0 ↔ 09:00.0 物理对调）：单卡致命配置两卡两槽全存活、死亡场景 5/5 存活、
  性能无回退（51.03 vs 50.43）——卡/槽无问题。
- AER 全 0、带宽五卡一致 → 链路质量排除。
- 53 次 Booter 失败（`Booter failed 0x31`，五卡每 boot 齐刷刷 53 次）= **cmpunlocker 解锁补丁的
  PLM pass 正常现象**（exploit payload 故意让 Booter 拒绝，判成功只看 PLM 寄存器回读；
  上游 troubleshooting.md 明说 0x31/0xffff harmless）。不是病。
- 科脑 X99 无 BMC（dmidecode 全表无 IPMI 记录，hostname 冒充超微型号）→ SEL/IPMI 路线不存在。
- runtime PM `power/control=on`（suspend 路径不可达）→ #968 走的是 open×close 变体，非 suspend 变体。

## 3. 已上缓解（2026-11-17，零成本无需重启）

1. **persistence mode ON**（五卡 `nvidia-smi -pm 1`，已写入
   `/etc/systemd/system/nvidia-170hx-power-limit.service.d/gpu0-250w.conf` boot 持久化）
   ——保持设备引用计数，缩小 open/close 竞态窗口（上游社区第一层缓解）。
2. **fastwatch 常停**（`systemctl stop dsv41-fastwatch`）——5Hz nvidia-smi 子进程循环是竞态
   燃料；死机取证改用 blackbox 1Hz。
3. **编排错峰建议**：Claude 编排提交放大任务后纯等待（不发起引擎推理），完成后再等 30-60s。

**验证判据**：缓解上线后死机频率显著下降/消失 = 根因确认。若仍死机 → 竞态还有其他源，回到取证。

## 4. 后续方向

1. **监控改造（正道）**：blackbox/fastwatch 改用 **pynvml 常驻连接**（NVML init 一次长连接，
   无 open/close 循环）——既保留监控又根除燃料。
2. **硬修复候选**：[gpuGUARD](https://github.com/cyclecore-dev/gpuGUARD) 的
   `patches/002-fix-open_q-flush-deadlock-comprehensive.patch`（把 open_q flush 移到
   ldata_lock 之前）——580.82.09 版本，移植到 610.43.03+cmpunlocker 需要分析其补丁基线。
3. **上游跟踪**：#968/#922/#979 open 状态，新驱动版本 release note 关注 deadlock 修复。
4. **智能插座兜底**（独立于根因）：`tools/dsv41_watchdog.ps1` 骨架已写，等插座品牌接 API——
   死机 2-3 分钟自动恢复（BIOS Restore-on-AC 已验证）。

## 5. 遗留未解

- 每次开机 GPU1=110W、GPU4=250W 的来源（journal/配置/crontab 全查无；嫌疑用户侧自动化栈
  pynvml 或监控插件）——已手动修正功耗墙 250/150/150/150/150。
