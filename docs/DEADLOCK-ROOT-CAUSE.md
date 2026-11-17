# 死机根因调查：从 #968 软死锁假说 → 硬冻结（2026-11-17 20:40 修订）

> 历史文档 [GPU4-HANG.md](GPU4-HANG.md) 记录了取证过程与"GPU4 嫌疑"；本文档是根因调查主线。
> **状态修订（19:12 死机 + 观测阵列 v2 首战）**：#968 软死锁**降级**为次要假说——
> 5 次死机中 4 次为"零内核输出瞬死"，死亡深到 NMI 都没来得及打印，指向平台级硬楔死。

## 0. 五次死机签名总表（2026-11-17 20:40）

| # | 时间 | 死亡形态 | 内核输出 | 当时负载 |
|---|---|---|---|---|
| 1 | 11-16 16:53 | 瞬死 | 0（P2P 实验触发） | P2P 大块拷贝 × matmul |
| 2 | 11-16 17:00 | **rwsem 栈可见**（唯一一次） | hung-task dump：nv_open_q blocked | 引擎+ComfyUI |
| 3 | 11-17 06:11 | 瞬死 | 0 | SeedVR2 放大 × 引擎补货 |
| 4 | 11-17 15:14 | 瞬死 | 0（journald 先断 6s） | 引擎 3 路 decode（ComfyUI 空闲） |
| 5 | 11-17 19:12:11 | 瞬死 | **0（29 分钟零输出后瞬死）** | 引擎 2 路 decode+补货 × ComfyUI 装载 VAE |

## 1. 原假说（#968 软死锁）及降级理由

**机理（保留备查）**：进程 A open GPU 设备（`nv_open_q`，blocked at `rwsem_down_write_slowpath`）与
进程 B close/shutdown GPU（需同一把 `ldata_lock`）互等 → 双方 D 状态 → 整机冻结。
上游报告原文（#968）与 11-16 17:00 的内核栈逐字吻合：

```
Process 1: nv_open_q (trying to OPEN/initialize GPU)
  → blocked at rwsem_down_write_slowpath (write lock)
Process 2: python3 (trying to CLOSE/shutdown GPU)
Both in state D — machine completely stalls, not even keyboard input.
```

驱动 535.x–595.x+ 全线中招（我们是 610.43.03 open + cmpunlocker 解锁补丁）。

**降级理由（19:12 死机，观测阵列 v2 完整观测下发生）**：
- 死前 29 分钟内核零输出（kmsg.log 与 netconsole 双向确认），死亡瞬间一条 printk 都没有
- hung-task 10s 已武装且 warnings 无限——若任务 D 卡 >10s 必然刷屏，实际零输出 →
  死前没有任何任务阻塞超过 10 秒，不符合"互等死锁逐步堆死"的形态
- nmi_watchdog=1 已武装——CPU 关中断自旋 >阈值会打 "hard lockup"，实际零输出
- probe.log（bpftrace v2）死亡时刻 open/close 全配对、无慢锁——设备 open/close 路径干净
- **能解释的残余角色**：11-16 17:00 那次栈可能是楔死边缘的 OS 可见投影，或 5 次里唯一的
  软死锁变体。缓解措施（persistence/停 fastwatch）保留。

## 1.5 现主假说：平台级硬楔死（X99 PCIe fabric / SMI 风暴）

支持证据：
- 死亡深于 NMI 打印：SMI（系统管理中断，BIOS/ME 发出）优先级高于 NMI，SMI 死循环可
  完全饿死内核打印路径 → **零输出瞬死**的优雅解释
- **"perf: interrupt took too long" 整小时持续恶化**（17:43→18:43，PMI 处理耗时
  2.5k→10k 周期，采样率 79k→20k Hz 一路降）——PMI 被某物偷时间的经典签名，SMI 嫌疑最大
- 死机总在引擎专家补货 DMA / ComfyUI 大上传并发的 PCIe 高压窗口（5/5）
- 死前最后一个 blackbox 采样：单引擎卡 SM 频率骤降 + SW Power Cap 限频（2/2 有此签名，
  但也可能只是正常功耗墙限频，弱信号）
- AER 全 0、MCE 全 0（rasdaemon 已跑满整个 boot -1）——链路质量和 CPU/内存无可见异常

## 2. 证据链（原 #968 分析，保留）

| # | 证据 | 吻合点 |
|---|---|---|
| 1 | 11-16 17:00 死机栈：`nv_open_q` + `rwsem` + `kflcnWaitForHalt` | #968 的直接现场 |
| 2 | 猝死型死机：零内核输出、journal 戛然而止、断电恢复 | #968 表型 |
| 3 | 死机跟随"事件"（ComfyUI 大块显存释放 × 引擎请求并发），不跟随卡也不跟随槽（对调实验 11-17：嫌疑卡/09 槽双双洗清后仍死） | 驱动级死锁与硬件无关 |
| 4 | 死机时序：两次都在 ComfyUI 任务**完成卸载后 2-6 秒**（blackbox：GPU0 功率 242W→77W 卸载瞬间） | 卸载（RM free / close）与引擎活动并发的窗口 |
| 5 | **竞态燃料**：blackbox 1Hz + fastwatch 5Hz 的 nvidia-smi 子进程循环 = 每秒 1-5 次完整设备 **open→close**（`-lms 200` 循环内重走 NVML init/deinit），自 11-15 起常驻生产 | 高频 open/close 把 #968 的概率窗口放大到 ~6h 中招一次 |
| 6 | 引擎在 **Docker 容器**（open 模块对容器 O_NONBLOCK open 有 deferred 路径） | #968 报告的触发场景 |
| 7 | 今早 10:00：journald 比 blackbox 早断 4 秒 | rwsem 死锁后内核日志通道先卡，用户态文件写多活几秒 |
| 8 | **缓解后仍死（11-17 15:14，persistence ON + fastwatch OFF 下 2.75h）**：ComfyUI 完全空闲（0% util 无卸载）、引擎 3 路大 KV decode（KV 25.3%）——**形态剧变，"卸载触发"被推翻**；且 14:50:06 内核记录 **MCE（Machine check）硬件错误**，死机前 25 分钟 | fastwatch 燃料非必要条件；剩余 open/close 源仅 blackbox 1Hz；**硬件/平台嫌疑上升**（MCE 详情因未装 mcelog/rasdaemon 已丢失，现已布防） |

**决定性对照实验（16:23 起）**：blackbox 改为 **pynvml 常驻连接版**（`tools/dsv41-blackbox-pynvml.py`，
一次 nvmlInit，无 open/close 循环，列格式兼容）——**系统内 nvidia-smi 子进程循环归零**。
- 之后死机消失/大降 → open/close 燃料假说实锤，根因 = #968 × 我们的监控实现
- 之后照死（rasdaemon 会记录下次 MCE/AER 详情）→ 燃料假说终结，转向硬件/平台方向
  （内存控制器/供电/CPU——X99 魔改板 + 5 卡压力）

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
   无 open/close 循环）——既保留监控又根除燃料。blackbox 已改 pynvml 版（11-17 16:23）。
2. **硬修复候选**：[gpuGUARD](https://github.com/cyclecore-dev/gpuGUARD) 的
   `patches/002-fix-open_q-flush-deadlock-comprehensive.patch`（把 open_q flush 移到
   ldata_lock 之前）——580.82.09 版本，移植到 610.43.03+cmpunlocker 需要分析其补丁基线。
3. **上游跟踪**：#968/#922/#979 open 状态，新驱动版本 release note 关注 deadlock 修复。
4. **智能插座兜底**（独立于根因）：`tools/dsv41_watchdog.ps1` 骨架已写，等插座品牌接 API——
   死机 2-3 分钟自动恢复（BIOS Restore-on-AC 已验证）。

## 5. 观测阵列 v2（2026-11-17 17:30 上线——死机瞬间直接定位卡死函数/锁）

四层独立取证，死机后无需猜：

| 层 | 组件 | 捕捉 | 判读 |
|---|---|---|---|
| 1 | **内核 hung-task**（已配：timeout 10s / warnings 无限 / panic 关 / all_cpu_backtrace=1 / `DETECT_HUNG_TASK_BLOCKER=y`） | D 状态任务卡 >10s 自动打印**阻塞栈+锁持有者** | 死机后看内核日志尾部"INFO: task ... blocked"——**卡死函数+持锁进程直接可读** |
| 2 | **netconsole**（`dsv41-netconsole.service`：UDP 192.168.3.28:6667，printk 级提到 7） | 全部内核 printk 实时外发，journald/磁盘死了照样发 | Windows 侧接收器落盘 `dsv41-deploy/logs/netconsole-*.log`（端到端已验证） |
| 3 | **bpftrace 探测器 v2**（`dsv41-probe.service`，脚本 `tools/dsv41-probe.bt`，逐行 fsync 落盘 `/var/log/dsv41-blackbox/probe.log`） | nv_open_device / nvidia_close / nv_close_device 全记录；rwsem/semaphore 慢路径 >300ms（含锁地址）；GPU 进程 down() 入口 | **未配对的 BEGIN = 持有者现场**（最后一条 OPEN/CLOSE-BEGIN 没有对应 END）；RWS-WAIT/SEM-WAIT 行 = 等待者与哪把锁 |
| 4 | **NVML 金丝雀**（`dsv41-tripwire.service`，`tools/dsv41-tripwire.py`） | pynvml init 每 10s 探测（0.1Hz 最小燃料）；卡住 >20s = 死锁判据 → **自动 sysrq w/l/t 三连 dump** | w=阻塞任务栈 / l=每 CPU 当前函数（自旋场景）/ t=全任务调度状态——dump 走 printk → netconsole 外发 |

死机后判读动线：Windows 收 netconsole 日志（活的最后瞬间）→ 服务器 `probe.log` 尾部（持有者）
→ hung-task dump（等待者+卡死函数）→ rasdaemon（MCE/AER 硬件事件）→ blackbox pynvml（GPU 曲线）。

**部署细节**：bpftrace v0.25 不支持 `=~` 正则谓词（用相等链）；`delete(@map[key])` 旧语法仍在
但建议 `delete(@map, key)`；Windows 接收器 `tools/netconsole_listener.py`（UDP 6667，已加防火墙
规则 dsv41-netconsole-udp6667；**开机自启已注册**（任务 `dsv41-netconsole-listener`，LogonTrigger
→ pythonw + netconsole_listener.py，2026-11-17 用户手动注册，XML 已核验）。

## 6. 遗留未解

- 每次开机 GPU1=110W、GPU4=250W 的来源（journal/配置/crontab 全查无；嫌疑用户侧自动化栈
  pynvml 或监控插件）——已手动修正功耗墙 250/150/150/150/150。
