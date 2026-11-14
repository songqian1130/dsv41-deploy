# PP5 部署模式手册

> 四种已验证的启动方案固化在脚本里，一条命令切换；systemd 看门狗会跟随当前模式自愈。

## 1. 切换方式

```bash
sudo /usr/local/bin/dsv41-pp5-ctl.sh use fast       # 切到 fast 模式（当前生产，最快）
sudo /usr/local/bin/dsv41-pp5-ctl.sh use ram        # 切到 ram 模式（旧生产，兜底用）
/usr/local/bin/dsv41-pp5-ctl.sh status              # 看模式/容器/健康/内存/显存
/usr/local/bin/dsv41-pp5-ctl.sh mode                # 只打印当前模式
```

- 当前模式记录在 `/etc/dsv41-pp5/mode`，重启机器后 `dsv41-pp5.service` 按它拉起。
- 看门狗 `dsv41-pp5-health.timer`（每 3 分钟）也用同一份模式文件，所以自愈拉起的永远是同一模式。
- 切换过程会：停其它模式容器 → 等显存释放 → **等 pinned 内存归还到 ≥190 GB** → 启动新容器。
  加载期 5~12 分钟，期间 `/health` 不通属正常，看门狗在预热期内不会插手。

## 2. 模式对照表

| 模式 | 容器名 | 启动脚本 | 引擎 | engram | 专家驻留 | 单流 decode | 22k TTFT | 启动 |
|---|---|---|---|---|---|---|---:|---:|---:|
| **`fast`（当前生产）** | `dsv41-pp5-dspthr` | `run-pp5-dspthr.sh` | V2 + **dspark k=5** + 20 层专家在 CPU + **decode 线程 4** | SSD 4 GiB 缓存 | 93.96% 在卡 / 7.7 GiB 在主机 | **100.1（代码）/ 85.4（JSON）/ 65.2（中文）/ 79.1（数学）** | 10.2 s | ~9 min |
| `dspark` | `dsv41-pp5-dspark` | `run-pp5-dspark.sh` | V2 + dspark k=3 + 20 层专家在 CPU（decode 线程 28） | SSD 4 GiB 缓存 | 93.5% 在卡 / 7.7 GiB 在主机 | 51.8（代码）/ 38（开放文本） | 8.6~10.5 s | ~9 min |
| `ram` | `dsv41-pp5-ram` | `run-pp5-ram.sh` | V1 + 惰性接收 | **全放 pinned 内存 189 GiB** | 100% 在卡 | 43.2 tok/s | **6.45 s** | ~11 min |
| `legacy` | `dsv41-pp5-legacy` | `run-pp5-legacy.sh` | V1 + 惰性接收 | SSD 8 GiB 缓存 | 100% 在卡 | ≈43 tok/s（同引擎，按 ram 推断） | ~11.9 s（旧口径，仅参考） | ~4.5 min |

**`fast` 与 `dspark` 只差一个参数**：`cpu_threads: [28,4]` 而不是 `[28,28]`。decode 线程从 28 降到 4（物理核 28，五个 rank 各 28 线程会超订 5 倍），
CPU 占用 2721% → 827%，ITL 61.7 ms → 36~40 ms，**全部用例提速 45~86%**。详见 `PP5-DSPARK-STUDY.md` §7。

**2026-11-14 17:45 生产又加一档：dspark k 从 3 提到 5**（`run-pp5-dspthr.sh`，备份 `run-pp5-dspthr.sh.bak-k3`）。
依据是 `hybrid.py:149` 里 `dspark_num_query_tokens` 的下限本来就是 5 —— k=3 没用满这个 block。
三轮固定用例（seed 51/52/53）均值：

| 用例 | k=3（fast 旧） | **k=5（现生产）** | 变化 |
|---|---:|---:|---:|
| code_python | ~89 | **100.1** | +12.5% |
| json_struct | ~77.5 | **85.4** | +10.2% |
| zh_tech | ~61.5 | **65.2** | +6.0% |
| math_steps | ~72 | **79.1** | +9.9% |

步时从 38~39 ms 涨到 41~46 ms，但每步 token 从 3.4 涨到 4.2，净赚约 10%。
**显存零成本**：专家覆盖率仍是 93.96%（7216/7680），与 k=3 逐字相同。
回滚：`cp run-pp5-dspthr.sh.bak-k3 run-pp5-dspthr.sh && dsv41-pp5-ctl.sh restart`。

## 3. 各模式的硬约束（踩过的坑，别再试）

**① dspark 与 engram=ram 互斥（已实测报错）**
新引擎 `host_memory.check_host_headroom()` 把 `64 GiB` 可用内存硬编码为下限，没有任何环境变量可覆盖：

```
RuntimeError: DeepSeek hybrid host allocation stopped before exhausting RAM:
64 GiB of available memory must remain
```

ram 模式每 rank 锁 94.42 GiB、两 rank 共 189 GiB，剩 39~48 GiB，必被拒。所以 dspark 只能配 SSD。

**② engram_cache_gib 只能配 ssd**
`hybrid.py:94`：`engram_cache_gib requires engram_storage='ssd'`。配 ram 时要把它删掉。

**③ dspark 需要 V2 runner，惰性接收需要 V1，两者互斥**
`gpu_worker.py` 里 `assert not self.use_v2_model_runner`。开 dspark 必须去掉 `VLLM_PP5_LAZY_RECEIVE=1` 和 `VLLM_USE_V2_MODEL_RUNNER=0`。

**④ dspark 的 MTP 层数与 k 无关，固定 7.39 GiB**
`dspark.py:83`：`num_dspark_layers = n_mtp_layers or num_nextn_predict_layers or 3`，不看 `num_speculative_tokens`。
k=1 省不了显存（实测 k=1 直接 OOM），所以旧引擎（只剩 1.93 GiB KV 余量）无法加 dspark。

**⑤ 5 卡上 dspark 必须"下放专家"换空间**
新引擎日志：`coverage=93.54% (7184/7680, 125.788 GiB)`、`host LRU 7.704 GiB` —— 为了 7.39 GiB 的 dspark，它把 496 个专家挪到主机。
这是 dspark 在 5 卡上的真实代价。

**⑥ 新引擎 init 时的内存预算是硬账，engram 缓存加不上去**
`hybrid_runtime.py:190-213` 在启动时按下面公式检查，超了就直接拒绝启动（`check_host_headroom` 里 64 GiB 写死）：

```
resident  = (num_hidden_layers - 20) × 384 专家 × expert_bytes   ≈ 134 GiB（20 层专家的 MXFP4 权重）
          + engram 缓存（ssd 模式 = engram_cache_gib；ram 模式 = 整张表 189 GiB）
check_host_headroom(resident + host_cache_gib(8) + 16 GiB)   要求 ≥ 64 GiB 仍然可用
```

- `engram_cache_gib=4`（prod 原值）：134+4+8+16+64 = **226 GiB**，247 总量下勉强通过。
- `engram_cache_gib=48`：**270 GiB**，实测直接报 `64 GiB of available memory must remain`。
- ram 模式：134+189+8+16+64 = **411 GiB**，永远不可能。

所以 dspark 模式的 engram 缓存**无法调大**，TTFT 也就无法改善——这是它与 `ram` 模式的根本差距。

## 4. 性能实测（口径见 `PP5-BASELINE.md`）

固定用例 `cases_v1`（`bench/cases_v1.json`，sha256 `ee98f898…`），服务端计数器口径。

| 模式 | 单流 decode | 每步 ITL | C2 22k TTFT | C3 三路聚合 | C3 每流 |
|---|---|---:|---:|---:|---:|
| **`fast`（decode 线程 4）** | **85.2~92.9（代码）/ 78.6~79.3（JSON）/ 61.7~62.1（中文）/ 67.7~73.6（数学）** | **36.0~40.4 ms** | 10.20 s | **77.94 tok/s** | **25.98** |
| `dspark`（decode 线程 28） | 54.9~58.7（代码）/ 47.6~48.4（JSON）/ 42.5（中文）/ 40.5（数学） | 61.7~65.0 ms | 8.63~10.51 s | 32.01~42.02 | 15.67~21.0 |
| `ram` | 43.18 tok/s（四类负载均 43.4~43.6） | 23.2 ms | **6.45 s** | 43.92 tok/s | 21.65 |

**`fast` 全面超过 `ram` 和 `dspark`**：单流 45~115%，三路聚合 77.94 对 43.92（+77%）。
唯一仍落后的是 22k 冷启 TTFT（10.20 s 对 6.45 s），因为 engram 还在 SSD 上——这是 dspark 与 `engram_storage=ram` 互斥导致的，不是线程问题。

`speed-up 的来源`（2026-11-14 15:00 定案，详见 `PP5-DSPARK-STUDY.md` §7）：

- dspark 让**每步产出 3.5 个 token**（旧引擎 1 个），这是提速的唯一来源；
- 但 `cpu_threads` 的 decode 档配成 28（= 物理核数），而 decode 时**只有 PP4 一个池**在跑（下放层 20~39 全在 PP4），
  28 个线程把 27 核占满，CPU 2721%——但其中真实计算只值 0.8 核，其余全是自旋；
- 降到 4 线程后 CPU 827%、ITL 61.7 → 36~40 ms，**白拿了 45~86%**。

> 2026-11-14 16:50 更正：①不是"五个 rank 各 28 线程 = 140 线程"，PP0~PP3 各只有 0.5~1.1 核（纯通信）；
> ②"频率降到 41%"作废，那是空闲读数，负载下实测 1.5~2.2 GHz，没有降频。

历史修正记录：「dspark 净亏」结论已作废；「51.8 是上限」也已作废（真实上限至少 92.9）。

## 5. 文件与回滚

| 文件 | 作用 |
|---|---|
| `/etc/dsv41-pp5/mode` | 当前模式（单行：legacy / ram / dspark） |
| `/usr/local/bin/dsv41-pp5-ctl.sh` | 模式化启停；旧版备份在 `dsv41-pp5-ctl.sh.bak-20261114` |
| `~/dsv41-repo-test/run-pp5-{legacy,ram,dspark}.sh` | 三个模式的完整启动参数 |
| `~/dsv41-repo-test/bench/bench_v1.py` | 固定基准脚本 |
| `~/dsv41-repo-test/bench/results.jsonl` | 历史结果（追加式） |
| `/var/log/dsv41-pp5-ctl.log` | ctl 操作与自愈日志；`/var/log/dsv41-crash/` 存崩溃现场 |

回滚到旧行为的完整命令：

```bash
sudo /usr/local/bin/dsv41-pp5-ctl.sh use legacy
```
