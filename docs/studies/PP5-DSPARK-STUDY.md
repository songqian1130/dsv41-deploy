# 新 PP5 的 dspark 是怎么做的，旧 PP5 怎么学

> 2026-11-14 13:30 实地读码得出。两个镜像的源码逐文件核对过，不是推测。

## 0. 先纠正三条此前说错的话

| 之前的说法 | 事实 | 错因 |
|---|---|---|
| "旧引擎镜像里没有投机解码代码，`num_dspark_layers`/`dspark_v2` 都是 0 命中" | **旧引擎有完整 dspark**：`v1/worker/gpu/spec_decode/dspark/{speculator,markov_argmax,utils}.py` + `models/deepseek_v4_1/nvidia/dspark.py`(577 行)，`config/speculative.py` 里 `DSparkModelTypes = Literal["dspark"]` | 检索的标识符根本不存在，把"没搜到名字"当成"没有功能" |
| "旧 PP5 上 dspark 这条路不存在" | 存在，且不需要移植任何代码 | 由上一条错误推出 |
| "新 PP5 靠 dspark 涨到 36 但旧引擎 43，所以净亏" | 用高熵用例+冷启动单轮测的，已作废（见 PP5-BASELINE.md 修正节） | 测量偏差 |

## 1. dspark 的机制（读码所得）

`dspark/speculator.py` 开头两行注释是核心：

```
DSpark drafts a block of `num_speculative_tokens` tokens in one parallel pass
over `num_speculative_tokens` query tokens (anchor + N-1 noise), NOT `1 + N`.
```

- **并行出块**：一次前向同时算 anchor + (k-1) 个 noise 位置，得到 k 个 draft。不是传统 EAGLE 那样自回归 k 次。所以 draft 成本与 k 几乎无关，这是它能上 k=3/5 的原因。
- **一次 verify**：主模型对 anchor + k 个 draft 做一次前向，权重带宽按 1 个 token 摊，产出最多 k+1 个 token。
- **markov_argmax.py**（442 行）：draft 之间的相关性建模，决定接受率上限。
- **adaptive_verification**：接受率掉下去时自动降 k，避免白算。
- **权重来源**：checkpoint 里的 `mtp.{0,1,2}.*`，无 embed/head（复用主模型的）。
- **层数**：`num_dspark_layers = config.n_mtp_layers or config.num_nextn_predict_layers or 3`，**与 k 无关**。所以 k=1 也照样建满 3 层 → 固定 7.39 GiB（mtp.0/1/2 = 2.47/2.40/2.52）。

这就是旧引擎"放不下"的真正原因：不是 dspark 本身贵，是它无条件按 checkpoint 的层数建 3 层。

## 2. 新 PP5 的"放法"（`deepseek_v41_hybrid` 运行时）

`additional-config` 里那几个字段对应的实现，共 3220 行 6 个文件：

| 文件 | 行数 | 作用 |
|---|---:|---|
| `hybrid_runtime.py` | 1314 | 编排：覆盖率标定、槽位分配、CPU/GPU 分工 |
| `expert_cache.py` | 692 | 主机 LRU 专家缓存 |
| `cpu_moe.py` | 445 | CPU 侧专家 GEMV |
| `cache_budget.py` | 308 | 显存预算测量与缓存 |
| `hybrid.py` | 270 | 混合层封装 |
| `cache_policy.py` | 191 | 换入换出策略 |

关键设计（`cache_budget.py`）：

1. **槽位粒度 = 8 个专家一个 bank**（校验里有 `count % 8`），按「每层 × 每卡」分配。
2. **启动时实测**显存余量，算出每层能放几个 bank，结果写
   `$VLLM_CACHE_ROOT/startup_plan/dsv41_experts_<指纹>.json`。
3. **指纹极严**：源码 sha256、`.so` 大小+mtime、驱动版本、CUDA/torch 版本、`CUDA_VISIBLE_DEVICES`、`PYTORCH_CUDA_ALLOC_CONF`、`max_num_seqs`、`max_num_batched_tokens`、prefix caching 开关、`decode_query_len`……任一项变了就重算。
   它甚至把 `cpu_threads`、`host_cache_gib`、`route_tail_skip` 从指纹里**剔除**——因为这几个不影响显存占用。
4. **放不下的进主机**：`host_cache_gib 8` 做 LRU（实测 7.704 GiB）。
5. **CPU 算**：`cpu_threads [28,28]`，两颗 E5 各 28 线程，走 `cpu_moe.py`。
6. **PP 分层故意不均衡**：`pipeline_layers [5,5,5,4,21]`，前四段层数少，把显存让给专家缓存，21 层压到最后一段。

实际账（11-08 生产日志）：7680 个专家 → 7184 在卡（125.788 GiB，coverage 93.54%），496 在主机（8.68 GiB）。

> **口径澄清（2026-11-14 15:40 核对 checkpoint）**：7680 不是全模型的专家数，而是 **PP4 上 layer 20–39 这 20 层的专家数**（20 × 384）。
> 全模型 40 层**每层都是 MoE**（`layers.N.ffn.experts.{0..383}` 在索引里 40 层齐全），共 **15,360 个路由专家、268.95 GiB**；
> 另有 engram 两层各 93.75 GiB（合计 187.5 GiB），checkpoint 总体积 475.24 GiB。
> 下放窗口来自 `cpu_moe.py` 的 `start_layer=20, end_layer=40` 默认值，见 §7.4。

### 代价

每步只要命中一个主机专家，就要走 PCIe Gen2 x4 + CPU GEMV。实测每步 23 → 62~66 ms。
**这套放法是为了"塞进 4 张/5 张卡"而存在，不是为了加速。** 在显存够的时候，它是纯负债。

## 3. 旧 PP5 的正确学法是：不学放法，学"只放 1 层"

旧引擎已经有 dspark，缺的只是显存。三个可用的杠杆：

| 杠杆 | 效果 | 已验证 |
|---|---|---|
| `--hf-overrides '{"n_mtp_layers":1}'` | 只建 mtp.0，2.47 GiB（省 4.92） | 启动日志已确认 draft model 解析成功 |
| `--gpu-memory-utilization 0.97 → 0.99` | +1.28 GiB | 本次实验 |
| `VLLM_CPU_OFFLOAD_GB_PER_RANK` | 旧引擎自带的分rank下放，备用 | 未启用（0,0,0,0,0） |

显存账：`0.99 × 64 = 63.36`，权重 58.2 → 余 5.16 GiB；MTP 一层 2.47 GiB 后余 2.69 GiB 给 KV/workspace（ram 模式原本 KV 余 1.93 GiB）。

代价：**dspark 必须用 V2 model runner**。`gpu_worker.py` 里
```python
def _num_workspace_lanes(vllm_config, use_v2_model_runner):
    if use_v2_model_runner and spec_config is not None and spec_config.use_dspark():
```
只有 V2 才分配 dspark workspace；而惰性接收处有 `assert not self.use_v2_model_runner`。两者互斥，所以本次实验把 `VLLM_PP5_LAZY_RECEIVE=0`、`VLLM_USE_V2_MODEL_RUNNER=1`。

## 4. 本次实验配置（`run-pp5-old-dspark.sh`）

与 `run-pp5-ram.sh` **只有 5 处差异**：

```
VLLM_PP5_LAZY_RECEIVE      1 -> 0
VLLM_USE_V2_MODEL_RUNNER   0 -> 1
gpu-memory-utilization  0.97 -> 0.99
+ --speculative-config '{"method":"dspark","num_speculative_tokens":1}'
+ --hf-overrides '{"n_mtp_layers":1}'
```

engram 保持 `storage:"ram"`（和 ram 模式同源），不引入 SSD 变量；`kv_cache_dtype=fp8_ds_mla` 不变。

### 预期

| 项 | 推算 |
|---|---|
| 每步产出 | 1 + acc(1)，acc≈0.8 → 1.8 token |
| 每步耗时 | verify(2 token, 权重带宽主导) ≈ 23~25 ms + draft(1 层 MTP) ≈ 3~6 ms → 26~31 ms |
| decode | 1.8 ÷ 0.028 ≈ **64 tok/s** |
| 对照 | ram 模式 43.18 |

失败模式：V2 runner 在旧镜像上可能另有开销（失去惰性接收），或 workspace 多分配的显存导致 OOM。两种情况都会在加载日志里显形。

## 5. 三个实测结果（2026-11-14 13:30~14:10）

### 5.1 旧引擎 + dspark：三次 OOM，机制性堵死

配置：`VLLM_USE_V2_MODEL_RUNNER=1`、`LAZY_RECEIVE=0`、util 0.99、`dspark k=1`、`n_mtp_layers=1`、engram ram。

三次都死在同一个位置 `dspark.py:114`（构造 draft 的 ModuleList），且 OOM 数字**逐字节相同**：

```
CUDA out of memory. Tried to allocate 720.00 MiB.
GPU 4 total 63.39 GiB, free 156.31 MiB,
in use 63.22 GiB, allocated by PyTorch 62.47 GiB
```

| 尝试 | 手段 | 结果 |
|---|---|---|
| 1 | 基线（util 0.99 + 1 层 MTP） | OOM 720 MiB |
| 2 | 加 `--kv-cache-memory-bytes=2147483648` | **数字完全不变** |
| 3 | 加 `VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,4` | **数字完全不变** |

为什么无效：
- `--kv-cache-memory-bytes` 不解决问题，因为 **KV cache 是在 draft 加载之后才分配的**，OOM 发生在更早的"主模型铺满 → draft 再来要"阶段。旧引擎没有任何"给 draft 预留"的机制。
- `VLLM_CPU_OFFLOAD_GB_PER_RANK` 在这个路径上没生效（权重实际占用未变，PP4 仍 56.69 GiB + head）。
- 想靠 `VLLM_PP_LAYER_PARTITION` 从 rank4 挪走一层也不可行：单层权重约 7 GiB，而其它 rank 各自只有约 6 GiB 余量。

**结论：旧引擎 + dspark 在物理上不可行**，不是参数没调对。新引擎那套 3220 行 `deepseek_v41_hybrid` 存在的意义，正是为了在这个"先到先得"的分配顺序之外，给专家/draft 留出可调度的空间。

### 5.2 新引擎 + 只留 1 层 MTP：步时几乎没降，速度腰斩

配置：`--hf-overrides '{"n_mtp_layers":1}'` + `k=1`，其余与生产 dspark 完全一致。

| 轮次 | code | json | zh_tech | math | 步时 ITL | acc |
|---|---:|---:|---:|---:|---:|---:|
| 第 1 轮 | 22.51 | 23.37 | 23.70 | 23.04 | 71~86 ms | 0.69~0.93 |
| 第 2 轮 | 32.21 | 32.07 | 28.48 | 31.30 | **57~60 ms** | 0.65~0.95 |

对比 3 层 + k=3：code **51.80**、步时 **65.6 ms**、每步 3.55 token。

**省掉 4.92 GiB 的 MTP 权重，步时只从 65.6 降到 58 ms（−12%），但每步产出从 3.55 掉到 1.8（−49%）。** 净效果腰斩。

这条数据证明了一件事：**步时的绝大部分来自 496 个下放专家的 PCIe 读取，不来自 MTP 层数。** 少建两层 MTP 只是省了显存，没省时间。

### 5.3 顺带发现：startup_plan 的覆盖率没有随 MTP 层数重算

`n_mtp_layers` 改了、可用显存多了 4.92 GiB，但日志里仍然是

```
coverage=93.54% (7184/7680, 125.788 GiB); host LRU 7.704 GiB
```

与 3 层版本**完全相同**。说明 `cache_budget.py` 的指纹没有嗅到 `hf_overrides` 的变化，直接复用了 `startup_plan/dsv41_experts_<指纹>.json`。

而同一行日志里的 `post-KV budgets` 显示各卡还有 **21.5~37.8 GB** 余量，却只放了 93.54% 的专家——**计划与真实余量不匹配**。这指向一个还没试过的实验：删掉 `startup_plan` 缓存强制重标定，看 coverage 能否逼近 100%。若真能全驻留，步时可能掉到 30 ms 级，那时候 1 层 MTP 也有 45~60 tok/s。

## 5.4 决定性结论：新 PP5 快的原因只有一个，且已到结构上限

### 反事实验证（这是最直接的证据）

只改 MTP 层数、不改其它任何东西：

| 配置 | 每步 token | 每步耗时 | code tok/s |
|---|---:|---:|---:|
| 3 层 MTP + k=3 | 3.55 | 65.6 ms | 51.80 |
| 1 层 MTP + k=1 | 1.80 | 58.0 ms | 22.5~32.2 |

**平台没变，只改了"每步产出几个 token"，速度就从 51.8 掉到 22.5。** 说明速度完全由 `(1 + k × acc) ÷ 步时` 决定，与引擎平台无关。这就是新 PP5 快的全部原因：**dspark 让每步产出 3.55 个 token，而不是 1 个。**

### 步时的结构下限

新引擎 `pipeline_layers` 有硬约束：

```
ValueError: pipeline_layers must partition 40 layers across PP stages,
with layers 19–39 on the last stage
```

最后一段必须承载 19~39 共 21 层，而 40 层**每层都是 MoE、每层 384 个专家（6.72 GiB）**，21 层光专家就 141 GiB，一张卡装不下 → 必须靠 hybrid 把 layer 20–39 的专家分摊出去。

**分布式缓存（2026-11-14 修正）**：这 20 层的专家缓存并不是"只剩 3 GiB 塞在 PP4 自己卡上"，而是**分摊到 5 张卡**：

```
Hybrid final expert capacity: device=0 slots=1128
                              device=1 slots=1336
                              device=2 slots=1128
                              device=3 slots=1744
                              device=4 slots=1880   → 合计 7216
```

前四段各只放 4~5 层（34.4 GiB 左右），空出来的 25~42 GB 显存被拿来当 PP4 的专家仓库，这也是必须开 CUDA MPS 的原因。剩下 464 个（7.7 GiB）放主机 LRU。所以步时 61 ms 不是"卡不够"，而是 20 层里总有几层的个别专家被挤到主机，每层都要过一次 CPU 线程池。

### 但 KV 池确实浪费了一大块（可回收）

生产脚本 `--kv-cache-tokens 524288`，而 `max-num-seqs` 只有 4。砍到 131072：

| 用例 | 生产（KV 524288） | KV 131072 | 变化 |
|---|---:|---:|---:|
| code_python | 51.80 | **55.93 / 58.25** | +10% |
| json_struct | 38.68 | **49.18 / 49.48** | **+27%** |
| zh_tech | 38.49 | 42.63 / 38.89 | +5% |
| math_steps | 36.29 | 41.64 / 40.03 | +12% |

步时从 65.6 → 60~62 ms，覆盖率从 93.54% → 93.96%（7216/7680）。

### 最终账（同一套固定用例，热机后）

| 配置 | code | json | zh_tech | math | 步时 |
|---|---:|---:|---:|---:|---:|
| 旧引擎 ram（无损） | 43.46 | 43.56 | 43.42 | 43.45 | 23.2 ms |
| 新引擎 dspark 生产 | 51.80 | 38.68 | 38.49 | 36.29 | 65.6 ms |
| **新引擎 dspark + KV 131072** | **57.1** | **49.3** | **40.8** | **40.8** | **61 ms** |

**新 PP5 快的机制 = dspark 一步多 token；代价 = PP4 的 21 层里，layer 20–39 的专家必须靠 5 卡分摊 + 主机 LRU 才放得下，每层多一次 CPU 往返。** 两边相抵后，代码类赢 30%，开放文本基本打平或略输。

## 7. 步时 61 ms 花在哪：不是"传权重"，是 CPU 线程超订（2026-11-14 15:00 实测）

### 7.1 先排除掉一个想当然的假设

直觉是"496 个专家下放 → 每步经 PCIe 读权重 → 慢"。**三条证据否掉它**：

1. 运行时统计：`GPU hit rate 99.07~99.44%`，每 10 秒只有 327~686 个 CPU route，折合 **每步 3~4 个主机专家**，不是 496 个。
2. 代码路径：`cpu_moe.py` 用 `backend=ik` 的 `CPUMXFP4Experts` **在 CPU 本地算 GEMV**，权重常驻主机内存，**不往 GPU 拷**。每步只往返 hidden states（4 token × 5120 × 4B ≈ 80 KB）。
3. 算术：单专家 17.7 MB（H=5120, I=2304, MXFP4），3.7 个 = 65 MB，55 GB/s 内存带宽只要 **1.2 ms**。

所以"传权重"和"CPU 算力"两个假设都解释不了 30 ms 的缺口。

### 7.2 真正的证据：CPU 被打满，但打满的不是"计算"

跑一次 500 token 输出，同时采样：

```
按 rank 读 /proc/<pid>/stat 增量（16:50 实测，4 线程配置）
  PP0 1.07   PP1 0.90   PP2 0.77   PP3 0.57   PP4 3.20 核   合计 6.4 核
28 线程配置（同一天更早）
  docker stats  CPU = 2721%     ← 27 核，全部来自 PP4 一个池
```

**只有 3~4 个专家要算，却占掉 27 个核，所以钱不是花在计算上，是花在"每层唤醒/同步一个 28 线程池"上。**

> **【2026-11-14 16:50 更正】频率不是原因，撤回。**
> 之前引用的 `lscpu CPU(s) scaling MHz = 41%`（≈0.98 GHz）是**空闲读数**：governor 是 `schedutil`，
> 无负载时就是 1.3 GHz（=41%×3.3 GHz）。实测负载下反而升到 1.5~2.2 GHz（decode 1.5~1.8、
> prefill 1.4~2.2），**没有任何降频**。这一点和 `aiultrayang-pp4-20261103/findings.md:97`
> 早先的结论一致（"抽样核未绑定工作线程，不能据此判定降频"）。

### 7.3 根因：`cpu_threads` 的 decode 档配了 28，而真实并行度只有 1 核

```json
"cpu_threads":[28,28]     // [prefill, decode]
```

- 机器物理核 **28**（2×E5-2680v4，14 核/颗，56 逻辑线程）
- **只有 PP4 有 CPU MoE**：下放窗口 20~39 层全在 PP4（见 §7.4）。PP0~PP3 的 0.5~1.1 核是 NCCL 通信
  + Python 驱动，不参与专家计算。所以不存在"140 线程争 28 核"，**是一个池的 28 线程把自己撑满**
- PP4 上的 20 层每层都要走一次 `gpu_cache.launch()` + `_forward_async()` + `join()`；
  OpenMP 把活切成 28 份，**barrier 要等最慢那个线程到达**，而 decode 每层只有几十行数据
- 20 次调用/步 × 每次几百微秒~几毫秒的固定开销 = 步时的主要增量

### 7.4 层结构核对：40 层全是 MoE，只有 20 层在下放窗口内

| 项 | 数值 | 依据 |
|---|---:|---|
| 总层数 | 40 | `text_config.num_hidden_layers=40` |
| MoE 层数 | **40（全部）** | checkpoint 里 `layers.{0..39}.ffn.experts.{0..383}` 齐全 |
| 每层路由专家 | 384（+1 shared） | `n_routed_experts=384` / `n_shared_experts=1` |
| 全模型路由专家 | **15,360** | 40 × 384，权重 268.95 GiB |
| 单层专家体积 | 6.72 GiB（6885 MiB） | 单专家 17.93 MiB × 384 |
| 单层非专家部分 | 162 MiB | attention / norm / shared expert |
| 下放窗口 | **layer 20–39** | `cpu_moe.py`：`start_layer=20, end_layer=40` |
| 窗口内专家 | **7,680** | 20 × 384 = 134.47 GiB，占全模型专家一半 |

这个模型**没有 dense 层**：`DeepseekV4DecoderLayer.__init__` 无条件 `self.ffn = DeepseekV4MoE(...)`，config 里也没有 `first_k_dense_replace` / `moe_layer_freq` 这类字段。所以"20 个 MoE 层"是错的，正确说法是"**20 层参与 CPU 下放**"。

窗口为什么正好是 20–39：PP 分段 `[5,5,5,4,21]` 让 PP0–PP3 各只装 4~5 层（每段约 34.4 GiB），专家全部放得下；PP4 装 layer 19–39 共 21 层，只有它的专家放不下。`cpu_moe_config()` 按层号过滤，PP0–PP3 手里的层号都 < 20，自然全部返回 `None` —— **不是"只支持最后一段"的硬性限制，而是窗口恰好只覆盖 PP4 里的这 20 层**。layer 19 也在 PP4，但不在窗口内，它的 384 个专家 100% 驻留本卡。

### 7.5 单变量实验：只把 decode 线程 28 → 4

prefill 保持 28，其余参数（dspark k=3、KV 131072、util 0.99、pipeline_layers[5,5,5,4,21]）一律不动：

```
CPU     2721%  →  827%（后段 400%）
ITL     61.7ms →  36.0~40.4ms
```

| 用例 | 28 线程 | **4 线程** | 提升 |
|---|---:|---:|---:|
| code_python | 54.89 / 58.68 | **85.24 / 92.88 / 87.71** | **+55%** |
| json_struct | 48.35 / 47.56 | **72.39 / 78.62 / 79.26** | **+66%** |
| zh_tech | 42.54 | **62.12 / 61.67** | +45% |
| math_steps | 40.46 | **67.66 / 73.59** | +75% |
| C1_decode（高熵，最差用例） | ~36 | **58.46** | +62% |
| C2_ttft22k | 8.6~10.5 s | **10.197 s**（prefill 未改） | 持平 |
| C3_conc3 三路聚合 | 32~42 | **77.94**（每流 25.98） | **+86%** |

CPU 占用从 2721% 掉到 827%，速度反而翻了 1.5 倍 —— **减少线程不是省算力，是消掉了"28 个线程等最慢那个"的同步开销**。

### 7.6 步时的正确拆账（16:50 修订）

| 组成 | 成本 | 说明 |
|---|---:|---|
| **GPU 流水线（地板）** | **~37 ms** | 40 层 / 5 段串行，dspark 每步要验 4 个 token。cov 实验（18 层不碰 CPU）步时一动不动，反证地板在 GPU |
| CPU 专家真实计算 | ~0.8 核（藏在 GPU 后面） | 4 线程时完全被 GPU 阴影覆盖，拿掉也不影响步时 |
| 28 线程的同步开销 | +25 ms | 61.7 − 37；20 次调用/步 × 固定开销 |
| ~~全核降频~~ | ~~—~~ | **不存在，撤回**（见 §7.2 更正） |

即：**"offload 导致步时长"这个判断方向对，但真正贵的是"每层都要同步一趟 28 线程池"，不是搬运权重、也不是 CPU 算力。** 把线程降到 4 以后 CPU 侧就彻底躲到 GPU 后面了，再减 CPU 工作量（§7.11 的覆盖率实验）没有任何边际收益。

### 7.7 配置落地

- 新脚本 `run-pp5-dspthr.sh`，容器 `dsv41-pp5-dspthr`
- 已加入 `ctl.sh` 模式表为 **`fast`**，`/etc/dsv41-pp5/mode` = `fast`，看门狗 `dsv41-pp5-health.timer` 已恢复（跟随模式自愈）
- 切换：`sudo /usr/local/bin/dsv41-pp5-ctl.sh use fast|ram|dspark|legacy`

### 7.8 线程数是否最优点：decode 4 vs 8 实测（2026-11-14 16:10）

单变量：只把 `cpu_threads` 的 decode 档 4 → 8，prefill 保持 28，其它一律不动。

| 指标 | decode=4 | decode=8 |
|---|---:|---:|
| code_python | 85.5 / 92.9 / 87.7 | 91.8 / 90.6 / 85.6 |
| json_struct | 78.7 / 78.6 / 79.3 | 81.0 / 78.7 / 81.3 |
| zh_tech | 62.1 / 61.7 | 65.9 / 63.4 |
| math_steps | 67.7 / 73.6 | 67.2 / 71.2 |
| **每步 ITL** | **38.9~40.3 ms** | **36.5~40.0 ms** |
| PP4 CPU 占用 | **410%** | **800~810%** |

**两档速度相同，步时都在 38~40 ms。** 表面上的 +3~6% 全部来自接受率波动：

```
decode = (1 + 3×acc) ÷ 步时
   acc=0.816 → 85.5 tok/s @ 40.3 ms
   acc=0.892 → 91.8 tok/s @ 40.0 ms     ← 同为 4 线程时也能出 92.9
```

**结论：decode CPU 线程数不是速度杠杆。** 4 线程已经把 3~4 个主机专家算完了，8 线程只是让 PP4 多烧一倍 CPU（410%→810%），不产出任何 token。保持 **4**。

### 7.9 为什么剩下的 20 个核用不上（重要，纠正一个前提）

"5 个 PP × 4 线程 = 20 核"这个前提不成立。实测每个 rank 的真实 CPU 占用：

| 阶段 | PP4（唯一有 CPU 专家的 rank） | PP0~PP3 | 合计 |
|---|---:|---:|---:|
| decode（线程=4） | **410%** | 各 50~110%（轮询/调度） | **~8 核** |
| decode（线程=8） | **800~810%** | 各 50~110% | ~11 核 |
| **prefill（22k）** | **2482%** | 各 ~91% | **~28 核（打满）** |

decode 阶段剩下的 20 核不是"没分配"，而是**没有活可干**：

- 每步只有 **3~4 个专家**路由到 CPU（GPU 命中率 99.07~99.44%），约 70 MB 权重，满带宽只要 **1.3 ms**；
- 但每步要走 **20 次**「launch → CPU GEMV → join」，因为 20 个下放层**每层都至少有 8 个专家在主机**；
- 这 20 次往返是**延迟**，不是吞吐。加线程不减少往返次数，所以加核无效。

**prefill 则相反：PP4 一个人吃掉 24.8 核，加上其它 4 个 rank 正好把 28 核打满。** 所以 prefill 保持 28 是唯一正确值——代码也不允许超过物理核数（`hybrid_runtime.py:220` 会直接报错），调低只会拖慢 TTFT。

### 7.10 真正还能提速的方向（不是线程）

线程这条路已经走到头，剩下两个方向：

1. **把 464 个主机专家塞回卡里（coverage 93.96% → 100%）**：一旦每层都没有主机专家，20 次 CPU 往返全部消失，步时有机会从 38 ms 掉到 26~28 ms → **~125 tok/s**。目前 `post-KV budgets` 合计 159.6 GB、7680 个专家只需 144.4 GB，但计划只用了 135.6 GB，还有约 24 GB 没被利用，需要读 `cache_budget.py` 的 `plan_expert_cache` 确认是安全预留还是分配粒度损耗。
2. **`--max-num-batched-tokens` 2048 → 4096**：启动日志已警告"开了投机后 2048 可能次优"。

## 6. 顺带发现

- 旧镜像启动日志报 `Unknown vLLM environment variable: VLLM_PP5_LAZY_RECEIVE / VLLM_PP5_SHARED_MAPPING`。这两个变量没注册在 `envs.py`，是直接用 `os.getenv` 读的，功能仍在，只是不享受 envs 的类型转换和文档。
- 本次启动警告：`max_num_scheduled_tokens is set to 2048 based on the speculative decoding settings. This may lead to suboptimal performance.` → 开了投机后 `--max-num-batched-tokens` 应该往上抬（4096）。下一轮再调，先不动，保持单变量。

### 7.11 覆盖率实验（已做，结论：无效，§7.10 第 1 条的预测作废）

2026-11-14 16:40 实测。改 `hybrid.py` 三处（190/202/227）把每层上限从 376 放到 384，
用 `-v` 只读挂载进容器（`/home/ubuntu/dsv41-repo-test/hybrid-cov.py`），其余配置与 `fast` 逐字相同：

| 项 | fast（未打补丁） | cov（打补丁） |
|---|---|---|
| 每层槽位 | 20~37 层 376，38 层 240，39 层 208 | 20~37 层 **384**，38 层 208，39 层 184 |
| 总槽位 | 7216 / 7680（93.96%） | **7304 / 7680（95.1%）** |
| code_python | 85~93 tok/s | 80.6 |
| json_struct | 76~79 | 77.5 |
| zh_tech | 61~62 | 60.6 |
| math_steps | 67~77 | 71.0 |
| ITL | 36~40 ms | 36.8~41.6 ms |
| PP4 CPU（decode） | 410% | 330% |
| 容器总 CPU（decode） | 827% | 679% |

**结论：速度在噪声内，一点没变快。** 两条根因：

1. **容量才是硬约束，不是 `bank_size - 1`**。五卡容量实测 1152/1336/1152/1744/1920 = **7304 个槽**，
   本身就装不下 7680。去掉 `-1` 只多塞了 88 个专家，最后两层仍缺 176/200。
   要真正 100% 得把 KV 池或 engram 再压出约 0.4 GB × 5，收益未知、风险明确。
2. **就算塞满了也不快**。CPU 专家的活早就"躲在 GPU 后面"了：把 18 层的 CPU 调用拿掉，
   PP4 CPU 从 410% 掉到 330%，步时一动不动。**38 ms 是 GPU 流水线的地板**（40 层 / 5 段串行，
   dspark 每步要验 4 个 token），不是 CPU 往返造成的。

→ §7.10 里"补满专家 → 26~28 ms → 125 tok/s"的推算是**错的，作废**。线程 28→4 的收益是真的，
但那是因为 28 线程让 CPU 侧从"躲在 GPU 后面"变成"卡在 GPU 前面"（20 次/步 × 28 线程 barrier 的固定开销），
不是因为 CPU 往返多，也不是因为降频（那个说法已撤回，见 §7.2）。4 线程已经把 CPU 侧压到 GPU 阴影内，再减 CPU 工作量没有边际收益。

回滚：`docker rm -f dsv41-pp5-cov` → `systemctl start dsv41-pp5.service` + health.timer，已恢复 fast。

### 7.12 `cpu_threads` 到底是什么（2026-11-14 17:05，代码 + 实测核对）

**语义**：两个数**都是 CPU 线程数**，是同一个线程池的两档，与 GPU 无关。

```python
"cpu_threads": [28, 4]      # [prefill, decode]；写单个数则两档同值（默认 22）
hybrid.py:122               # 解析 + 校验格式
hybrid.py:163               # -> additional_config["cpu_phase_threads"]
model_state.py:201          # 每个调度步：phase_threads[0 if has_prefill else 1]
                            #   只对"有 CPU 专家后端"的模块 backend.set_num_threads()
hybrid_runtime.py:217       # max(phases) 必须 <= 物理核数(28)，写 29 直接拒绝启动
```

**不是核、不绑核、不独占、不是每个 PP 一份**：

- 日志里 `CPU routed experts: backend=ik, threads=28 ...` 共 **20 条，全部来自 `Worker_PP4`**，PP0~PP3 零条
  —— 0~19 层没有下放专家，没有 CPU 专家模块，**连这个参数都不看**。
- 线程数实测（`ps -o nlwp`）：PP0 108 / PP1 70 / PP2 71 / PP3 70 / **PP4 94**。
- **进程级绑核，线程级不绑（2026-11-14 17:15 更正）**：`gpu_worker.py:560` 在 `load_model` 里
  `os.sched_setaffinity(0, physical_cpus())`，实测 `/proc/942/status` → `Cpus_allowed_list: 0-27`。
  `physical_cpus()` 按 `(package_id, core_id)` 去重得 `[0..27]`，而 `cpu28` 与 `cpu0` 同为 pkg0/core0
  → **超线程 28~55 被排除在外**，所以 56 个逻辑 CPU 只有 28 个可用，上限只能是 28。
  绑的是"进程能跑在哪 28 个 CPU"，**不是"线程绑给某个专家"**：池内线程按编号静态切块（线程 i 拿第 i 块），
  块内容每次调用都变（取决于这步路由到哪些专家/token）。少数在绑核前创建的 CUDA/NCCL 线程仍是 0-55。
- 所以"5 个 PP × 4 = 20 核"这个理解是错的，decode 全程只用到 6.8 核。

**prefill 的 28 是真吃满的（推翻 §7.4 的 1 秒采样结论）**

冷 prefill 97,537 token（随机 nonce 破前缀缓存）、0.2 s 采样、82 个样本：

| | PP0 | PP1 | PP2 | PP3 | PP4 | PP4 单线程占用 >0.5 核的线程数 |
|---|---|---|---|---|---|---|
| 峰值 | 1.20 | 1.15 | 1.20 | 1.10 | **18.95** | **27 / 28** |

即 prefill 阶段 28 个线程里 27 个都有真活，28 用得其所。
上一版写的"PP4 峰值 5.1 核、全程 4~9 核"是 **1 s 采样 + 命中前缀缓存（TTFT 仅 1.1 s）** 造成的假象，作废。

**两种负载的对照（同一个池，两种形状）**

| | 一次调用要算多少行 | 28 线程里几个有活 | PP4 实占 |
|---|---|---|---|
| prefill（97k token） | 几十万行，持续几百 ms | **27** | 18.95 核（真算） |
| decode（1 步） | 几行，几十 µs | 3~4 | 4.1 核（几乎全是自旋，真算 0.8） |

脚本：`bench/probe_prefill_threads.py`（`PROBE_NONCE` 换值即换冷/热）。注意 1 s 采样会抹平 prefill 的峰值，必须 ≤0.2 s。

### 7.13 dspark k=3 → 5（已上线，2026-11-14 17:45）

**线索**：`hybrid.py:149`

```python
spec.setdefault("dspark_num_query_tokens", max(5, int(spec["num_speculative_tokens"])))
```

query block 的下限本来就是 **5**，而生产只填了 k=3 —— 等于只用掉 3/5 的额度。

**三轮固定用例（seed 51/52/53，全 cold）**

| 用例 | k=3（旧 fast） | k=5 轮1 | k=5 轮2 | k=5 轮3 | **k=5 均值** | 变化 |
|---|---:|---:|---:|---:|---:|---:|
| code_python | ~89 | 96.4 | 108.9 | 94.9 | **100.1** | +12.5% |
| json_struct | ~77.5 | 82.9 | 86.5 | 86.7 | **85.4** | +10.2% |
| zh_tech | ~61.5 | 64.8 | 65.0 | 65.8 | **65.2** | +6.0% |
| math_steps | ~72 | 77.4 | 80.2 | 79.6 | **79.1** | +9.9% |

- 步时：38~39 ms → **41~46 ms**（+8~19%）
- 每步 token：3.4 → **4.2**（+24%，因为 1+k×acc）
- acc/draft 随 k 上升而下降（code 0.70~0.82，中文 0.48~0.50），这是正常的：越靠后的 draft 越难被接受
- **显存零成本**：覆盖率仍是 93.96%（7216/7680），与 k=3 逐字相同（dspark 层数与 k 无关，见 §2）
- 回滚：`cp run-pp5-dspthr.sh.bak-k3 run-pp5-dspthr.sh`

**为什么不是 `max(5,k)` 暗示的"白送 66%"**：query block 是"能装几个"，k 是"实际发几个"。
k=5 时验证 token 从 4 个涨到 6 个，GPU 每步多算 50%，所以只净赚约 10%。

### 7.14 外部项目清单（STRATA / MRV2 / Delta）的可搬性评估

前提：这批数字来自**另一个 C++ 引擎 + 2.25bpw 量化包**
（`~/strata-dsv41-data/strata-*.json`：`engine-tp3/strata-dsv41`、`packs/...-2.25bpw`，`IQ2_XS` ≈ 2.25 bit/权重），
模型体积约为原版 1017 的 1/10，**不能与本机 268.95 GiB 满血权重直接比**。

| 条目 | 结论 | 依据 |
|---|---|---|
| MRV2 Runner（+10%） | **已含** | 日志 `[gpu_worker.py:475] Using V2 Model Runner`；dspark 本身就依赖 V2 |
| 路由偏置（+25%） | **不适用** | 前提是"大量 token 落到主机专家"。我们 GPU 命中率 99.07~99.44%，每步只 3~4 个主机专家；且 §7.11 已证明拿掉 18 层 CPU 调用步时不变 → 收益≈0 |
| DSpark k=5 | **已上线** | 见 §7.13，+6~12.5% |
| PP_KEEP_BATCH（+24.8% @128K） | 需移植，低优先 | 针对"空 pipeline step 把请求踢出 batch"，我们 `max-num-seqs=4`、无该症状证据 |
| packed KV + KV 热环（−1.09 ms @300K） | 需移植，低优先 | 收益随上下文长度增长；我们 `max-model-len=131072`、短请求为主 |
| 前缀缓存分槽 | **已含（内存级）** | 生产已带 `--enable-prefix-caching`；他们的 `STRATA_DS4_SLOT_GB` 是磁盘 slot，属另一引擎 |
| Engram 优化（hugetlb/NUMA/热集/预读，+22%） | **唯一值得做，但要改代码** | 我们 engram 在 SSD，实测 19~24 倍读放大、有效带宽仅 550~570 MB/s，TTFT 10.2 s。ram 模式需 183 GiB pinned，现只剩 73 GiB 可用 → 只能走"加大行缓存/host_cache"这条配置路 |

### 7.15 Humming 后端评估：镜像里已装，但这台机器上用不了（2026-11-14 19:35）

**结论先说：不是"没试"，是"接不上"。** 三条独立证据，任何一条都足以否掉。

**① 镜像里本来就有，不需要 pip install**

```
/workspace/vllm-backport/requirements/cuda.txt:36   humming-kernels[cu13]==0.1.12
/workspace/vllm-backport/.venv/.../humming/__version__ = 0.1.12   （已装）
```

它是 vLLM 的**声明依赖**，不是可选插件。所以"vLLM 0.29+ 自带 Humming"这件事对我们早已成立。

**② `--linear-backend humming` 是非法值**

`vllm/config/kernel.py:187` 的 `LinearBackend` 只有 16 个值：auto / cutlass / flashinfer_cutlass /
flashinfer_cutedsl / flashinfer_trtllm / flashinfer_cudnn / flashinfer_b12x / b12x / marlin / triton /
deep_gemm / torch / aiter / machete / fbgemm / conch / exllama / emulation / xpu / xpu_woq —— **没有 humming**。
只有 `MoEBackend`（:122）里有 `"humming"`。

**③ 我们的专家路径根本不经过通用 FusedMoE，`moe_backend` 读不到**

| 路径 | 实现 | 是否看 `kernel_config.moe_backend` |
|---|---|---|
| 通用 vLLM MoE | `layers/fused_moe/` + `oracle/mxfp4.py`（有 HUMMING 分支） | 是 |
| **我们的 DeepSeek-V4.1** | `models/deepseek_v4_1/expert_cache.py:645` **无条件**调 `fused_marlin_moe(...)`；CPU 侧 `cpu_moe.backend="ik"`（`hybrid.py:155`） | **否** |

`deepseek_v4_1/` 里唯一引用 `kernel_config.moe_backend` 的地方是
`nvidia/model.py:257/489` 的 `in MEGA_MOE_BACKENDS` 判断，与 humming 无关。
dspark 的 3 层 MTP 也不建 FusedMoE（`nvidia/dspark.py` 零命中）。

启动日志直接印证：

```
Using MarlinMxfp8LinearKernel for MXFP8 GEMM
Using 'MARLIN' Mxfp4 MoE backend.
Using MarlinExperts
```

**④ 更根本：sm80 上根本不存在 Humming 想做的事**

`humming/tune/sm8x.py::Sm80Heuristics`：

```python
sm_version: int = 80
b16_allowed_dtypes = [float16, bfloat16]
b8_allowed_dtypes  = [int8]
b4_allowed_dtypes  = [int4]        # 没有 fp4，也没有 fp8
```

Humming 在 H20/H200 上的卖点是 **W4A8：MXFP4 权重在 kernel 内转 FP8**（省一半带宽、用 FP8 tensor core）。
GA100（sm80）**没有 FP8 tensor core**（FP8 从 Hopper sm90 才有），所以这条路在物理上不存在。
sm80 上 Humming 能做的 = int4 权重 dequant → BF16 计算，**正是 Marlin 已经在做的事**。
换句话说：我们已经跑在"安培版的 Humming"上了，换过去最多是同一件事的另一种写法。

**顺带否掉日志里那条自荐优化**

启动日志有一行：

```
[marlin_utils.py:612] Marlin kernel can achieve better performance for small size_n
with experimental use_atomic_add feature. You can consider setting VLLM_MARLIN_USE_ATOMIC_ADD to 1
```

看起来正好对症（我们是小 batch decode），但两条都堵：
- **MoE 不认这个开关**：`experts/marlin_moe.py:152/216` 两次 `marlin_gemm(...)` 都硬编码 `use_atomic_add=False`；
- **dense 侧在 sm80+bf16 被显式拒绝**：`marlin_utils.py:638`
  `if device_capability[0] < 9 and dtype == torch.bfloat16: return False`，而我们 `dtype=torch.bfloat16`。

→ `VLLM_MARLIN_USE_ATOMIC_ADD=1` 在这台机器上不会产生任何变化，不需要试。

### 7.16 TTFT 实测（2026-11-14 19:55，线上 fast + k=5，全部冷启动）

脚本 `bench/probe_ttft.py`（`PROBE_NONCE` 自动加随机前缀破前缀缓存；读 `delta.reasoning_content`
而不是 `delta.content` —— 推理模型的第一个 token 在思维链里，读错字段会得到 nan）。

**同一份配置，三种文本的差距接近 2 倍** —— 说明 TTFT 不是纯算力问题：

| 口径 | prompt_tokens | TTFT | 等效 prefill 吞吐 |
|---|---:|---:|---:|
| 重复文本（同一段落 ×N，engram 最容易命中） | 21,882 | **5.00 s** | 4,374 tok/s |
| 归档用例 C2（随机生僻字，n-gram 全 miss） | 22,093 | **9.40 s** | 2,351 tok/s |

**随长度变化（重复文本口径）**

| prompt_tokens | TTFT | 等效吞吐 |
|---:|---:|---:|
| 133 | 0.52 s | 256 tok/s |
| 497 | 0.65 s | 764 tok/s |
| 1,862 | 1.20 s | 1,550 tok/s |
| 7,322 | 2.63 s | 2,783 tok/s |
| 21,882 | 5.00 s | 4,374 tok/s |
| 32,802 | 6.83 s | 4,800 tok/s |

**随机词序（无重复、接近自然语言多样性）**

| prompt_tokens | TTFT | 等效吞吐 |
|---:|---:|---:|
| 795 | 1.16 s | 682 tok/s |
| 3,158 | 2.60 s | 1,215 tok/s |
| 9,561 | 4.24 s | 2,253 tok/s |
| 14,398 | 5.09 s | 2,828 tok/s |

**结论**

1. **TTFT ≈ 固定开销(0.5~1.0 s) + tokens ÷ 边际吞吐(3.5k~5.5k tok/s)**。
   固定开销来源：5 段流水线启动 + chunk 调度 + engram 首次 touch。
2. 吞吐随长度**上升**（短请求全是固定开销），所以"22k 用 2.2k tok/s"这种线性外推是错的。
3. **同一 token 数下，文本 n-gram 多样性直接决定 TTFT（5.0 vs 9.4 s，1.9 倍）**——
   这正是 engram 的 SSD 读放大（19~24 倍，§4）在 prefill 阶段的表现：
   n-gram 命中就快，全 miss 就每步都去盘上捞。
4. 归档口径 **9.40 s**，与之前记录的 10.2 s 同量级（差 8%，属噪声/engram 冷热），
   **k=5 对 prefill 无影响**（dspark 只在 decode 生效），符合预期。
5. `--max-num-batched-tokens 4096` → 22k 要跑 6 个 chunk，每 chunk 一次完整 PP 流水；
   这是固定开销的主要来源之一，**调到 8192 值得一试**（chunk 数 6→3）。

C3 三路并发 TTFT 8.15 s（每路 4.6k token）：并发时 prefill 被分片排队，不是单请求变慢。

### 7.17 主机内存构成：有一份 134.5 GiB 的专家重复副本（2026-11-14 20:10）

`free`：总 247 GiB / 已用 173 GiB / 页缓存 61 GiB / 可用 73 GiB（swap 只用了 255 MiB）。

**最大单项是 PP4 的一个匿名映射：141,028,772 kB = 134.5 GiB，全部驻留**（`pmap -x`）。

证据链（四条独立吻合）：

1. 日志原话：`CPU expert NUMA: ... threads=28, row sharding=False; explicit membind preserved;
   **one resident weight copy**`
2. `cpu_moe.py:198`：`if self.num_loaded == 3 * self.backend.num_experts: ... flush()`
   —— 每层 384 个专家**全部加载**，不是只加载 GPU 上缺的。
3. 同段代码里 `self.gpu_cache.load_weight(expert, ...)` 又把其中 93.96% 拷进显存。
4. 算术：20 层（layer 20~39）× 384 专家 × 17.9 MiB = **134.3 GiB** ✅

**即：同一批专家权重在主机和显存各有一份。** 主机必须保留的只有 GPU 没驻留的
464 个（7.7 GiB）；另外 7216 个（≈126 GiB）是纯冗余。

其余占用：PP4 的 host LRU 7.25 GiB（`host_cache_gib=8`）、PP0~PP3 各 2.8~5.2 GiB、
页缓存 61 GiB（模型文件，可回收但回收无收益 —— engram 走 O_DIRECT 不吃页缓存）。

**能腾多少**：把主机池从"每层全量"改成"只装 GPU 未驻留的"，可释放约 **126 GiB**，
届时可用 73+126 = 199 GiB > engram 全上 RAM 所需的 183 GiB。
→ 那会把 TTFT 的最不利口径从 9.4 s 拉到 5 s 量级，并彻底消掉 19~24 倍 SSD 读放大。
**风险**：GPU 缓存换出时，专家来源会从"主机池"退化为"SSD"；且 `ik` 后端是否能接受
稀疏布局尚未验证。属于改代码，不是改配置。

磁盘侧（顺带）：141 个容器只有 1 个在跑，**16.4 GB 是可回收的退出容器**；250 G 盘剩 72 G。

### 7.18 `--max-num-batched-tokens 4096 → 8192`：实测变慢，已回滚（2026-11-14 20:25）

**动机**：22k prompt 在 4096 下要切 6 个 chunk，每个 chunk 一次完整 5 段流水；8192 只需 3 个。

**实验**：`run-pp5-bt8.sh`，与生产脚本逐字相同，只改 3 处（容器名 / 端口 9013 / batch 8192）。

**结果：全项变慢。**

| 指标 | 4096（生产 fast+k5） | 8192 | 变化 |
|---|---:|---:|---:|
| **C2 22k TTFT** | **9.40 s** | **11.37 s** | **+21%（更差）** |
| C1 decode（400 tok） | — | 63.03 | ITL 39.9 ms |
| C3 三路并发 | — | 31.33 /路（aggregate 53.04） | ITL 77 ms |
| code_python | 100.1 | 93.79 | −6.3% |
| json_struct | 85.4 | 83.45 | −2.3% |
| zh_tech | 65.2 | 62.58 | −4.0% |
| math_steps | 79.1 | 73.22 | −7.4% |

**根因（启动日志直接给出）**：

```
batch 4096: coverage=93.96% (7216/7680, 128.xx GiB)
batch 8192: coverage=88.54% (6800/7680, 119.06 GiB)   ← 少了 416 个 GPU 槽位
```

激活缓冲按 `max_num_batched_tokens` 缩放，8192 比 4096 多吃约 **7.4 GiB 显存**，
正好等于 416 个专家槽（416 × 17.9 MiB = 7.3 GiB）。少掉的槽位全部落到 CPU：

- prefill 会路由到每层全部 384 个专家 → **CPU 上的专家从 464 个涨到 880 个（+90%）**，
  而 prefill 的瓶颈正是 PP4 的 CPU 专家计算 → TTFT +21%。
- decode 的 GPU 命中率下降 → 每步多几次主机往返 → 各用例 −2~7%。

**结论：这个杠杆方向是反的。** 4096 下省下的 3 次 chunk 启动（每次几十 ms），
远不如 7.4 GiB 显存换来的 416 个 GPU 专家值钱。**不要再试 8192/16384。**
真要压 TTFT 只有两条路：加 GPU 显存（不可能）或让 engram 别读盘（§7.17）。

### 7.19 那 134.5 GiB 主机专家副本：是硬结构，删不掉（2026-11-14 20:20，代码核对）

上一节曾说"主机副本里约 126 GiB 是纯冗余，可以释放"。**核对代码后作废：做不到。**

1. 主机 bank 由 IK 后端按 `hf.n_routed_experts = 384` 一次性分配（`cpu_moe.py:105`
   `CPUMXFP4Experts(config, hf.n_routed_experts, ...)`），Python 侧没有"只分配一部分"的开关。
2. **`prepare()` 有硬校验**（`cpu_mxfp4.py:558`）：

   ```python
   def prepare(self) -> None:
       if len(self._loaded) != self.num_experts * 3:
           raise RuntimeError("CPU expert weights are incomplete")
   ```

   384 × 3 个投影必须全部 `load_expert()` 过一遍，少一个直接拒绝启动。
3. 主机副本还承担两件事，不只是"备份"：
   - `GPUExpertCache(weight_source=self.backend.export_expert)` —— GPU 缓存 miss 时**从这里导出**，
     不退化成读盘；
   - `stage_expert_r8()` —— R8 staging 的源。
4. `CheckpointPageReclaimer` 回收的是 **safetensors 文件页**（`madvise(fd, MADV_DONTNEED)`，
   只对 `/proc/self/maps` 里 `.safetensors` 段生效），回收不了这块匿名 bank。

**所以要释放它必须改 IK 的 C 后端**（bank 尺寸 + 专家 id → 槽位重映射 + 路由改写），
不是配置项。风险远大于收益，且释放后 engram 也用不上（本次要求不动 engram）。

**已做且安全的清理**：`docker container prune -f` 删掉 136 Exited + 4 Created 容器，
回收 **16.43 GB 磁盘**（盘 167G→151G used，可用 72G→87G），内存零影响，可随时重建。

**主机内存的最终账**（247 GiB 总）：

| 项 | 大小 | 能否释放 |
|---|---:|---|
| PP4：20 层 × 384 专家 IK bank | ~138 GiB | ✗ 硬结构（见上） |
| PP4：host LRU（`host_cache_gib=8`） | 7.7 GiB | 可调小，但会拖慢 miss |
| PP4：engram 行缓存（`engram_cache_gib=4`） | ~2 GiB | 本次不动 engram |
| PP0~PP3 各 | 2.5~4.7 GiB | ✗ 必需 |
| 页缓存（模型 safetensors） | ~62 GiB | 可 drop，但无收益（engram 走 O_DIRECT） |
