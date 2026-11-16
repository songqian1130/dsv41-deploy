# 方案 L · PP4 CPU-MoE（zebgop 版）—— 原版权重 + 9 层专家搬 CPU + engram 走 SSD

> 4 卡 PP4 `8,12,8,12` · **V1 runner** · CPU 专家 `16-19,35-39`（16 线程 AVX2）· engram 189 GiB 走 NVMe
> **单流 zh 26.4 tok/s（初测 1 轮）· GPU3 空出给 ComfyUI · 7:50 起服务 → ⚠️ 有条件可用（不崩优先）**

2026-11-16 照抄 [zebgop-ops/dsv41flash-pp](https://github.com/zebgop-ops/dsv41flash-pp) 原版路线。
选它的唯一理由：**GPU4 那个独立根端口在"PCIe 搬运 + GPU 计算重叠"时会整机冻结**（见 [`docs/GPU4-HANG.md`](../../docs/GPU4-HANG.md)），
而这版把 GPU4 降级成只承载 PP rank 的小包流量，是唯一实测不崩的组合。

代价写在 B 段：速度只有方案 B（PP4 t16）的 43%，只有方案 A 的 32%。

---

## A 元信息

| 项 | 值 |
|---|---|
| 日期 / 编号 | 2026-11-16 · L（PP4-CPUMOE） |
| 引擎 | **旧引擎 V1**（`VLLM_USE_V2_MODEL_RUNNER=0`）· 镜像 `vllm/vllm-openai:deepseekv41-flash-0909` · ID `sha256:00d577a6a632…`（30.4 GB）· vLLM `0.1.dev20904+g179dd0fa9` · 源码 [`zebgop-ops/dsv41flash-pp`](https://github.com/zebgop-ops/dsv41flash-pp) @ `e29e9fdc9bbc419b34301fc581c2da9f9062164a` · 补丁 `overlay/` 37 个 `patch_*.py` + `hybrid/cpu_moe.cpp` + `engram_ssd/row_store.cpp` · 作者 **zebgop-ops，Apache-2.0** · 恢复路径：**rebuild**（`overlay/apply-overlay.sh`）+ **load**（镜像本机已有） |
| 模型 | 1017 原版 ☑（`quantization=deepseek_v4_fp8`，原生 FP8，未改权重、未二次量化） |
| 用例 | **未按 cases_v1 跑**（初测：单轮 zh，200 tok；并发取自 `~/manju/engine_busy.log`） |
| 硬件 | **GPU 0,1,2,4**（GPU3 空出给 ComfyUI）· GPU4 **不是 0 MiB**（承载 PP3，57.3 GiB）· 主机内存 103 GiB used / 247 GiB |
| 启动参数差异 | `pipeline_layers=8,12,8,12` / `cpu_threads=16` / `spec=ngram k=3` / `kv 131072` / `batch 2048` / `seqs 4` / `util 0.97` / `engram=ssd` |
| 脚本 | 启动 `run.sh`（本目录）｜ 回滚 `./deploy.sh pp4-cpumoe --stop`，改回 `./deploy.sh pp5-fast` |

> **与方案 B 的区别不是"更快/更慢"，是"谁在算专家"**：B 用 GPU 装 21 层专家（覆盖率 56%），
> L 把 9 层专家直接丢给 CPU（AVX2 手写 kernel），GPU 显存只装权重和 KV。

---

## B decode 单流（tok/s，N 轮均值）

| 用例 | 均值 | 区间 | ITL ms | acc/draft |
|---|---:|---|---:|---:|
| code_python | 未测 | — | — | — |
| json_struct | 未测 | — | — | — |
| zh_tech | **26.4**（1 轮，非 3 轮均值） | — | — | — |
| math_steps | 未测 | — | — | — |
| 四档均值 | 未测 | — | — | — |
| C1_高熵（最差情形） | 未测 | — | — | — |

> ⚠️ **26.4 不能和 A 的 82.5 / B 的 60.9 直接相减**：那是四档三轮均值，这是 zh 单轮。
> 按 A/B 的 zh 档（65.2 / 47.7）对照，量级上大约是它们的 40% / 55%。要正式横向比必须先跑 `bench/run_cases.py`。

---

## C 首字与 prefill

| 项 | 值 |
|---|---|
| 短请求 TTFT（<100 tok） | **14.85 s**（44 tok prompt，含 CPU 专家冷页 + engram 首次寻址） |
| 22k 冷 TTFT | 未测 |
| 22k 重复文本 TTFT（engram 命中） | 未测 |
| **prefill 斜率** | 未测 |
| 固定开销（截距） | 未测 |

> TTFT 是这个方案最痛的地方：哪怕 44 token 的短请求也要 14.85 s。原因不在 GPU，在
> 首 token 之前要把 9 层专家的路由表从 NVMe 预热、engram 两张表首次 `pread`。

---

## D 并发（以 4 路为主）

D1 稳态扩展性

| 并发 | 每路 tok/s | 稳态聚合 | ITL ms | TTFT max | preempt | KV 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 1（单流基准） | 26.4 | 26.4 | — | 14.85 | 0 | — |
| 4 | 未测 | 未测 | — | — | — | — |
| **6**（实测，含 ComfyUI 并发） | **5.0 ~ 5.5** | **~31** | — | — | 0 | — |

> 6 路时每路只剩 5 tok/s，聚合 31 ≈ 单流 26.4。**这版几乎不随并发扩展**——
> 瓶颈是 CPU 上那 16 个 AVX2 线程和 NVMe 的 engram 读带宽，不是 GPU。
> `max_num_seqs=4` 理论上限制到 4 路，实测 6 路仍能提交（部分排队）。

D2 真实 4 路（每路输入 ___ tok、输出 ___ tok，含 prefill 排队）

| 聚合 tok/s | 墙钟 | TTFT max | ITL 均值 | preempt | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 未测 | — | — | — | — | — |

---

## E 容量与上下文

| 项 | 值 |
|---|---|
| max_model_len | **131,072**（128k，脚本默认值，未改） |
| KV 池（聚合 token / GiB） | 每 rank 可用 **0.96 GiB**（日志 `Available KV cache memory: 0.96 GiB`），聚合容量未测 |
| max_num_seqs / max_num_batched_tokens | 4 / 2,048 |
| 4 路均分每路可用上下文 | 未测（受 0.96 GiB/rank 限制，远小于 32k） |
| 不 preempt 的会话 × 上下文上限 | 未测 |
| 实测越界点 | 未测 |
| 越界后的后果 | 未测 |

> 这一版**不是为长上下文设计的**：engram 两表 189 GiB 走盘，KV 只剩不到 1 GiB/rank。
> 要 512k 请回方案 A/B。

---

## F 资源与代价

- 显存：GPU0 **61.7** / GPU1 **62.1** / GPU2 **61.0** / GPU3 **0（空出）** / GPU4 **57.3** GiB
- 专家：9/30 层在 CPU（`16-19,35-39`）→ GPU 覆盖率理论上限 **70%**（21×256/7680），实测未测
- 相对基准：单流 **≈−60%**（对 A）｜ prefill 未测 ｜ TTFT **+14 s 量级** ｜ 并发 未测
- CPU：16 个 AVX2 线程跑 CPU 专家；V1 runner 4 个 `VLLM::Worker_PP` **空转也吃 11.6 核**（自旋）
- 主机内存：103 GiB used（engram 不在内存，省下 ~190 GiB）
- **服务启动时长：7 分 50 秒**（03:18:37 容器首行 → 03:26:27 `Application startup complete`；其中权重 58.42 GiB / 252 s）

---

## G 判定与边界

- 验收线：**不崩** → ✅ 通过（引擎 6 路满载 + ComfyUI H3 124 帧并发，机器存活）
- 适用：**GPU4 存在时必须上生产的兜底**、单流问答、不需要长上下文的场景
- 不适用：长文 prefill（TTFT 15 s 起）、高并发（6 路聚合仅 31 tok/s）、需要 512k 上下文
- 已证伪、别再试：`DSV41_CPU_EXPERT_THREADS` 从 16 调到 28 未见收益（未做配对复测，勿引用）｜
  `DSV41_ENGRAM_STORAGE=host` 需要 ~190 GiB 内存，本机 247 GiB 装不下
- 回滚：`./deploy.sh pp4-cpumoe --stop` → `./deploy.sh pp5-fast`

**共存约束（2026-11-16 定）**：ComfyUI 必须锁 GPU3（`CUDA_VISIBLE_DEVICES=3`），且 GPU3 功耗墙必须
提到 250 W（150 W 会把 SM 压到 1035 MHz，H3 124 帧从 128 s 退化到 185 s）。
