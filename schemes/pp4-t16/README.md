# 方案 B · PP4 t16 —— 4 卡，GPU4 留给 ComfyUI

> 4 卡 PP4 `[6,6,7,21]` + `cpu_threads:[28,16]` + dspark k=5
> **单流 60.9 tok/s（四档均值）· GPU4 全程 0 MiB · 22k TTFT 13.71 s → ⚠️ 有条件可用**

由方案 A 直接派生：同镜像、同权重、同投机参数，只改分区、卡数和 CPU 线程数。
**用 26% 的单流速度，换出一整张 64 GB 的卡。**

---

## 引擎

| 项 | 值 |
|---|---|
| 引擎家族 | **新引擎 V2**（与方案 A 同代、同镜像） |
| 镜像 | [`18gogogo/170hx4-256r-dsv41-1017-flash:sm80`](https://hub.docker.com/r/18gogogo/170hx4-256r-dsv41-1017-flash) |
| 镜像 ID | `sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157`（与 A 完全相同，47.6 GB） |
| 源码 | [`github.com/buliaoyin/vllm-backport`](https://github.com/buliaoyin/vllm-backport) @ `c3d3104de0` —— 第三方 fork |
| 所用补丁 | 同 A：`engine/v2-dspark/patches/0001-*.patch` —— [LocalFrameLab](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99) 的 [`x99-stable.patch`](https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch)，Apache-2.0，逐行未改 |
| vLLM | `0.1.dev21380` · V2 Model Runner |
| 投机解码 | dspark k=5 + `enable_adaptive_verification` |
| 是否改源码 | **否（本方案没动）**：镜像与 A 逐字节相同，只改分区 `[6,6,7,21]` 与 `cpu_threads [28,16]` |
| 模型 | DeepSeek-V4.1-Flash-1017 官方原版 |
| 可移植性 | ⚠️ 同 A，pull 未验证；重建走 `engine/v2-dspark/build.sh` |
| 详细来源 | [`engine/v2-dspark/SOURCE.md`](../../engine/v2-dspark/SOURCE.md) · 全部外部地址与 sha256：[`engine/LINKS.md`](../../engine/LINKS.md) |

---

## 前置条件

| 项 | 要求 | 原机实测 |
|---|---|---|
| GPU | 4 张，每张 ≥64 GB，sm80 及以上 | 4 × CMP 170HX（GPU0-3） |
| 主机内存 | **≥160 GiB**（20 层 CPU 专家 bank 138 GiB） | 247 GiB |
| 物理核 | ≥28（decode 用 16） | 28 核 |
| 磁盘 | 同方案 A（镜像 + 权重共用） | — |
| 软件 | 同方案 A | 见文末 |

> **与方案 A 不能同时跑**：抢同一批卡和同一份主机内存。切换必须先停一个。

---

## 部署

### 一键

```bash
cd dsv41-deploy
./deploy.sh pp4-t16                      # 端口默认 9015，可与别的东西共存
./deploy.sh pp4-t16 --model /data/DeepSeek-V4.1-Flash-1017-official
./deploy.sh pp4-t16 --gpus 0,1,2,3       # 默认就是这四张
./deploy.sh pp4-t16 --dry-run            # 只打印命令
./deploy.sh pp4-t16 --stop
```

启动后 GPU4 应该是 **0 MiB**，这是这个方案唯一的产出，务必确认：

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

### 与方案 A 的差异（只这 4 处）

| 参数 | 方案 A | 方案 B |
|---|---|---|
| `--pipeline-parallel-size` | 5 | **4** |
| `CUDA_VISIBLE_DEVICES` | 0,1,2,3,4 | **0,1,2,3** |
| `pipeline_layers` | `[5,5,5,4,21]` | **`[6,6,7,21]`** |
| `cpu_threads` | `[28,4]` | **`[28,16]`** |

> 分区为什么不能是 `[8,8,3,21]`：`hybrid.py:101` 硬约束「19~39 层必须在最后一段」，
> 前 3 段合计只能 19 层；`[8,8,3,21]` 会让单卡专家 54.1 GiB + KV + 激活 > 64 GB 装不下。
> 完整推导见 [`../../docs/studies/PP4-MODE.md`](../../docs/studies/PP4-MODE.md)。

> 为什么 decode 线程从 4 提到 16：少一张卡后专家驻留率 93.96% → 56.25%，
> CPU 路由占比 0.27% → **6.8%**（25 倍），4 线程会被打满成为瓶颈。
> 16 与 20 做过配对复测，差异 −1.1%~+0.8% 属噪声，故选 16（ITL 更低、CPU 更省）。

---

## 性能（方案卡 v1.1）

**A 元信息**

| 项 | 值 |
|---|---|
| 日期 / 编号 | 2026-11-14 · PP4-MODE |
| 引擎 | 新引擎 V2 · 未改源码 |
| 模型 | 1017 原版 ☑ |
| 用例 | cases_v1 · sha `ee98f898` · 3 轮 · nonce **101/102/103**（与 t20 配对） |
| 硬件 | GPU 0-3 · **GPU4 = 0 MiB** · 主机内存 ~150 GiB |
| 启动参数 | `[6,6,7,21]` / `[28,16]` / k=5 / kv 1048576（11-15 起，待验证） / batch 4096 / PP=4 |
| 脚本 | `run.sh` ｜ `deploy.sh` |

**B decode 单流（tok/s，三轮均值）**

| 用例 | 均值 | 区间 | ITL ms | acc/draft |
|---|---:|---|---:|---:|
| code_python | **74.6** | 71.4~76.6 | 61.7 | 0.768 |
| json_struct | 64.4 | 62.2~66.3 | 54.8 | 0.645 |
| zh_tech | 47.7 | 45.9~49.7 | 51.0 | 0.476 |
| math_steps | 56.8 | 53.7~60.2 | 52.0 | 0.513 |
| **四档均值** | **60.9** | ±2.3 | 54.9 | — |
| C1_高熵 | 51.6 | 单轮 | — | — |

> 保留率：相对方案 A 每档都掉 24~28%，**不是某一档塌了**，是整体等比下降。

**C 首字与 prefill**

| 项 | 值 |
|---|---|
| 短请求 TTFT | 0.79 s |
| 22k 冷 TTFT | **13.71 s**（方案 A 是 9.40，退 46%） |
| prefill 斜率 | 未测（v1.1 新增指标） |
| 固定开销 | 未测 |

> 22k 退 46% 的原因：prefill 会路由到每层全部 384 个专家，
> CPU 上的专家从 464 个涨到 3360 个，而 prefill 瓶颈正是 PP4 段的 CPU 计算。

**D 并发**

| 档 | 结果 |
|---|---|
| C3 三路并发 | 23.5 tok/s/流，ITL 99.6 ms |
| 4 路稳态 | 未测 |
| 4 路 × 2.5k | 未测 |

**E 容量与上下文**

> **🔄 2026-11-15 变更：KV 池 131,072 → **1,048,576**，`max_model_len` 同步（未上机验证）。**
> B 的形态（4 卡 PP4）和上游 aiultrayang 的参考配置**完全同构**，而那份参考跑的就是
> `kv_cache_tokens=524288 / max_model_len=524288 / max_num_seqs=8`，KV 显存合计 ~1.8 GiB ——
> 而**模型原生上限就是 1M**（`/model/config.json` `text_config.max_position_embeddings = 1048576`），
> 上游 344303947 的 9010 原始启动器（PP5、util 0.97）默认值也正是 `MAXLEN=1048576` —— 1M 是这套引擎的**常规操作**。
> KV 按上游 512k 实测 1.8 GiB 外推，1M 池也只有 **~3.6 GiB**，B 的显存完全装得下。
> 唯一没对齐的是 `max_num_seqs`（我们 4，参考 8）。

| 项 | 新值（待验证） |
|---|---|
| max_model_len | **1,048,576**（1M，模型原生上限） |
| KV 池 | **1,048,576 聚合 token**（`--kv-cache-tokens 1048576`，≈3.6 GiB 外推） |
| 并发会话槽上限 | **4**（`max_num_seqs=4`）。第 5 个排队等位，不淘汰已在跑的会话 |
| 4 路均分每路上下文 | **262,144**（1,048,576 ÷ 4，推算） |
| 可保住的空闲会话数 | 8k→128 · 16k→64 · 32k→32 · 64k→16（推算） |
| 实测 | ❌ **一档都没测**。要坐实必须在 B 上跑 `python3 bench/probe_capacity.py table` |

**上限：1M 就是顶，且 fp8 下装得下（2026-11-15 推算）**

模型硬顶：`/model/config.json` `text_config.max_position_embeddings = 1048576`，
再大引擎直接拒绝（`get_and_verify_max_len` 按模型上限校验），没有「超过 1M」的选项。

装不装得下，按 A（V2/PP5/131072 bf16）的启动实测
`Reserving [0.181, 0.327, 0.327, 0.147, 0.508] GiB KV per worker` 折算层单价，
换算到 B 的 [6,6,7,21] 拓扑（合计 1.49 GiB 与 A 吻合，交叉验证通过）：

| B 的 rank | 层 | bf16 @131072 | **fp8 @1M（最重档）** | 比 bf16@131072 多要 | 折合专家槽（17.9 MiB/个） |
|---|---|---:|---:|---:|---:|
| rank0 (L0-5) | 6 | 0.25 GiB | 0.99 GiB | +0.87 GiB | ~49 |
| rank1 (L6-11) | 6 | 0.39 GiB | 1.57 GiB | +1.37 GiB | ~78 |
| rank2 (L12-18) | 7 | 0.34 GiB | 1.37 GiB | +1.20 GiB | ~69 |
| rank3 (L19-39) | 21 | 0.51 GiB | **2.03 GiB** | +1.78 GiB | ~102 |

代价：合计挤掉 **~300 个专家槽**（约 4%），覆盖率 56.25% → **~52%**，
CPU 路由 6.8% → ~7.5%，decode 预计 **−1~2%**。B 本来就是低覆盖方案，
这些「挤掉的槽」全部落在 CPU 路由上，边际代价比 A 小得多。

对照：如果不加 fp8（bf16），1M 池最重的 rank3 要 **4.07 GiB**，
挤掉 ~570 槽（覆盖率 56% → ~48%）。fp8 把这笔代价砍半。

> ⚠️ 全表按启动参数推算。B 是新 V2 引擎 + PP4，**没有上机实测过**；
> 上面那份上游 PP4 参考是**旧引擎**（`kv_cache_dtype=fp8_ds_mla`、cpu_threads [28,24]），
> 它的 KV 显存数字不能直接搬到 B。

**1M 池的显存账（2026-11-15 按 V2 实测外推，未上机验证）**

A（V2、PP5、131072 池）的启动日志实测：`Reserving [0.181, 0.327, 0.327, 0.147, 0.508] GiB KV per worker`
—— 即 V2 每 131,072 token 的 KV 池共 **1.49 GiB**（40 层）。线性外推：

| 池大小 | KV 总量（40 层） | 相对 131072 增量 | 折合专家槽（17.9 MiB/个） |
|---|---:|---:|---:|
| 131,072（现状） | 1.49 GiB | — | — |
| 524,288 | 5.96 GiB | +4.5 GiB | ~250 个 |
| **1,048,576** | **11.92 GiB** | **+10.4 GiB** | **~580 个** |

**结论：装得下，但不是免费的。** KV 是先预留、专家缓存吃剩余（引擎源码
`kv_cache_utils.py` 的 `reserved_memory = min(memory, required)`），所以 1M 池
挤掉的是 **~580 个 GPU 专家槽**，专家覆盖率估计从 93.96% 掉到 **~86%**
（参照：batched 8192 实验覆盖率 93.96→88.54%，decode 掉 2~7%）。
decode 预计 **−5~10%**，prefill/TTFT 基本不受影响。

**两条安全网（源码已核实）：**

1. 启动时硬校验：`KV cache budget of X tokens requires Y GiB on worker N, but only Z GiB is available`
   —— 真不够会**直接报错拒绝启动**，不会静默挤爆或 OOM。
2. 图校准有 `required free` 检查（A 实测每卡要求 1.85~2.33 GiB 余量），不满足同样报错。

**想省一半代价**：给 run.sh 加 `--kv-cache-dtype fp8_ds_mla`（V2 已实现该布局，
`deepseek_v4_1/attention.py:113`，旧引擎 C 的默认值就是它），KV 字节数减半，
覆盖率损失从 ~580 槽降到 ~290 槽。
> 精度代价有多大：这是 **DeepSeek 官方的 KV 存储格式**（引擎日志会打
> `Using DeepSeek's fp8_ds_mla KV cache format`，UE8M0 分块缩放），不是民间压缩；
> 每层每 token 576B（512 维隐变量 + 64 维 rope，各 1 字节）vs bf16 的 1152B，正好减半。
> 方案 C（旧引擎）默认就是它，一直在生产上跑，没有异常记录。

**F 资源与代价**

- 显存：GPU0-3 用满 · **GPU4 = 0 MiB**（这是这个方案唯一的产出）
- 专家：覆盖率 **4320/7680 = 56.25%** · GPU 命中 93.18~94.61% · CPU 路由 **6.8%（方案 A 的 25 倍）**
- CPU：PP3（最后一段）decode 16 线程饱和
- 主机内存：~150 GiB（20 层 CPU 专家 bank 138 GiB）
- 服务启动时长：未测（约 9 分钟量级，未计时）

**G 判定与边界**

- 验收线：单流 ≥40 tok/s 且 GPU4 空闲 → **⚠️ 有条件通过**（60.9，达标）
- 适用：需要同时跑 ComfyUI；可接受 TTFT 13.7 s 的长文场景
- 不适用：prefill 密集（长文档、RAG 灌入）——TTFT 退 46% 主要打在这里
- 关键取舍：**用 26% 的单流速度换一张完整的 64 GB 卡**
- 已证伪：`cpu_threads [28,20]`（与 16 平局，ITL 更差）

---

## 待补

v1.1 标准新增的 D2/E/F 三项当时没测。补测约需：启动 9 分钟 + 测量 3 分钟。

```bash
cd bench
python3 probe_capacity.py prefill --label pp4t16 --sizes 20,80,240
python3 probe_capacity.py conc    --label pp4t16 --conc 4 --ctx 200  --out 300
python3 probe_capacity.py conc    --label pp4t16 --conc 4 --ctx 2000 --out 300
python3 probe_capacity.py table   --kv-tokens 1048576 --seqs 4
```

---

## 已验证环境

| 项 | 值 |
|---|---|
| OS | Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic |
| 驱动 | 610.57.04 |
| Docker | 29.1.3 |
| GPU | 4 × CMP 170HX 64 GB（GPU4 留空） |
| CPU / 内存 | 2 × E5-2680 v4（28 核）· 247 GiB |

---

## 文件

| 文件 | 作用 |
|---|---|
| `deploy.sh` | 一键部署（4 卡预检 + 等健康） |
| `run.sh` | `docker run`，默认端口 9015 |
| `meta.json` | 数据源，供根目录 `index.html` 生成 |
| `README.md` | 本文件 |
