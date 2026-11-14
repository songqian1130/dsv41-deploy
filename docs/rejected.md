# 已否掉的方向（不要再试）

按"能省多少时间"排序。每条都写清**为什么否**，避免下轮重踩。

---

## 1. `max_num_batched_tokens` 8192 / 16384

- **实测**：单流 93.79 / 83.45 / 62.58 / 73.22（均值 78.3，比 4096 低 2~7%）；22k TTFT 9.40 → 11.37 s（+21%）
- **机制**：每 +4096 batch token ≈ 7.4 GiB 显存 ≈ 416 个专家槽位。覆盖率 93.96% → 88.54%
- **结论**：省下的 3 次 chunk 启动 ≪ 丢掉的显存。**4096 是甜点。**
- ⚠️ 但注意：这次是**用单流标准否决的**。4 路 × 2.5k 上下文的 TTFT 9.6 s 恰恰是被 4096 串行化卡住的，
  **并发场景下可能是净赚**，值得用 v1.1 标准重测一次（D2 那一栏）。

## 2. decode CPU 线程 8 / 20 / 28

- **实测**：4→8 步时无变化，CPU 白烧一倍；16→20（PP4）配对差 −3.5%~+6.1%，无一致方向
- **机制**：decode 的真实并行度只有 ~0.8~1 核（MoE 每次调用几十行数据），4 线程已留 4 倍余量
- **结论**：PP5 用 4，PP4 用 16（因为 CPU 路由量涨 25 倍）。**再往上不加 token。**
- 反例警戒：`lscpu` 显示的 `scaling MHz: 41%` 是**空闲读数**，不是降频。别拿它当证据。

## 3. 覆盖率补丁（`bank_size - 1` → `bank_size`）

- **实测**：93.96% → 95.10%，速度变化在噪声内
- **机制**：38 ms 是 GPU 流水线地板，CPU 专家不在关键路径上
- **结论**：不值得为 1.1 个百分点动代码。

## 4. Humming / Marlin 加速

- `humming-kernels==0.1.12` 镜像里**已经装了**，不是缺依赖
- `--linear-backend humming` 是**非法值**：`config/kernel.py:187` 的 16 个 `LinearBackend` 里没有 humming
  （只有 `MoEBackend` 有这个值，而本模型不走通用 `FusedMoE`）
- 专家路径写死：`models/deepseek_v4_1/expert_cache.py:645` 无条件 `fused_marlin_moe()`
- 硬件：sm80（GA100）**没有 FP8 tensor core**，`humming/tune/sm8x.py` 里 `b4_allowed_dtypes=[int4]`、
  `b8=[int8]`，无 fp4/fp8 → sm80 上 Humming ≡ int4→BF16，就是 Marlin
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` 也无效：`marlin_moe.py:152/216` 硬编码 `use_atomic_false`，
  dense 侧 `marlin_utils.py:638` 对 `cap<9 && bf16` 直接 return False

## 5. 把 PP4 段那 ~138 GiB 主机专家 bank "释放"出来

- 曾经判断"126 GiB 是冗余，可释放" → **错误**
- `cpu_moe.py:105` 按 `n_routed_experts=384` 全量分配；`cpu_mxfp4.py:558` `prepare()` 硬校验
  `len(_loaded) != num_experts*3 → RuntimeError`
- 它还兼两职：GPU miss 的导出源、`stage_expert_r8` 的 R8 staging 源
- `CheckpointPageReclaimer` 只 madvise safetensors **文件页**，回收不了这块匿名 bank
- **只能改 IK C 后端**（bank 尺寸 + id 重映射），不是配置项

## 6. 旧引擎两条路（已弃用但留档）

| 模式 | 说明 | 为什么弃 |
|---|---|---|
| `legacy` | 旧引擎 V1 + 惰性接收 + engram SSD | 41.8 tok/s，比新引擎慢一倍 |
| `ram` | 同上，engram 全放 pinned 内存（183 GiB） | 与 ComfyUI / PP4 主机 bank 冲突；decode 无收益 |

`VLLM_PP5_LAZY_RECEIVE=1` 还和 `VLLM_USE_V2_MODEL_RUNNER=1` **互斥**（`gpu_worker.py` 里 assert），
而 dspark 只支持 V2。写新脚本时默认 V2，别顺手加惰性接收。

## 7. dspark k=1 / k=2 / k=3

- `hybrid.py:149`：`dspark_num_query_tokens = max(5, k)` —— block 下限本来就是 5
- k<5 等于白扔 query 位，k=3 → k=5 实测全项 +6~12.5%，**零显存成本**
- 不要指望 k 再往上：k=5 → 更大时验证 token 变多，GPU 每步要多算，不是白送

---

## 8. KV cache 卸载到主机内存 / 磁盘（想「多留几个会话不重算」）

- **想法**：空闲会话的 KV 存到 RAM 或 NVMe，下次带同样历史回来直接拉回显存，跳过 prefill。
- **引擎现状：没有这个能力。** 镜像里没有 LMCache；三个 `run.sh` 里没有任何 KV 卸载参数；
  唯一带 offload 字样的 `VLLM_CPU_OFFLOAD_GB_PER_RANK` 下放的是**专家权重**；
  `engram` 也是权重（两层各 93.75 GiB），不是 KV。要用得先接 vLLM 的 `KVConnector` 接口。
- **关键纠正：卸载不会「支持更多会话」。** 显存池还是 **22.37 GiB**，同时能**驻留**的会话数不变。
  它只是二级缓存，改变的是「换入」的成本，不是容量。
- **在这台机器上收益还要打折**：瓶颈是 **PCIe Gen2 x4 ≈ 1.6 GB/s**，而且这根总线同时被
  MoE 的 CPU 专家占着（decode 每步都在走）。

| 会话长度 | KV 大小 | 从盘/内存拉回（1.6 GB/s） | 直接重算 |
|---:|---:|---:|---:|
| 2k | 0.35 GiB | 0.23 s | 2.5 s |
| 8k | 1.40 GiB | 0.94 s | 3.9 s |
| 32k | 5.59 GiB | 3.75 s | 9.1 s |
| 64k | 11.19 GiB | 7.51 s | 16.1 s |

- **表里没算的两笔代价**：拉 5.6 GiB 会把 PCIe 占满 3.75 s，同一时刻正在跑的 4 路 decode 会被拖住；
  写回盘还要再花约 1.7 s（NVMe 3.56 GB/s）。盘空间按 100 个 32k 会话算是 560 GiB。
- **结论**：墙钟上确实省（约 2.4 倍），但省的是「换入时间」，不是「容量」。
  盘后端在这台机器上尤其没意义 —— NVMe 比 PCIe 快，瓶颈在总线不在盘。
  要做的最小验证：接 LMCache 的 CPU 后端 + 8k 会话回放，先在方案 B 上试（有余量）。

## 9. fp8_ds_mla KV 格式给新引擎（V2）的 PP4/PP5 用

- **想法**：KV 字节数减半（576B vs 1152B/token/层），1M 池挤掉的专家槽从 ~580 降到 ~290。
- **实测（2026-11-15 01:45，B [6,6,7,21] bf16→fp8）**：启动即死 ——
  `NotImplementedError: Compressed-KV source language_model.model.layers.20.attn not found
  on this rank; PP splits inside a v4.1 kv-sharing group are not supported.`
- **根因**：V4.1 有 compressed-KV 共享组（多层的 KV 存在组内 source 层上），fp8_ds_mla 布局
  要求 source 层与它的 consumer 层**同 rank**；[6,6,7,21] 把 layer 20 所在的组切在了 rank 边界。
  按共享组结构推断，[5,5,5,4,21]（A/C 拓扑）同样切组，**V2 上 PP 分层基本都无法对齐**。
- **旧引擎 C 为什么能用**：V1 的 fp8_ds_mla 实现没有这个 source-rank 检查。
- **要启用只能**：重排 PP 分层对齐共享组边界（动 [5,5,5,4,21] 的硬约束，风险未知）或
  改 V2 源码放宽检查（改引擎，超出调参范围）。
- **结论**：V2 上放弃 fp8 KV，用 bf16。1M 池 bf16 的代价 = 挤 ~580 专家槽（A：覆盖
  93.96%→~86%；B：56.25%→~48%），decode −5~10%。**KV 显存账全部按 bf16 口径重算。**

## 一句话总结

**这台机器的速度上限由三步决定**：decode 线程数（4）、dspark k（5）、KV 池大小（131072）。
其余参数调来调去都在 ±5% 噪声内。**别再在噪声里找突破。**
