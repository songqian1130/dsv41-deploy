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

## 一句话总结

**这台机器的速度上限由三步决定**：decode 线程数（4）、dspark k（5）、KV 池大小（131072）。
其余参数调来调去都在 ±5% 噪声内。**别再在噪声里找突破。**
