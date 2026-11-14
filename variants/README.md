# variants —— 试过但没采用的配置

这里只留**脚本原件**，每个都能起，但都因为实测更差或物理不可行被否掉。
留着是为了下次别再花一晚上重试。判定理由的完整版在 [`../docs/rejected.md`](../docs/rejected.md)。

| 脚本 | 是什么 | 结论 |
|---|---|---|
| `run-pp5-k3.sh` | 方案 A 的上一版，dspark **k=3** | 被 k=5 取代：`hybrid.py:149` 的 query block 下限本来就是 5，k=3 白丢两个位置；k=5 全项 +6~12.5%，**零显存成本** |
| `run-pp4-t20.sh` | 方案 B 的 `cpu_threads [28,20]` | 与 16 平局（60.7 vs 60.9），逐 nonce 配对差 −3.5%~+6.1% 无方向，ITL 还更差 |
| `run-pp5-bt8.sh` | `max_num_batched_tokens 8192` | 每 +4096 batch token ≈ 7.4 GiB 显存 ≈ 416 个专家槽位，覆盖率 93.96%→88.54%，单流 −2~7%、22k TTFT +21% |
| `run-pp5-cov.sh` | 覆盖率补丁（`bank_size-1` → `bank_size`） | 覆盖率 93.96%→95.10%，速度在噪声内；38 ms 是 GPU 流水线地板，CPU 专家不在关键路径 |
| `run-pp5-dspbal.sh` | dspark 权重分布调整实验 | 未带来收益，显存账没变 |
| `run-pp5-dspkv.sh` | dspark + KV 池调整 | 同上，KV 已不是瓶颈 |
| `run-pp5-dsplite.sh` | 轻量 dspark 变体 | 同上 |
| `run-pp5-old-dspark.sh` | 旧引擎强行上 dspark | **物理不可行**：5 卡 util 0.97 只剩 1.93 GiB，dspark 固定要 7.39 GiB（3 个 MTP 层，与 k 无关），k=1 也 OOM |

> 这些脚本是按原机路径写死的（`/run/media/xpz/...`），只作留档，不适合直接拿去部署。
> 要部署请用 `schemes/` 下的三个方案。
