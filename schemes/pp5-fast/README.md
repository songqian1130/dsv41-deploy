# 方案 A · PP5 fast —— 5 卡生产基线

> 5 卡 PP5 `[5,5,5,4,21]` + dspark k=5 + `cpu_threads:[28,4]` + engram SSD
> **单流 82.5 tok/s（四档均值）· 4 路并发聚合 135.6 · 启动 8:57 → ✅ 生产基线**

三个方案里最快的，也是唯一把 5 张卡都吃满的。要跑 ComfyUI 就换 [方案 B](../pp4-t16/)。

---

## 引擎

这是决定性能的那一行 —— 同一个模型，换一代引擎差 90%。

| 项 | 值 |
|---|---|
| 引擎家族 | **新引擎 V2**（`deepseek_v41_hybrid`，3220 行自定义实现） |
| 镜像 | [`18gogogo/170hx4-256r-dsv41-1017-flash:sm80`](https://hub.docker.com/r/18gogogo/170hx4-256r-dsv41-1017-flash) |
| 镜像 ID | `sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157`（47.6 GB，校验用） |
| 源码 | [`github.com/buliaoyin/vllm-backport`](https://github.com/buliaoyin/vllm-backport) @ `c3d3104de0` —— **第三方 fork，不是官方 vLLM** |
| 所用补丁 | `0001-host-staging-and-cache-interval.patch`（8.3 KB，3 文件 +74/−5）—— **[LocalFrameLab](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99) 的 [`x99-stable.patch`](https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch)，Apache-2.0，我们逐行未改**，核对见 [`PROVENANCE.md`](../../engine/v2-dspark/PROVENANCE.md) |
| vLLM | `0.1.dev21380` · V2 Model Runner |
| 投机解码 | **dspark k=5** + `enable_adaptive_verification` |
| 是否改源码 | **是**（上游改的，非我们）：3 文件 74 行 = LocalFrameLab 的 `x99-stable.patch`；缺了它 `DSV41_HOST_STAGING_CACHE` / `cache_decode_interval` 会被静默忽略 |
| 模型 | DeepSeek-V4.1-Flash-1017 官方原版（未改、未量化，权重 475 GiB） |
| 可移植性 | ⚠️ pull 未验证（本机连不上 Docker Hub）；**重建不依赖镜像**：`engine/v2-dspark/build.sh`（260 KB 配方 + 外网，40~90 分钟） |
| 详细来源 | [`engine/v2-dspark/SOURCE.md`](../../engine/v2-dspark/SOURCE.md) · 全部外部地址与 sha256：[`engine/LINKS.md`](../../engine/LINKS.md) |

---

## 前置条件

| 项 | 要求 | 原机实测 |
|---|---|---|
| GPU | 5 张，每张 ≥64 GB，**sm80 及以上** | 5 × CMP 170HX 64 GB |
| 主机内存 | **≥200 GiB**（20 层专家 bank 常驻 138 GiB） | 247 GiB |
| 物理核 | ≥28（decode 用 4，prefill 用 28） | 28 核 / 56 线程 |
| 磁盘 | 镜像 ~30 GB + 权重 269 GB + 编译缓存 ~5 GB | — |
| 软件 | Docker 20.10+ · NVIDIA 驱动 535+ · nvidia-container-toolkit | 见文末 |
| 模型 | HF 格式目录，含 `config.json` + `*.safetensors` | — |

> 卡必须是 sm80（A100/GA100 系）。sm90 的卡能用但没验证过；消费级 24 GB 卡装不下（单层专家 6.76 GiB）。

---

## 部署

### 一键

```bash
cd dsv41-deploy
./deploy.sh pp5-fast
```

默认模型路径是原机的 `/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official`，别处跑要指定：

```bash
./deploy.sh pp5-fast --model /data/DeepSeek-V4.1-Flash-1017-official --port 9010
# 或者用环境变量
DSV41_MODEL=/data/... DSV41_PORT=9010 ./deploy.sh pp5-fast
```

### 它做了什么

1. **预检**：docker 可用、nvidia runtime 已注册、卡数 ≥5、模型目录像 HF 权重、镜像在不在本机（不在就 pull）、可用内存是否够
2. **起容器**：见 `run.sh`，就是把生产参数原样交给 docker
3. **等健康**：轮询 `http://127.0.0.1:9010/health`，最长 30 分钟；容器中途退出会直接打印最后 40 行日志并失败
4. **打印端点**：给出 `v1` 地址、模型名、日志/停止命令和一条 curl 示例

首次启动约 **8 分 57 秒**（权重 3:56 → engine init 4:54 → 起服务 7 s），慢在每层 5 秒的专家缓存校准，属正常。

### 只想看看要跑什么

```bash
./deploy.sh pp5-fast --dry-run
```

打印完整 docker 命令，不碰系统。

### 验证

```bash
curl -s http://127.0.0.1:9010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017",
       "messages":[{"role":"user","content":"用一句话说明 MoE 是什么"}],
       "max_tokens":128}'
```

### 停止

```bash
./deploy.sh pp5-fast --stop
# 或
docker rm -f dsv41-pp5-fast
```

---

## 性能（方案卡 v1.1）

**A 元信息**

| 项 | 值 |
|---|---|
| 日期 / 编号 | 2026-11-14 · PP5-BASELINE |
| 引擎 | 新引擎 V2 · 镜像 `18gogogo/...:sm80` · 未改源码 |
| 模型 | 1017 原版 ☑（未改权重、未量化） |
| 用例 | cases_v1 · sha `ee98f898` · 3 轮 · nonce 51/52/53 |
| 硬件 | GPU 0-4 全占 · 主机内存 ~190 GiB |
| 启动参数 | `[5,5,5,4,21]` / `[28,4]` / k=5 / kv 1048576（11-15 起，待验证） / batch 4096 / max-num-seqs 4 |
| 脚本 | `run.sh` ｜ 部署 `deploy.sh` ｜ 停止 `deploy.sh --stop` |

**B decode 单流（tok/s）**

| 用例 | 均值 | ITL ms | acc/draft |
|---|---:|---:|---:|
| code_python | **100.1** | 46.3 | 0.753 |
| json_struct | 85.4 | 43.9 | 0.648 |
| zh_tech | 65.2 | 40.2 | 0.509 |
| math_steps | 79.1 | 40.5 | 0.498 |
| **四档均值** | **82.5** | 42.7 | — |
| C1_高熵 | 未测 | | |

> ITL / 接受率取自 22:13 的复测轮（同配置实测 97.3 / 83.9 / 63.4 / 71.1，与三轮均值在噪声内）。

**C 首字与 prefill**

| 项 | 值 |
|---|---|
| 短请求 TTFT | 0.43 ~ 1.32 s |
| 22k 冷 TTFT | **9.40 s** |
| 22k 重复文本 TTFT | 5.00 s（engram 命中高） |
| **prefill 斜率** | **4,683 tok/s** |
| 固定开销（截距） | 2.10 s |

> 回归三点：6,819 tok → 3.40 s ｜ 27,052 tok → 8.10 s ｜ 81,105 tok → 19.36 s。
> 用 TTFT÷token 会得 2,005~4,189 的假斜率，因为把 2.1 s 固定开销摊进了分子。

**D 并发**

D1 稳态扩展性（输入 ~290 tok，输出 300 tok）

| 并发 | 每路 tok/s | 稳态聚合 | ITL ms | TTFT max | preempt | KV 峰值 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 97.3 | 97.3 | 46.3 | 1.32 s | 0 | — |
| **4** | **33.9** | **135.6** | **29.6** | 1.42 s | 0 | 3.8% |

> 4 路时每步反而更快（ITL 46→30 ms），但每路只分到 1/3 的 token。
> 稳态聚合 136 是单流的 1.40 倍 —— batching 有收益，但不是线性。

D2 真实 4 路（每路输入 2.5k tok，输出 300 tok）

| 聚合 tok/s | 墙钟 | TTFT max | ITL 均值 | preempt | KV 峰值 |
|---:|---:|---:|---:|---:|---:|
| 62.4 | 19.2 s | **9.63 s** | 40.9 ms | 0 | 25.4% |

> **慢的不是算力，是排队。** `max_num_batched_tokens=4096` 把 4 路 × 2.5k 的 prefill 切成块串行，
> 后三路的 9.6 s 里大部分是在等前一路算完。

**E 容量与上下文**

> **🔄 2026-11-15 变更：KV 池 131,072 → **1,048,576**，`max_model_len` 同步提到 1,048,576（都未上机验证）。**
> 依据：上游 aiultrayang 的 PP4 参考配置就是 `kv_cache_tokens=524288 / max_model_len=524288`，
> 其启动日志显示 512k 池的 KV 显存合计只有 **~1.8 GiB**（各 rank `Hybrid cache budget` 行：
> 0.240 + 0.435 + 0.435 + 0.693 GiB），按此**外推 1M 池也只有 ~3.6 GiB**。
> 上限依据：`/model/config.json` 里 `text_config.max_position_embeddings = 1048576`（YaRN 从 64k 扩展），
> **1M 就是模型原生上限**；上游 344303947 的 9010 原始启动器默认值也正是 `MAXLEN=1048576`。
> **KV 不是这项变更的瓶颈，专家覆盖率才是** ——
> 上机验证看两处：① 各 rank `Hybrid cache budget: KV=` 行；② 覆盖率是否还在 93.96% 附近。
> 覆盖率明显下滑就先退 262,144。

| 项 | 新值（待验证） | 变更前（131,072 池，已实测） |
|---|---|---|
| max_model_len | **1,048,576**（1M，模型原生上限） | 131,072（128k） |
| KV 池 | **1,048,576 聚合 token**（≈3.6 GiB，外推） | 131,072 聚合 token（当时记「22.37 GiB」，与上游 1.8 GiB / 512k 对不上，**存疑待核**） |
| max_num_seqs / batched_tokens | 4 / 4,096（不变） | 4 / 4,096 |
| 4 路均分每路可用上下文 | **262,144**（1,048,576 ÷ 4，推算） | 32,768（刚好占满、零余量） |
| 各上下文下的安全会话数 | 8k→128 · 16k→64 · 32k→32 · **64k→16** · 128k→8 · 256k→4（推算） | 4k→4 · 8k→4 · 16k→4 · 32k→4 · 64k→2 · 128k→1 |
| 实测越界点 | 未测 | 4 × 37k（149,581 token）触顶 95.4%，**preempt 仍为 0** |
| 越界后果 | 未测 | prefill 被串行化：TTFT 9.5 → 36.4 s，聚合 62.4 → **1.7** tok/s（不报错，只是变慢） |

> **不随池大小变的结论**：容量是「总共 512k」不是「512k × 4」；`max_num_seqs=4` 是并发上限，
> 第 5 个请求**排队等位**、不淘汰别人；超池不报错，只是把后续 prefill 排队。
> **变化最大的**：变更前 4×37k（149k）就越界，新池下 149k 只占 14% —— 4 路长上下文终于放得下。
> 旧池的启动日志原文：`GPU KV cache token budget: 131,072 aggregate tokens for up to 4 requests`。

**能保住多少个「聊过一轮、过会再来」的会话**（按新池 1,048,576 推算）

| 每会话历史长度 | 池子能同时保住 | 被挤掉后重算的代价（TTFT） |
|---:|---:|---:|
| 2k | 256 个 | ≈ 2.5 s |
| 4k | 128 个 | ≈ 3.0 s |
| 8k | **128 个** | ≈ 3.9 s |
| 16k | 64 个 | ≈ 5.6 s |
| 32k | 32 个 | ≈ 9.1 s |
| 64k | 16 个 | ≈ 16.1 s |
| 128k | 8 个 | ≈ 30.1 s（外推） |
| 256k | 4 个 | ≈ 57 s（外推） |

> 换算 = `1,048,576 ÷ 每会话历史`；重算代价用实测口径 `TTFT = 2.103 s + token / 4,683`。
> ⚠️ **整列推算**，新池没跑过多轮会话回放。变更前（131,072 池）对应值：2k→64 · 8k→16 · 32k→4 · 64k→2。

三条机制（不随池大小变）：

1. **被挤的不是正在生成的那 4 个**。正在跑的块在用，vLLM 不会回收，LRU 优先淘汰空闲会话的块。
   所以「4 路跑得很快」和「第 N+1 个进来把老会话挤掉」**可以同时成立** —— 挤掉的是空闲的那个。
2. **共享的 system prompt 只存一份**。N 个会话用同一段 500 token 的系统提示，那 500 token 只占一份，
   各占各的只有后面的对话内容，所以实际能保住的会话往往比上表多。
3. **必须前缀完全一致才命中**。历史里改一个字，或者插入新消息导致前缀变了，命中就断在变化点。

> **能不能把 KV 存到硬盘、用时拉回来，免得重算？** 原理上可行（业界叫 KV offload / LMCache），
> 但这套引擎**没有这个能力**，而且它**不会增加同时驻留的会话数** —— 显存池多大就住多少，
> 它只把「换入」的成本从重算改成传输。这台机器受 PCIe Gen2 x4 ≈ 1.6 GB/s 限制，
> 32k 会话拉回 3.75 s vs 重算 9.1 s，只便宜 2.4 倍，还要跟 MoE 抢总线。
> 完整账见 [`docs/rejected.md`](../../docs/rejected.md) 第 8 条。

**F 资源与代价**

- 显存：GPU0 47.8 / GPU1 51.8 / GPU2 52.3 / GPU3 51.5 / GPU4 58.6 GiB
- 专家：覆盖率 **7216/7680 = 93.96%** · GPU 命中 99.5~99.73% · CPU 路由 0.27%
- CPU：PP4 decode 4 线程（~410%）· prefill 28 线程（峰值 18.95 核）
- 主机内存：~190 GiB（PP4 段 20 层专家 bank ≈ 138 GiB 常驻）
- **服务启动时长：8 分 57 秒**（权重+5 卡上下文 3:56 ｜ engine init 4:54 ｜ 起服务 7 s）

**G 判定与边界**

- 验收线：单流 ≥40 tok/s → **✅ 通过**，作为新一代基准
- 适用：单流长文生成、批量后台任务、4 路以内并发
- 不适用：4 路以上并发（`max_num_seqs=4` 硬限制）· 需要 GPU4 的场景（换方案 B）
- 已证伪：batch 8192 · decode 8 线程 · `[28,28]` · 覆盖率补丁 · Humming/Marlin
- 回滚：`./deploy.sh pp5-fast --stop`

---

## 已知问题与排障

| 现象 | 原因 / 处理 |
|---|---|
| 卡在 `Loading safetensors` 很久 | 正常。权重 269 GB，首次加载 4 分钟；`docker logs -f` 看得到进度 |
| `engine init` 阶段每层停 5 秒 | 正常。这是专家缓存校准，20 层 ≈ 100 秒 |
| 启动即 OOM | 主机内存不足。`[28,4]` 这套需要约 190 GiB；先 `free -g` 确认 available ≥200 |
| `nvidia-container-toolkit` 未注册 | `docker info --format '{{json .Runtimes}}'` 里没有 `nvidia` 就重装 toolkit 并 `systemctl restart docker` |
| 端口被占 | `--port 9011`，同时改客户端 |
| 4 路并发时 TTFT 涨到 9.6 s | 已知限制，不是故障。`batch 4096` 在串行化 prefill；见 `variants/run-pp5-bt8.sh`（试过，单流会掉 2~7%） |

---

## 已验证环境

| 项 | 值 |
|---|---|
| OS | Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic |
| 驱动 | 610.57.04 |
| Docker | 29.1.3 |
| GPU | 5 × CMP 170HX 64 GB（PCIe Gen2 x4，无 P2P） |
| CPU / 内存 | 2 × E5-2680 v4（28 核 / 56 线程）· 247 GiB |

---

## 文件

| 文件 | 作用 |
|---|---|
| `deploy.sh` | 一键部署：预检 → 起容器 → 等健康 → 打印端点 |
| `run.sh` | 只负责 `docker run`，参数可用环境变量覆盖 |
| `meta.json` | 数据源，`tools/build_index.py` 用它生成根目录 `index.html` |
| `README.md` | 本文件 |
