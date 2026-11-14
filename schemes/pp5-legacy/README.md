# 方案 C · 旧 PP5（V1 引擎 · 无 dspark）—— 上一代生产

> 5 卡 PP5 + 默认 `[8,8,8,8,8]` + 专家 **100% 在卡** + engram SSD 或 RAM
> **单流 43.5 tok/s（四档均值）· 3 路聚合 43.9（不扩展）· 启动 4:30 → 🕘 已被方案 A 取代**

2026-11-14 12:00 之前的线上方案，也是"稳态 40 多 tok/s"那个数。
它和方案 A/B **不是同一代引擎**：没有投机解码、没有 CPU 专家下放，参数体系完全不同。

**还留着的理由**：启动只要 4 分半（方案 A 要 9 分钟）、`engram=ram` 时 22k TTFT 6.45 s 比方案 A 还快 31%、
速度与内容无关永不抖。当兜底和对照仍有价值。

---

## 引擎

| 项 | 值 |
|---|---|
| 引擎家族 | **旧引擎 V1**（V1 Model Runner + 惰性接收 `VLLM_PP5_LAZY_RECEIVE=1`） |
| 镜像 | `local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102` |
| 镜像 ID | `sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1`（29.7 GB，校验用） |
| 源码 | [`github.com/344303947/dsv41-flash-pp5-170hx`](https://github.com/344303947/dsv41-flash-pp5-170hx) @ `d63af5a472` —— 第三方 fork |
| 所用补丁 | **重建只用 3 个**：`0001` + `0002`（[aiultrayang](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx)，3.3 + 39 KB）+ `0005`（**我们写的**，4.2 KB，`gpu_worker.py` +37/−9）。<br>`0003` 是没进镜像的旧实验、`0004` 是含 `0003` 的旧快照 —— **这两个重建时都用不到**，留在仓库里只作记录 |
| 运行时 | vLLM 0.13.0 基线（`lazymio/vllm-backport@sha256:bfe237ee…`）+ 编译 `libdsv41_cuda.so` |
| 投机解码 | **无**（每步 1 个 token，因此快慢只由步时决定） |
| 是否改源码 | **是**：主体是 [aiultrayang](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx) 的 0001/0002 + O_DIRECT/Linux AIO 的 C++ 扩展；**我们加的是 `0005`（+37/−9，惰性接收）** |
| 模型 | DeepSeek-V4.1-Flash-1017 官方原版（镜像标签 `local.deepseek.checkpoint=official-1017-unchanged`） |
| 可移植性 | ⚠️ 镜像本身只在原机（`local/` 从未推送），但**源码 + 4 个 patch + Containerfile 全在仓库里**，可重建 |
| 详细来源 | [`engine/v1-legacy/SOURCE.md`](../../engine/v1-legacy/SOURCE.md) · 全部外部地址与 sha256：[`engine/LINKS.md`](../../engine/LINKS.md) |

### 换机器怎么拿到镜像

这个镜像**没有** registry 可拉，但**不等于复现不了** —— 它的完整配方在
`engine/v1-legacy/`（156 KB）：fork 地址 + commit + 4 个 patch + 两阶段 Containerfile。

```bash
# 路径 1（主路径，需要 GitHub 可达）：从源码重建
./engine/v1-legacy/build.sh            # 取 d63af5a + 应用 0001/0002/0005 + 编译 C++ 扩展

# 路径 2（仅无外网时）：离线整包，带 sha256 与镜像 ID 校验
./tools/engine-bundle.sh save v1-legacy --out /path/to/big/disk   # 29.7 GB → 约 12~15 GB
./tools/engine-bundle.sh load /path/to/big/disk/v1-legacy-*.tar.zst
```

手工 `docker save | gzip` 也行，但记得自己比对镜像 ID：
`sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1`。

重建配方在 2026-11-14 夜间做过逐文件复核：`d63af5a` + 0001 + 0002 + 0005 能复现镜像内 9 个
Python 文件的 git blob；同时发现 0003 从未进过镜像，旧的「只应用 0004」写法会造出另一个引擎。
但 `build.sh` 仍未在干净机器上端到端跑过 —— 见 [`engine/README.md`](../../engine/README.md) 第七节。
**这也是它被方案 A 取代的现实原因之一**（新引擎至少有公开的基础镜像和明确的依赖 freeze）。

---

## 前置条件

| 项 | 要求 | 原机实测 |
|---|---|---|
| GPU | 5 张，每张 ≥64 GB，sm80 | 5 × CMP 170HX |
| 主机内存 | SSD 模式 40 GiB / **RAM 模式 190 GiB pinned** | 247 GiB |
| 磁盘 | 镜像 ~28 GB + 权重 269 GB | — |
| 软件 | Docker + nvidia-container-toolkit（同方案 A） | 见文末 |

> `engram=ram` 会 `cudaHostAlloc(Mapped)` 掉 189 GiB **pinned** 内存，机器上基本不能再跑别的吃内存的东西。
> 它和方案 A 的 138 GiB 专家 bank 是同一批内存，**不能同时开**。

---

## 部署

### 一键

```bash
cd dsv41-deploy

./deploy.sh pp5-legacy                    # engram 走 SSD（默认，4:30 启动）
./deploy.sh pp5-legacy --engram ram       # engram 全放 pinned 内存（22k TTFT 6.45 s）
./deploy.sh pp5-legacy --dry-run
./deploy.sh pp5-legacy --stop
```

首次启动：SSD 模式约 **4 分 30 秒**，RAM 模式约 **11 分钟**（慢在 189 GiB pinned 分配）。

### 两种模式的区别只有一行

| | `--engram ssd`（默认） | `--engram ram` |
|---|---|---|
| engram 存放 | SSD + 8 GiB 缓存 | `cudaHostAlloc(Mapped)` 189 GiB |
| 22k 冷 TTFT | 11.9 s（客户端口径） | **6.45 s** |
| decode tok/s | 43.5（decode 不读 engram，两者相同） | 43.5 |
| 主机内存 | ~30 GiB | **189 GiB pinned** |

> SSD 模式慢在 `O_DIRECT` 的 4 KB 页对齐：每行只有 258 B，实测 **19~24 倍读放大**。
> ram 模式用 UVA 按 256B gather，绕开了页税。

### 验证

```bash
curl -s http://127.0.0.1:9010/health
curl -s http://127.0.0.1:9010/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash-1017","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

---

## 性能（方案卡 v1.1）

**A 元信息**

| 项 | 值 |
|---|---|
| 日期 / 编号 | 2026-11-14 11:12（ram）/ 更早（legacy） |
| 引擎 | 旧引擎 V1 · 本地镜像 · 有补丁 |
| 模型 | 1017 原版 ☑ |
| 用例 | `cases_v1` nonce `run0061`（服务端口径）｜ legacy 为客户端 chunk 口径 |
| 硬件 | GPU 0-4 全占 · 主机内存 30 / 189 GiB |
| 启动参数 | `[8,8,8,8,8]` · 无 `cpu_threads` · 无 dspark · `max_model_len 262144` · batch 2048 · util 0.97 |
| 脚本 | `run.sh` ｜ `deploy.sh` |

**B decode 单流（tok/s）**

| 用例 | legacy（SSD） | ram | ITL ms |
|---|---:|---:|---:|
| code_python | 未测（客户端 41.8） | **43.46** | 23.1 |
| json_struct | 未测（≈43 推定） | **43.56** | 23.0 |
| zh_tech | 未测（≈43 推定） | **43.42** | 23.1 |
| math_steps | 未测（≈43 推定） | **43.45** | 23.1 |
| **四档均值** | — | **43.5** | 23.1 |
| C1_高熵 | — | 43.18 | 23.2 |

> **无投机 → 速度与内容无关**：四档全部落在 43.4~43.6，换成高熵随机汉字还是 43.18。
> 这是它唯一比新引擎"干净"的地方：不用考虑接受率，也没有内容相关的抖动。
> `acc_per_draft` 为空是正常的（未开 spec decode）。

**C 首字与 prefill**

| 项 | legacy | ram |
|---|---:|---:|
| 短请求 TTFT | 0.74 s | 0.33~0.47 s |
| 22k 冷 TTFT | **11.92 s**（客户端口径） | **6.45 s** |
| 22k 重复文本 | 未测 | 未测 |
| prefill 斜率 | 未测 | 未测 |
| 固定开销 | 未测 | 未测 |

> prefill 斜率当时没建探针，只有 TTFT 绝对值；不能拿 TTFT÷token 反推斜率（会把固定开销算进分子）。

**D 并发**

| 并发 | 每路 tok/s | 聚合 | ITL ms | 备注 |
|---|---:|---:|---:|---|
| 1 | 43.2 | 43.2 | 23.2 | |
| 3 | 21.65 | **43.92** | 46.3 | **聚合零增长**，GPU 已饱和 |
| 4 | 未测 | 未测 | 未测 | `max_num_seqs=4` 允许，但按 3 路趋势也不会涨 |

> 这是旧引擎最致命的一条：**并发完全不换吞吐，只是把同一块蛋糕切小**。
> 方案 A 4 路聚合 135.6（单流 97.3 的 1.4 倍），同样 5 卡，多出的 3 倍吞吐就来自这里。

**E 容量与上下文**

| 项 | 值 |
|---|---|
| max_model_len | **1,048,576**（1M；2026-11-15 起，未验证。此前 262,144） |
| KV 池 | **显式 `--kv-cache-tokens=1048576`**（2026-11-15 起，未验证。上游 9010 原始启动器默认即 1M） |
| max_num_seqs / batched_tokens | 4 / **2,048** |
| 并发会话槽上限 | **4**（配置值）。实际 3 路已经零扩展，4 路未测 |
| 4 路均分每路上下文 | 未测 |
| 可保住的空闲会话数 | 未测（池大小未知） |
| 越界点 | 未测 |

> `max_model_len` 是**单请求上限**，不等于 KV 池；旧引擎的真实池子没量过，别拿 256k × 4 去规划。

> **并发在本方案上不成立**：实测 3 路聚合 43.92 tok/s，单流 43.5，几乎相等 —— 加会话不加吞吐。
> 所以「几个会话槽」对它意义不大，它只适合单线程用。

**F 资源与代价**

- 显存：5 卡，每卡 8 层专家，`util 0.97`，**无 CPU 下放**（`VLLM_CPU_OFFLOAD_GB_PER_RANK=0,0,0,0,0`）
- 专家：等效覆盖率 **100%**（不存在 cache miss，因为没有主机侧专家 bank）
- 主机内存：SSD 模式 ~30 GiB；RAM 模式 **189 GiB pinned**
- **服务启动时长：4 分 30 秒（SSD）/ 11 分（RAM）**
- CPU：几乎不参与推理 —— CPU 专家池那套机制在旧引擎里根本不存在

**G 判定与边界**

- 验收线：单流 ≥40 tok/s → 当时通过；已被方案 A 以 **+90%** 取代
- 唯一还领先的项：启动快 4 分半、`ram` 模式 22k TTFT 6.45 s、ITL 23 ms（但每步只出 1 个 token）
- 不适用：任何需要并发的场景（3 路就零增长）
- 已证伪：旧引擎加 dspark（显存不够，k=1 也 OOM，见 `../../docs/studies/PP5-DSPARK-STUDY.md` §5.1）

---

## 已知问题

| 现象 | 原因 / 处理 |
|---|---|
| `docker pull` 失败 / 提示 not found | 镜像是本地构建的，见上面「换机器怎么拿到镜像」 |
| 启动即报 `Unknown vLLM environment variable: VLLM_PP5_LAZY_RECEIVE` | 这只是 warning，旧镜像用 `os.getenv` 直接读，不影响启动 |
| RAM 模式起不来 / 被 OOM killer 杀 | 189 GiB pinned 必须一次性拿到；`free -g` 确认 available ≥200，且没有别的容器占内存 |
| 参数里写了 `serve` 反而报错 | 旧镜像 ENTRYPOINT 已经是 `[vllm, serve]`，直接跟模型路径 |
| SSD 模式 22k TTFT 特别慢 | `O_DIRECT` 读放大，属正常；换 `--engram ram` 或接受它 |

---

## 已验证环境

| 项 | 值 |
|---|---|
| OS | Ubuntu 26.04.1 LTS · kernel 7.0.0-38-generic |
| 驱动 | 610.57.04 |
| Docker | 29.1.3 |
| GPU | 5 × CMP 170HX 64 GB |
| CPU / 内存 | 2 × E5-2680 v4（28 核）· 247 GiB |

---

## 文件

| 文件 | 作用 |
|---|---|
| `deploy.sh` | 一键部署，支持 `--engram ssd\|ram` |
| `run.sh` | `docker run`，两种模式共用，只改一个字段 |
| `meta.json` | 数据源，供根目录 `index.html` 生成 |
| `README.md` | 本文件 |
