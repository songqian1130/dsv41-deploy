# DSV41 部署方案 — DeepSeek-V4.1-Flash-1017 × 5×CMP 170HX

一台 5 卡 64 GB 的推理服务器，把**原版 DeepSeek-V4.1-Flash-1017**（未改权重、未量化）跑到可用速度。
三个可部署方案，每个目录自带 `deploy.sh`，换台机器拉下来就能起服务。

原则：**模型不动，引擎和参数可以随便试。** 所有数字都是服务端实测，未测的写「未测」，不做估算。

👉 **先看对比页：[`index.html`](index.html)**（本地双击即可打开，无外部依赖）

| | [A · pp5-fast](schemes/pp5-fast/) | [B · pp4-t16](schemes/pp4-t16/) | [C · pp5-legacy](schemes/pp5-legacy/) |
|---|---|---|---|
| 引擎 | 新引擎 V2 · dspark k=5 | 新引擎 V2 · dspark k=5 | 旧引擎 V1 · 无投机 |
| 卡 | 5 | 4（GPU4 空出） | 5 |
| 单流 tok/s | **82.5** | 60.9 | 43.5 |
| 4 路聚合 | **135.6** | 未测 | 43.9（3 路，不扩展） |
| 22k TTFT | 9.40 s | 13.71 s | 6.45 s（ram） |
| 启动 | 8:57 | ~9 min | **4:30** |
| 判定 | ✅ 生产 | ⚠️ 有条件 | 🕘 兜底 |

---

## 30 秒开始

```bash
git clone <this-repo> dsv41-deploy && cd dsv41-deploy

./deploy.sh pp5-fast                  # 5 卡生产，单流 82.5 tok/s
./deploy.sh pp4-t16                   # 4 卡，GPU4 留给 ComfyUI
./deploy.sh pp5-legacy --engram ram   # 旧引擎，22k TTFT 6.45 s

./deploy.sh pp5-fast --dry-run        # 只打印 docker 命令，不启动
./deploy.sh pp5-fast --stop           # 停掉
```

`deploy.sh` 会依次做：预检（docker / nvidia runtime / 卡数 / 模型目录 / 镜像 / 内存）→ 起容器 →
轮询 `/health`（最长 30 分钟，容器中途死掉会直接打日志）→ 打印端点和一条 curl 示例。

模型路径默认 `/run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official`，别处跑要指定：

```bash
./deploy.sh pp5-fast --model /data/DeepSeek-V4.1-Flash-1017-official --port 9010
```

---

## 引擎：**上游地址 + 我们的补丁**，就这两样（260 KB）

三个方案跑的是两个第三方引擎镜像。把它们造出来的东西全在 `engine/`，**一共 260 KB**，不用存镜像。

```
上游地址（别人给的 URL）  +  我们的 .patch（3 个文件，在仓库里）  =  能重建出同一个引擎
```

### 外部文件就两类，一眼能分

| | 有哪些 | 它们的「地址」怎么写 |
|---|---|---|
| **别人写的**（原封不动复制） | 4 个，外加 1 份同内容副本 | 直接贴 URL，点开就能下 |
| **我们改的** | 3 个补丁 + 1 个本机镜像 + 1 份反推的 Containerfile | 上游 commit 的 URL + 仓库里这份 `.patch` |

**别人写的**（2026-11-14 实测 HTTP 200，sha256 与本地逐字节相同）：

| 文件 | 从哪下 |
|---|---|
| `0001-prime-pp-communicators-before-model-load.patch` | [aiultrayang @ `29b03af5`](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx) → `pp6-ssd/patches/` |
| `0002-engram-ssd-port.patch` | 同上（**上游文件名没有 `0002-` 前缀**，照我们的名字拼 URL 会 404） |
| `Containerfile` | 同上 → `pp6-ssd/build/Containerfile`，一字未改 |
| `x99-stable.patch` | [LocalFrameLab](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99) → `patches/x99-stable.patch`（Apache-2.0） |
| `0001-host-staging-and-cache-interval.patch` | **副本**：就是上一行那个文件（只差 23 字节 patch 元数据，代码逐行相同） |

**我们改的** —— 上游从来没发布过，所以**没有第三方 URL，这是正常的，不是丢件**：

| 文件 | 基座（这个有 URL） | 我们改了什么 |
|---|---|---|
| `0003-dspark-construction-offload.patch` | `344303947/dsv41-flash-pp5-170hx` @ `d63af5a` | dspark 构造阶段 UVA 下放（2026-10-31 r11 实验，**没进镜像**） |
| `0004-runtime-cumulative-snapshot.patch` | 同上 | 11-02 构建工作区快照 = 0001+0002+0003+0005，**用它重建会得到另一个引擎** |
| `0005-pp5-lazy-receive.patch` | 同上 + 0001 + 0002 | `gpu_worker.py` +37/−9，惰性接收 |
| 方案 C 的 `local/` 镜像 | 上面三个补丁 build 出来的 | 本机构建、从未推送，**不需要地址** |
| `v2-dspark/Containerfile` | 镜像层历史 | 没有 Dockerfile 可抄，**按 `docker history` 反推**的配方 |

拿到「基座 commit + 这个 patch」就能重建，这两样都在。完整 URL、逐项 sha256、再生命令在
**[`engine/LINKS.md`](engine/LINKS.md)**，一条命令复验：

```bash
./tools/verify-sources.sh     # OK / DRIFT / GONE / NET，任何一条变了都会报出来
```

### 47.6 GB 里，只有 191 MB 是真正不可再生的

| 组成 | 大小 | 要存吗 |
|---|---:|---|
| CUDA 基础镜像 + torch + 194 个包 + 源码树 | 47.4 GB | ❌ 记得住地址就能重下（版本已 freeze） |
| 我们改的两个补丁（`0003` + `0005`） | 10 KB | ✅ 已在 `engine/` |
| **5 个编译产物 `.so`** | **191 MB** | ⚠️ **唯一不可再生**，待导 GitHub Release |

所以按手上的条件选一条，**不是二选一**：

| 你手上有什么 | 怎么办 | 体积 |
|---|---|---|
| 有外网 | `engine/v2-dspark/build.sh` 或 `engine/v1-legacy/build.sh`（主路径，40~90 分钟） | 260 KB |
| 还想沿用历史基准 | 重建后把那 5 个 `.so` 覆盖回去 | +191 MB |
| 完全没外网 | `tools/engine-bundle.sh save <引擎>`（应急，永不进 git） | 20~25 GB |

```bash
# 有外网（主路径）：克隆 → 打补丁 → 编译 → 自检
engine/v2-dspark/build.sh

# 只导 / 只补那 5 个 .so
tools/engine-bundle.sh save-so v2-dspark --out /big/disk
tools/engine-bundle.sh load-so /big/disk/v2-dspark-so-*.tar.zst --into <重建出的镜像>
```

⚠️ `docker pull 18gogogo/...` 只在**验证过仓库公开**时可用（本机 Docker Hub 不通，未验证）；
方案 C 的镜像是 `local/` 前缀，从未推送，那条路对它无效。模型权重也**不在镜像里**（475 GiB，自己准备）。

逐项改动清单（改几处、编什么、装什么、哪些已验证）见 **[`engine/REBUILD.md`](engine/REBUILD.md)**，
来源与风险清单见 **[`engine/README.md`](engine/README.md)**。

---

## 致谢与归属：引擎是上游的，方案和数字是我们的

本仓库做三件事：**参数化、实测、把方案落成可复现的目录**。引擎代码基本都不是我们写的，
这一点必须说清楚，否则既误导读者，也不尊重原作者。

| 组成 | 作者 / 来源 | 许可 | 我们的处理 |
|---|---|---|---|
| vLLM 推理引擎 | [vLLM 项目](https://github.com/vllm-project/vllm) | Apache-2.0 | 未改 |
| 模型权重 | DeepSeek [`DeepSeek-V4.1-Flash-1017`](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017) | MIT | 未改（官方原版，未量化） |
| `deepseek_v4_1` 模型实现（CPU/GPU 混合推理） | [`buliaoyin/vllm-backport`](https://github.com/buliaoyin/vllm-backport) @ `c3d3104de`，分支 `codex/deepseek-v41-cpu-hybrid`（= `sm80-optimize`，同一 HEAD） | 见上游 | 未改 |
| **x99 部署补丁 + 4 卡启动参数 + Decode/Prefill/TTFT 基准** | **[LocalFrameLab/deepseek-v41-cmp170hx-x99](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99)** | **Apache-2.0** | **逐行原样使用，一字未改**（核对见 [`PROVENANCE.md`](engine/v2-dspark/PROVENANCE.md)） |
| 4 卡部署经验 | [aiultrayang/deepseek-v41-flash-cmp170hx](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx) | 见上游 | 未改 |
| 旧引擎 5 卡 PP 方案 | [`344303947/dsv41-flash-pp5-170hx`](https://github.com/344303947/dsv41-flash-pp5-170hx) @ `d63af5a` + [aiultrayang](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx) 的 0001/0002 | 见上游 | 加了 **`0005`**（`gpu_worker.py` +37/−9，惰性接收）和 **`0003`**（r11 构造 UVA 实验，未进镜像）—— **这两个是我们自己写的**，见 [`engine/v1-legacy/SOURCE.md`](engine/v1-legacy/SOURCE.md) |
| Docker 镜像打包 | [`18gogogo/170hx4-256r-dsv41-1017-flash:sm80`](https://hub.docker.com/r/18gogogo/170hx4-256r-dsv41-1017-flash)（是否公开未验证） | — | 未改，原样使用 |

**我们真正做的是这四件：**

1. **把 4 卡方案改成 5 卡**：PP 分区 `[5,5,5,4,21]`、`cpu_threads [28,4]`、dspark `k=3→5`、
   batch 4096 的显存换算（每 4096 token ≈ 7.4 GiB ≈ 416 专家槽）。
2. **实测**：四档 decode、22k 冷 TTFT、4 路并发、KV 容量上限、启动时长 —— 全部同批 nonce 配对、
   丢冷启动首轮、带区间。
3. **把「引擎怎么造出来的」钉死**：上游地址 + 我们的补丁 + freeze + Containerfile + 能跑完的 `build.sh`。
4. **两个反直觉发现**：decode 的 CPU 线程数 28→4 反而快 45~86%（barrier 固定开销）；
   `max_num_batched_tokens` 8192 会让 coverage 掉 5.4 个点、22k TTFT 反而变慢 21%。

一句话：**引擎与补丁归上游，参数验证与可复现性归本仓库。**

### 一句话分账

- **A/B 那 74 行补丁** = LocalFrameLab 的 `x99-stable.patch`（Apache-2.0），**逐行原样，一字未改**。
- **旧引擎里我们只改了两处**：`0005`（惰性接收，进了镜像）、`0003`（r11 实验，没进镜像）。
- **其余全部原封不动**。逐项分类见上面「一共 9 个外部文件」表，逐项地址见 [`engine/LINKS.md`](engine/LINKS.md)。

> 引用上游补丁的 Apache-2.0 许可副本：`engine/v2-dspark/patches/LICENSE`。
> 逐项版权与许可（含「上游未声明许可」的那两个补丁怎么处理）见 **[`NOTICE.md`](NOTICE.md)**。

## 目录

```
dsv41-deploy/
├── index.html                ★ 方案对比页（由 tools/build_index.py 生成）
├── SCHEMES.md                全部方案的参数 / 实测 / 判定，含已否方向
├── deploy.sh                 一键部署入口（分发到 schemes/<方案>/deploy.sh）
├── schemes/
│   ├── README.md             三个方案怎么选
│   ├── pp5-fast/             方案 A：5 卡生产
│   │   ├── README.md         方案卡：引擎 / 前置 / 部署 / 性能六段 / 排障
│   │   ├── deploy.sh         一键部署
│   │   ├── run.sh            只做 docker run，参数可覆盖
│   │   └── meta.json         数据源，供 index.html 生成
│   ├── pp4-t16/              方案 B：4 卡，GPU4 留给 ComfyUI
│   └── pp5-legacy/           方案 C：V1 旧引擎（SSD / RAM 两种 engram）
├── bench/                    基准与探针
│   ├── cases_v1.json         冻结用例集（有 sha256 校验，别改）
│   ├── run_cases.py          单流四档 + C1/C2/C3
│   ├── probe_capacity.py     并发 / prefill / 容量
│   └── results|capacity.jsonl  原始结果
├── engine/                   ★ 引擎配方（260 KB）：镜像 99.6% 是公开资源，只有 191 MB .so 不可再生
│   ├── README.md             体积拆解 / 三种还原法 / 来源 / 补丁 / 还缺什么
│   ├── LINKS.md              ★ 外部依赖清单：每个 commit / raw URL / digest / revision + sha256
│   ├── REBUILD.md            ★ 从基础镜像+源码重建：逐项改动清单与验证状态
│   ├── v2-dspark/            A/B 用：fork commit + 74 行补丁 + Containerfile + freeze + build.sh
│   └── v1-legacy/            C 用：5 个补丁（3 个进镜像）+ Containerfile + build.sh
├── docs/
│   ├── SCHEME-CARD.md        方案输出标准 v1.1（写新方案必须按它）
│   ├── rejected.md           已否掉的 6 个方向：为什么否，精确到代码行号
│   └── studies/              深度材料（基准推导 / dspark 研究 / PP4 推导 / 运维）
├── tools/
│   ├── build_index.py        读 meta.json → 生成 index.html
│   ├── index.template.html   对比页模板
│   ├── engine-bundle.sh      save-so/load-so（191 MB 编译产物）· save/load（整包应急）· verify
│   └── verify-sources.sh     逐个复验 LINKS.md 里的地址：OK / DRIFT / GONE / NET
├── variants/                 试过但没采用的配置（留档，别删）
└── archive/                  2026-10-28 ~ 11-05 的旧时代材料，只读，已被 .gitignore 排除
```

---

## 机器与模型

| 项 | 值 |
|---|---|
| GPU | 5 × CMP 170HX，64 GB HBM2e/卡，**PCIe Gen2 x4**，无 NVLink |
| CPU / 内存 | 2 × Xeon E5-2680v4（28 物理核 / 56 线程）· 247 GiB（实为 4 通道） |
| 系统 | Ubuntu 26.04.1 · 驱动 610.57.04 · Docker 29.1.3 |
| 模型 | DeepSeek-V4.1-Flash-1017，**475.2 GiB** / 48 分片（ModelScope rev `f04a9786`），40 层全 MoE，每层 384 专家 |
| 引擎 | `18gogogo/170hx4-256r-dsv41-1017-flash:sm80`（vLLM 0.1.dev21380，V2 runner，fork 自 `buliaoyin/vllm-backport`）· 旧方案用本机构建的 V1 镜像 |
| 关键约束 | 40 层里 19~39 层必须放最后一段流水线；20~39 层专家可下放 CPU |

---

## 口径（为什么这些数字能并排比）

1. 单流 tok/s = 服务端 `completion_tokens ÷ (总时长 − TTFT)`；四档均值是 4 个 real 用例的算术平均。
2. 同一配置三轮之间的极差通常 **5~7%**，差值 <5% 一律记「噪声内」，不写「提升」。
3. 跨方案比较必须**同一批 nonce**；不同批 nonce 的绝对值不能直接相减。
4. 冷启动首轮丢弃；不许用高熵用例单独下结论。
5. 代价必须写：GPU4 是否被占、主机内存吃了多少、启动多久。

```bash
# 测一次单流四档（换 nonce，别复用）
python3 bench/run_cases.py <label> <nonce> --set real

# 4 路并发 / prefill 斜率 / 容量速算
python3 bench/probe_capacity.py conc    --label <label> --conc 4 --ctx 200 --out 300
python3 bench/probe_capacity.py prefill --label <label> --sizes 20,80,240
python3 bench/probe_capacity.py table   --kv-tokens 131072 --seqs 4
```

---

## 名词

| 词 | 意思 |
|---|---|
| **PP5 / PP4** | 流水线并行 5 段 / 4 段（不是 5 卡跑 5 份模型，是切层） |
| **V2 / V1 引擎** | 新引擎带 `deepseek_v41_hybrid` 与 CPU 专家下放；旧引擎是 V1 runner + 惰性接收 |
| **dspark k** | 投机解码每步猜几个 token（k=5，接受率 0.5~0.77） |
| **覆盖率** | 7680 个专家槽位里多少个常驻 GPU（93.96% = 7216） |
| **GPU 命中率** | 每个 token 路由到的专家，实际在卡上的比例（99.5%） |
| **engram** | 模型权重里那张 189 GiB 的 n-gram 表，可走 SSD 或 pinned 内存 |
| **噪声内** | 同一配置不同 nonce 之间的波动（5~7%），小于它就是没差别 |

---

## 更新对比页

改了方案的 `meta.json`（或加了新方案目录）之后：

```bash
python3 tools/build_index.py     # 重新生成 index.html
```

不要手改 `index.html`，它每次都会被覆盖。
