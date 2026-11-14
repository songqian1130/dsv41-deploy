# 旧引擎 V1 · legacy（方案 C）

## 镜像

| 项 | 值 |
|---|---|
| tag | `local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102` |
| image ID | `sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1` |
| 体积 / 层数 | 29.7 GB / 123 层 |
| 构建时间 | 2026-11-02 14:07 (+08:00) |
| 基础镜像 | `lazymio/vllm-backport@sha256:bfe237ee0886d9b4b2357b84bfbe09805678cc80b2182dd2a87cefb89497342d`（= `:v0.13.0-sm80`，本机仍在） |
| 是否可拉 | ❌ **不可能**。`local/` 前缀，从未推送到任何 registry |
| 是否带 `.git` | ❌ 没有。源码是 `COPY vllm/` 覆盖进 site-packages 的，**无法自证 commit** |

镜像标签（构建时写入，`docker image inspect --format '{{json .Config.Labels}}'` 可核）：

```
io.canozziia.upstream=344303947/dsv41-flash-pp5-170hx@d63af5a
io.workbuddy.patch=prime-pp-communicators + engram-ssd-port
local.deepseek.checkpoint=official-1017-unchanged
local.deepseek.engram=aiultrayang-O_DIRECT-Linux-AIO
local.deepseek.pp=5
local.deepseek.pp-layer-partition=8,8,8,8,8
local.deepseek.pp5-performance=lazy-receive
local.deepseek.reference-commit=29b03af5eedfac9af12dfa14e8cfa7e2eafc852a
```

> `pp-layer-partition=8,8,8,8,8` 是**构建时的默认值**（40 层均分）。线上实际跑的是
> `schemes/pp5-legacy/run.sh` 传的 `--pipeline-parallel-config`，不是这个标签。

## 源码

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/344303947/dsv41-flash-pp5-170hx.git`（第三方 fork） |
| commit | `d63af5a472dc76b12d7a73d50a5af142844c15d1`（`dsv41-pp6: fix rank2 OOM, guard PP shape mismatch, add load progress`） |
| 补丁来源仓库 | [`aiultrayang/deepseek-v41-flash-cmp170hx`](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx) @ `29b03af5`，目录 `pp6-ssd/` |
| 构建方式 | 两阶段：先编 `libdsv41_cuda.so`，再把整棵 `vllm/` 覆盖进基础镜像 |

## 补丁（5 个入库；实际进镜像的只有 0001 + 0002 + 0005）

| 文件 | 大小 | sha256（前 12） | 进镜像 | 说明 |
|---|---:|---|---|---|
| `0001-prime-pp-communicators-before-model-load.patch` | 3.3 KB | `16d8b7554700` | ✅ | 加载前预热 PP 通信器（来自 aiultrayang） |
| `0002-engram-ssd-port.patch` | 39 KB | `5ec439e0d300` | ✅ | engram 后端移植到 SSD，O_DIRECT + Linux AIO（来自 aiultrayang） |
| `0005-pp5-lazy-receive.patch` | 4.2 KB | `30b53a49fd64` | ✅ | **本仓库写的**，只改 `gpu_worker.py`，+37/−9（5 个 hunk） |
| `0003-dspark-construction-offload.patch` | 5.8 KB | `57c0ef0e3808` | ❌ | **本仓库写的**：2026-10-31 的 r11 构造 UVA 实验（dspark 构造阶段下放），**不在本镜像** |
| `0004-runtime-cumulative-snapshot.patch` | 52 KB | `f8c6cb49b08a` | ❌ | 历史快照，**含 0003**，用它重建会得到另一个引擎 |

### 每个文件的出处（2026-11-14 实测 HTTP 200，拉下来 sha256 与归档逐字节一致）

| 文件 | 出处（完整地址） |
|---|---|
| `0001` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/patches/0001-prime-pp-communicators-before-model-load.patch> |
| `0002` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/patches/engram-ssd-port.patch> —— **上游文件名没有 `0002-` 前缀**，照我们的名字拼 URL 会 404 |
| `Containerfile` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/build/Containerfile> —— 我们**原样使用，一个字节没改**（`5af7be182795…`） |
| `0003` | **我们改的**（2026-10-31 r11 实验，上游只有 2 个补丁、没有它）。原件 `archive/legacy/patches/dspark-construction-offload.patch` sha256 相同；也可从 `0004` 按文件名抽出那 3 个 dspark 块，**实测 sha256 完全相同** → [`LINKS.md`](../LINKS.md) 2.1 |
| `0004` | **我们改的**（2026-11-02 构建工作区的工作树，= 0001+0002+0003+0005），只有仓库里这份，再生不了 |
| `0005` | **我们改的**。定义 = 镜像内 `gpu_worker.py` 减去 `d63af5a+0001+0002`，再生命令见 [`LINKS.md`](../LINKS.md) 2.1 |

全部地址、镜像 digest、模型 revision 与一键复验脚本：**[`engine/LINKS.md`](../LINKS.md)**（`./tools/verify-sources.sh`）。

### 2026-11-14 复核：镜像 = `d63af5a` + 0001 + 0002 + 0005

方法：从 GitHub 取 `d63af5a` 的 tarball → `git apply 0004` → 算每个文件的 **git blob**；
再 `docker run` 进镜像算 `/usr/local/lib/python3.12/dist-packages/vllm/` 下同 9 个文件的 blob。
blob 是内容哈希，与行尾、路径、工作区策略无关。

| 文件 | 镜像内 blob | `d63af5a+0004` | 结论 |
|---|---|---|---|
| `config/engram.py` | `5a180151` | `5a180151` | ✅ |
| `config/vllm.py` | `a65d074f` | `a65d074f` | ✅ |
| `models/deepseek_v4_1/common/engram.py` | `4ef765f9` | `4ef765f9` | ✅ |
| `models/deepseek_v4_1/common/engram_ssd.py` | `3aaffff1` | `3aaffff1` | ✅ |
| `models/deepseek_v4_1/host_memory.py` | `d35f525e` | `d35f525e` | ✅ |
| `v1/worker/gpu_worker.py` | `831a4f34` | `831a4f34` | ✅（0001+0005 结果） |
| `model_executor/models/qwen3_dspark.py` | `1ecdf5fa` | `ff2413d4` | ❌ 镜像 = **0003 的 pre-image** |
| `models/deepseek_v4_1/nvidia/dspark.py` | `c03b1760` | `361c0e6d` | ❌ 同上 |
| `v1/worker/gpu/spec_decode/dspark/utils.py` | `5b13530e` | `f5370e37` | ❌ 同上 |

三条结论：

1. **镜像里没有 0003。** 那 3 个文件的 blob 与 0003 的 pre-image 完全相等
   （`0003` 的 `index 1ecdf5f..` / `c03b176..` / `5b13530..`，左边正是镜像里的值）。
2. **0004 不能重建这个镜像。** 它把 0003 也包含进去了，重建出来会比镜像多 3 个文件的
   dspark 构造下放改动（CPU/UVA head + offloader reset）。
3. **可复现镜像的最小配方**：`d63af5a` → 0001 → 0002 → 0005。从干净解压的源码重跑一遍，
   **9 个 Python 文件的 blob 全部命中镜像**（`5a180151` / `a65d074f` / `1ecdf5fa` / `4ef765f9` /
   `c03b1760` / `5b13530e` / `831a4f34` / `3aaffff1` / `d35f525e`）—— 源码层面已是逐字节相同。

> 归档里那份 `pp5-ssd-async-build.sh` 确实有 `apply dspark-construction-offload` 的步骤，
> 但**无法确认它就是 11-02 那次构建用的版本**（归档同时有 **11-01** 的 r11/r12 配方：
> `pp5-dspark-r11-construction-uva.Containerfile` 07:24、`pp5-dspark-r12-host-packed-uva.Containerfile` 07:50，
> 而 0003 补丁本身的原件时间是 **10-31 13:21**）。
> 有矛盾时以镜像实测 blob 为准 —— 这也是为什么 0005 是从「镜像文件 vs 0001 后文件」直接 diff 出来的，
> 而不是从任何脚本推断的。
>
> 顺带纠正一句旧说法：这里没有「补丁 sha256 与构建脚本硬编码值一致」这回事，
> 归档脚本只钉了基础镜像和 Containerfile 的 sha256，没钉补丁。

## 归属：这套引擎里哪些是「我们的」

| 来源 | 内容 |
|---|---|
| **上游别人**：`lazymio/vllm-backport:v0.13.0-sm80`（[Docker Hub](https://hub.docker.com/r/lazymio/vllm-backport)，digest `bfe237ee…`） | 基础镜像（CUDA 13 / py3.12 / torch），占了镜像 29.7 GB 的绝大部分 |
| **上游别人**：`344303947/dsv41-flash-pp5-170hx@d63af5a` | `vllm/models/deepseek_v4_1/` 整棵自定义模型树（V4.1 的支持代码） |
| **上游别人**：`aiultrayang/deepseek-v41-flash-cmp170hx@29b03af5`（`pp6-ssd/`） | 0001 预热 PP 通信器 · 0002 engram SSD 移植 · `Containerfile`（原样使用，一字未改） |
| **我们本地写的**（两个文件，上游从未发布过，见下） | `0005-pp5-lazy-receive.patch`：`gpu_worker.py` 一个文件，**+37/−9 行、5 个 hunk** —— `comm_loader` 延迟接收机制、`VLLM_PP5_LAZY_RECEIVE` 开关、5 条拓扑断言、`receive_intermediates()` 闭包重构、一个调试计数器<br>`0003-dspark-construction-offload.patch`：**2026-10-31 的 r11 构造 UVA 实验**，3 个 dspark 文件，**从未进过这个镜像** |

核对方法（任意有镜像的机器）：

```bash
# 0005 之外还有没有我们不知道的改动？对 blob 即可，9 个文件应全部命中
docker run --rm --entrypoint python3 <镜像> -c "
import hashlib
for f in ['v1/worker/gpu_worker.py','config/engram.py']:
    d=open('/usr/local/lib/python3.12/dist-packages/vllm/'+f,'rb').read()
    print(hashlib.sha1(b'blob %d\0'%len(d)+d).hexdigest(), f)"
# gpu_worker.py 应为 831a4f34decb278bfe75d838786f815700884722
```

**结论：这个引擎的速度不是靠我们改代码换来的**，是靠调参（PP 分区、`cpu_threads`、engram `ssd|ram`、
`max_model_len 262144`、`max_num_batched_tokens 2048`、`gpu_memory_utilization 0.97`）。
代码层面我们只加了一个惰性接收开关，而它恰恰是「必须用 V1 runner」这个约束的来源。

这也决定了**它的可分发形态**：值得传播的是上面那张归属表 + 5 个补丁 + `build.sh`（合计 155 KB），
不是那个 29.7 GB、且因为 `COPY vllm/` 覆盖进 site-packages 而没有 `.git` 的镜像。

## 重建

```bash
engine/v1-legacy/build.sh [输出镜像名]
```

脚本做的六步：取 `d63af5a` → 校验 5 个补丁 sha256 → 依次应用 **0001 → 0002 → 0005**（0003/0004 不用）→ 组上下文
（`vllm/` + `native/engram_ssd.cpp` + `Containerfile`）→ `docker build` → 自检
`libdsv41_cuda.so` 的 `dsv41_engram_ssd_version` 能加载。

⚠️ 未在干净机器上端到端验证过。它能解决「补丁丢失」，但**不能解决「GitHub 仓库消失」**——
那时才需要镜像离线包（`tools/engine-bundle.sh save v1-legacy`，29.7 GB → 约 12~15 GB）。

## 校验

```bash
docker image inspect local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102 \
  --format '{{.Id}}'
# sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1

docker run --rm --network none --entrypoint python3 <镜像> -c \
  'import ctypes,pathlib,vllm; p=pathlib.Path(vllm.__file__).parent/"libdsv41_cuda.so";
   lib=ctypes.CDLL(str(p)); lib.dsv41_engram_ssd_version.restype=ctypes.c_char_p;
   print(lib.dsv41_engram_ssd_version())'
```
