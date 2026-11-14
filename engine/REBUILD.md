# 从原始参考仓库到能跑：两个镜像各自要做什么改动

> 这份文件回答一个问题：**把基础镜像和源码仓库拉下来之后，还差哪些改动？**
> 答案分两个引擎写，每项都标了「是原作者的 / 是我们的 / 已验证 / 未验证」。
>
> ⚠️ 先说清楚：**3 个方案只对应 2 个镜像**。A（PP5 fast）和 B（PP4 t16）是同一个镜像，
> 差别只在启动参数；C（PP5 legacy）是另一个镜像。

## 一、总览

| 步骤 | 新引擎 V2（A / B） | 旧引擎 V1（C） |
|---|---|---|
| **1 起点** | `nvidia/cuda:13.0.3-devel-ubuntu24.04`（NVIDIA 官方；**tag 名是推断**，环境变量已证 CUDA 13.0.3 / Ubuntu 24.04.4 / 无 cuDNN） | `lazymio/vllm-backport@sha256:bfe237ee…`（第三方，非官方） |
| **2 源码** | [`buliaoyin/vllm-backport`](https://github.com/buliaoyin/vllm-backport) @ `c3d3104de0`（分支 `sm80-optimize`） | [`344303947/dsv41-flash-pp5-170hx`](https://github.com/344303947/dsv41-flash-pp5-170hx) @ `d63af5a472` |
| **3 改代码** | **74 行 / 3 文件**（未提交补丁） | **0001 + 0002 + 0005**（+37/−9 是我们写的；0003 未进镜像） |
| **4 编译** | **5 个 CUDA 扩展**，最大 138 MB | **1 个 C++ 扩展** `libdsv41_cuda.so` |
| **5 装依赖** | **197 个包**（已从镜像导出 freeze） | 基础镜像自带，无需另装 |
| **6 安装方式** | `pip install -e .`（editable，**保留 `.git`**） | `COPY vllm/` 覆盖 site-packages（**无 `.git`**） |
| **7 产物** | 47.6 GB / 69 层 | 29.7 GB / 123 层 |
| **8 现成脚本** | `engine/v2-dspark/build.sh`（2026-11-14 新写） | `engine/v1-legacy/build.sh` |
| **9 端到端验证** | ⚠️ 未跑过（配方是反推的） | ⚠️ 未跑过（配方对应原构建脚本） |

## 二、新引擎 V2：拉完基础镜像和 fork 之后，要改的是这 4 件事

### 改动 1 · 打 74 行补丁（3 个文件）

`engine/v2-dspark/patches/0001-host-staging-and-cache-interval.patch`，sha256 `7448fefe8ccc…`

**来源：[LocalFrameLab/deepseek-v41-cmp170hx-x99](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99) 的 [`patches/x99-stable.patch`](https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch)（Apache-2.0），
逐行相同，我们未作任何修改。** 核对过程见 [`../v2-dspark/PROVENANCE.md`](../v2-dspark/PROVENANCE.md)。

| 文件 | 行数 | 改了什么 | 对应我们启动脚本里的参数 |
|---|---|---|---|
| `vllm/models/deepseek_v4_1/expert_cache.py` | +43 −4 | `DSV41_HOST_STAGING_CACHE=1` 时把专家暂存到 **pinned 主机内存**（固定地址，跨 CUDA graph replay 不变） | `-e DSV41_HOST_STAGING_CACHE=1` |
| `vllm/models/deepseek_v4_1/hybrid.py` | +11 −0 | 校验 `cache_decode_interval` 必须是正整数；`DSV41_FORCE_LOCAL_CACHE` 把「非本卡容量」清零 | `additional-config` 里的 `cache_decode_interval: 64` |
| `vllm/models/deepseek_v4_1/hybrid_runtime.py` | +20 −1 | `decode_interval` 改为读配置（原代码硬编码 `16 if r8_refill else 64`） | 同上 |

**为什么不能省**：这 74 行不在 fork 的 `c3d3104de0` 提交里。少了它，上面两个参数会被**静默忽略**
或走回硬编码值——那是另一个引擎，不是测出 82.5 tok/s 的那个。

### 改动 2 · 按 sm80 编译 5 个 CUDA 扩展

必须设 `TORCH_CUDA_ARCH_LIST=8.0`（CMP 170HX 是 GA100 / sm80）。不设会连 sm90 一起编，时间翻几倍，
而且用不上。产物：

| 产物 | 大小 |
|---|---:|
| `vllm/_C_stable_libtorch.abi3.so` | 138 MB |
| `vllm/_moe_C_stable_libtorch.abi3.so` | 48 MB |
| `vllm/_exl3_C.abi3.so` | 4.7 MB |
| `vllm/cumem_allocator.abi3.so` | 104 KB |
| `vllm/fs_io_C.abi3.so` | 402 KB |

### 改动 3 · 装 197 个 Python 包

已把**运行中镜像里的确切版本**导出成 `engine/v2-dspark/requirements.freeze.txt`（197 行，非推测）。
几个关键版本：

| 包 | 版本 | 备注 |
|---|---|---|
| torch / torchvision / torchaudio | 2.13.0 / 0.28.0+cu130 / 2.11.0+cu130 | 必须从 `download.pytorch.org/whl/cu130` 装，PyPI 默认是 cu128 |
| triton | 3.7.1 | |
| flashinfer-python | 0.6.18.post1 | |
| humming-kernels | 0.1.12 | 装了但没用上（见 `docs/rejected.md` H 行） |
| transformers | 5.20.2 | |

### 改动 4 · 保留 `.git`（容易被忽略）

`vllm` 的版本号由 `setuptools_scm` 从 git 描述算出来：`0.1.dev21380+gc3d3104de.d20261020`。
把 `.git` 删掉再 `pip install -e .`，版本会变成 `0.0.0`，安装直接失败。镜像里那 260 MB `.git` 不是垃圾。

### 不需要改的东西（同样重要）

模型权重（1017 原版，镜像里没有）、量化格式（MXFP4 原样）、算子实现、vLLM 其余部分——都没动。

## 三、旧引擎 V1：拉完基础镜像和 fork 之后，要改的是这 3 件事

### 改动 1 · 依次应用 0001 → 0002 → 0005

```bash
git -C src apply engine/v1-legacy/patches/0001-prime-pp-communicators-before-model-load.patch
git -C src apply engine/v1-legacy/patches/0002-engram-ssd-port.patch
git -C src apply engine/v1-legacy/patches/0005-pp5-lazy-receive.patch
```

| 补丁 | 内容 | 在镜像里 |
|---|---|---|
| 0001（3.3 KB） | 预热 PP 通信器 | ✅ |
| 0002（39 KB） | engram 移植到 SSD（O_DIRECT + Linux AIO） | ✅ |
| 0005（4.2 KB） | 本仓库写的惰性接收，`gpu_worker.py` +37/−9 | ✅ |

⚠️ **不要用 0004**。它是含 0003（dspark 构造下放）的旧快照，而 0003 **不在镜像里** ——
2026-11-14 实测：镜像内那 3 个 dspark 文件的 blob 与 0003 的 pre-image 完全相等。
用 0004 重建会得到一个「多 3 个文件改动」的不同引擎。证据见
[`v1-legacy/SOURCE.md`](v1-legacy/SOURCE.md)。0003/0004 留档只为追溯，不参与构建。

### 改动 2 · 编译一个 C++ 扩展

`csrc/cpu/dsv41/engram_ssd.cpp` → `libdsv41_cuda.so`，用 g++ 编，链接 base 镜像里的 `libcudart`。
见 `engine/v1-legacy/Containerfile` 的两阶段（stage1 编译 → stage2 覆盖）。

### 改动 3 · 用 `COPY` 覆盖 site-packages（不是 editable）

这就是它**没有 `.git`、无法自证 commit** 的原因。也意味着重建时源码树必须完整打完补丁再 COPY。

## 四、验证状态：哪些已证实，哪些只是推断

| 环节 | V2 | V1 |
|---|---|---|
| 补丁内容与镜像一致 | ✅ 已证（`git status -uall` 只有 3 个文件，diff == +74/−5；补丁 sha256 == 镜像内 `git diff \| sha256sum`） | ✅ 已证（9 个 Python 文件的 **git blob** 与「`d63af5a`+0001+0002+0005」完全相等；同时证出 0003 不在镜像里） |
| 依赖版本 | ✅ 已证（从运行中镜像导出） | ✅ 基础镜像自带，无需导出 |
| 基础镜像 | ⚠️ 环境变量已证 `CUDA_VERSION=13.0.3` / Ubuntu 24.04.4 / NCCL 2.28.3 / 无 cuDNN；**但 tag 名是按命名规则推断的**（本机到 Docker Hub 不通，无法核对）。脚本支持 `DSV41_V2_BASE` 覆盖 | ⚠️ 第三方 `lazymio`，能否 pull 未验证 |
| 源码仓库还在 | ✅ 三个仓库均公开（2026-11-14 实测 HTTP 200，tarball 可取） | ✅ 同上（已实际下载 `d63af5a` 并复现） |
| 构建配方 | ⚠️ **反推所得，从未跑过**（原作者没留 Dockerfile） | ⚠️ 对应原构建脚本，未在干净机器跑过 |
| 重建后二进制一致 | ❌ 不保证 | ❌ 不保证 |

## 五、重建成本

| | V2 | V1 |
|---|---|---|
| 时间 | 40~90 分钟（编译是大头） | 5~10 分钟 |
| 磁盘 | 60~80 GB（基础镜像 9 GB + torch/deps 23 GB + 构建缓存） | 30~40 GB |
| 外网依赖 | GitHub + Docker Hub + PyPI + download.pytorch.org | GitHub + Docker Hub |
| 一条命令 | `engine/v2-dspark/build.sh` | `engine/v1-legacy/build.sh` |

## 六、重建 vs 还原：两个不同的问题

先把问题拆开，混在一起就会得出「必须存 47.6 GB」的错误结论：

- **「这个引擎是怎么造出来的」** → 上表五步全部可查，占 **260 KB**，已在 git 里。这就是本文件的意义。
- **「换台机器能不能造出同一个」** → 卡在最后一格：编译产物不保证逐字节一致。
  而这格只需要存 **191 MB**（5 个 `.so`），不需要存 47.6 GB。

47.6 GB 的构成决定了这一点：CUDA 基础镜像 7.4 GB、torch 9.6 GB、194 个包 13.5 GB、
源码树 408 MB —— **99.6% 都是「记得住地址就能重新下载」的公开资源**。

因此还原方式按手上的条件选，不是按「哪个更保险」排：

| 条件 | 方式 | 耗时 | 二进制等同原引擎 |
|---|---|---:|---|
| 有外网 + 那 5 个 `.so`（191 MB） | `build.sh` 重建，然后覆盖 `.so` | 40~90 分钟 | ✅ |
| 有外网，没有 `.so` | `build.sh` 从头编译 | 40~90 分钟 | ⚠️ 不保证，测速数字需重新立方案 |
| 无外网 | 离线整包 `engine-bundle.sh load` | 10~20 分钟 | ✅ |

取舍就三句话：

- **配方（260 KB）必须一直在 git 里** —— 别人看懂和审计的唯一入口。
- **那 5 个 `.so`（191 MB）应该进 Release 附件** —— 不省编译时间，但让重建产物的性能等同于原引擎，
  历史基准数字可以沿用，不必重新立方案。
- **整包（20~25 GB）只是无网应急副本** —— 永不进 git，`.gitignore` 已排除。

> 一句话：**仓库里已经有「上游地址 + 我们的 patch」了，缺的只是那 191 MB 编译产物。**
> 之前把离线整包写成「首选」，是因为没算这笔体积账。
