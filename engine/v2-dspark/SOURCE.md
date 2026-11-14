# 新引擎 V2 · dspark（方案 A / B 共用）

## 镜像

| 项 | 值 |
|---|---|
| tag | `18gogogo/170hx4-256r-dsv41-1017-flash:sm80` |
| image ID | `sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157` |
| 体积 / 层数 | 47.6 GB / 69 层 |
| 构建时间 | 2026-10-20 12:49 (+08:00) |
| 基础镜像 | `nvidia/cuda:13.0.3-devel-ubuntu24.04`（**tag 名为推断**；实测环境变量：CUDA 13.0.3 / cudart 13.0.96 / Ubuntu 24.04.4 / NCCL 2.28.3 / 无 cuDNN）+ Python 3.12.3 + torch 2.13.0 |
| 运行时 | `vllm.__version__ = 0.1.dev21380+gc3d3104de`，V2 Model Runner |
| 是否可拉 | **未验证**（本机到 Docker Hub 全部超时，且无登录信息） |

## 源码

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/buliaoyin/vllm-backport.git`（第三方 fork） |
| commit | `c3d3104de0163c389d26d9fcba0180f55ad7964d`（detached HEAD） |
| 分支 | `codex/deepseek-v41-cpu-hybrid`（上游文档用的名字）· `sm80-optimize`（另一个名字）—— 2026-11-14 核实：**两者 HEAD 相同**（`12d7c186`），都已在 `c3d3104de` 之后前进，所以必须钉 commit 而非分支 |
| 提交说明 | `[Bugfix] Fix DeepSeek hybrid CPU KV reuse and cache budgeting` |
| 仓库体积 | 镜像内 `.git` 260 MB（**镜像自带完整仓库，可自证 commit**） |
| 模型代码位置 | `/workspace/vllm-backport/vllm/models/deepseek_v4_1/` |

镜像内 `git status` 有 **3 个未提交文件**，也就是本目录的补丁：

| 文件 | 改动 | 内容 |
|---|---|---|
| `vllm/models/deepseek_v4_1/expert_cache.py` | +43 −4 | `DSV41_HOST_STAGING_CACHE=1` 时走 pinned 主机暂存（固定地址，跨 graph replay 不变） |
| `vllm/models/deepseek_v4_1/hybrid.py` | +11 −0 | 校验 `cache_decode_interval` 必须为正整数；`DSV41_FORCE_LOCAL_CACHE` 把非本卡容量清零 |
| `vllm/models/deepseek_v4_1/hybrid_runtime.py` | +4 +16 | `decode_interval` 改为读配置（原来硬编码 `16 if r8_refill else 64`）；轮换 CUDA Graph pool |

合计 **74 增 / 5 删**，补丁 sha256 `7448fefe8ccc05da5ae4ce53bbf59866a954946c243bc04b38ac249da0e9e8d6`，
与镜像内 `git diff | sha256sum` **完全一致**，且 `git apply -R --check` 可干净反向应用。

> **⚠️ 归属（2026-11-14 核实更正）**：这 74 行**不是我们写的**。它与
> [`LocalFrameLab/deepseek-v41-cmp170hx-x99`](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99)
> 的 [`patches/x99-stable.patch`](https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch)（Apache-2.0）
> **逐行相同** —— 只有 `index` 行的 hash 缩写长度和 5 处行尾空格不同，代码一字不差。
> 证据与复核命令见 [`PROVENANCE.md`](PROVENANCE.md)。上游原件已存档为
> `patches/upstream-x99-stable.patch`（sha256 `16f2a4606dae…`，2026-11-14 实测 HTTP 200）；`build.sh` 仍只校验和使用 `0001`。
> 早前文档写的「来源不明，可能是镜像作者也可能是我们」作废。

> **为什么这个补丁不能省**：`schemes/pp5-fast/run.sh` 里有 `-e DSV41_HOST_STAGING_CACHE=1`，
> 并且 `additional-config` 里传了 `cache_decode_interval`。没有这 74 行，这两个参数要么无效、
> 要么走回硬编码默认值 —— 那是另一个引擎，不是我们测出 82.5 tok/s 的那个。

## 构建层序

原作者**没有留下 Dockerfile**。下面是 `image-history.txt` 反推的层序，以及据此写出的
`Containerfile`（2026-11-14 新写，**未跑过**）：

1. `FROM nvidia/cuda:13.0.3-devel-ubuntu24.04` + apt 装 python3.12 / build-essential / git / cmake / ninja
2. `ENV VLLM_TARGET_DEVICE=cuda TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=8`（**sm80 专用编译**）
3. `COPY vllm-backport /workspace/vllm-backport`（408 MB，**含 .git**）
4. `uv venv --python 3.12 .venv` + `uv pip install torch==2.13.0`（9.6 GB，必须用 cu130 索引）
5. `uv pip install -r requirements.freeze.txt` 其余 197 个依赖（13.5 GB）
6. `uv pip install -e . --no-build-isolation` 编译 5 个 CUDA 扩展
7. `ENV PATH=/workspace/vllm-backport/.venv/bin:...`

第 3 步 COPY 的是**含未提交改动的整棵树**，所以补丁是「烤进镜像」的，不是构建时 `patch` 出来的。
这也解释了为什么从 fork commit 重建会少东西 —— 我们的 `build.sh` 补上了 `git apply` 这一步，效果等价。
第 6 步是 editable 安装，版本号靠 `.git` 由 `setuptools_scm` 算出；删掉 `.git` 会变成 `0.0.0` 并安装失败。

## 还原

| 方式 | 命令 | 说明 |
|---|---|---|
| **源码重建（主）** | `engine/v2-dspark/build.sh`（克隆 → 补丁 → `Containerfile` 构建 → 自检） | 需外网 + 60~80 GB 磁盘，40~90 分钟；配方未在干净机器跑过 |
| **+ 编译产物** | `tools/engine-bundle.sh save-so v2-dspark` → `load-so --into <重建出的镜像>` | 191 MB；让重建产物性能等同原引擎 |
| 直接拉 | `docker pull 18gogogo/170hx4-256r-dsv41-1017-flash:sm80` | ⚠️ 未验证仓库是否公开 |
| 离线整包 | `tools/engine-bundle.sh save v2-dspark --out <盘>` → `load` | 20~25 GB（仅无外网应急），永不进 git |

## 校验

```bash
docker image inspect 18gogogo/170hx4-256r-dsv41-1017-flash:sm80 --format '{{.Id}}'
# sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157

docker run --rm --network none --entrypoint bash <镜像> -c \
  'cd /workspace/vllm-backport && git rev-parse HEAD && git diff | sha256sum'
# c3d3104de0163c389d26d9fcba0180f55ad7964d
# 7448fefe8ccc05da5ae4ce53bbf59866a954946c243bc04b38ac249da0e9e8d6
```
