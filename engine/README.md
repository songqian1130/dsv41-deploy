# 引擎：每个方案到底跑的是什么

> **一句话：三个方案 = 两个第三方引擎镜像。A/B 同一个，C 一个。**
> 没有一个是「官方 vLLM + 我们的补丁」—— 官方 vLLM 里根本没有 `deepseek_v4_1` 这层模型实现。
>
> 三件容易搞混的事，先摆平：
>
> 1. **3 个方案 = 2 个镜像**。A（PP5 fast）和 B（PP4 t16）只差启动参数。
> 2. **属于我们写的代码，只有旧引擎的 `0005` 进了镜像**（`gpu_worker.py` +37/−9，惰性接收）。
>    新引擎那 74 行是 LocalFrameLab 的 `x99-stable.patch`（Apache-2.0），**一字未改**
>    （核对见 [`v2-dspark/PROVENANCE.md`](v2-dspark/PROVENANCE.md)）。速度优势来自调参，不是改代码。
> 3. **「拉下参考仓库」≠「能部署」**：还差补丁、编译、依赖。逐项清单见 [`REBUILD.md`](REBUILD.md)。
>
> 📍 每个外部依赖的完整地址（commit / raw URL / digest / revision / sha256）全在 **[`LINKS.md`](LINKS.md)**，
> 一条命令复验：`./tools/verify-sources.sh`。

## 零、重建公式：上游地址 + 我们的补丁

```
上游地址（别人给的 URL）  +  我们的 .patch（3 个，在仓库里）  =  同一个引擎
```

`engine/` 一共 **260 KB**，里面是：基础镜像名、fork 地址、钉死 commit、补丁、197 个包的确切版本、
一条能跑完的 `build.sh`。这就是全部，不需要存镜像。

### 外部文件分两类

| | 有哪些 | 地址怎么写 | 在哪 |
|---|---|---|---|
| 别人写的（原样复制） | 4 个 + 1 份副本 | 完整 URL，点开就能下 | 第三节逐条列出 |
| **我们改的** | 3 个补丁 + 1 个本机镜像 + 1 份反推的 Containerfile | 上游 commit 的 URL + 仓库里这份 `.patch` | 第三节 + [`LINKS.md`](LINKS.md) 2.1 |

**「我们改的」没有第三方 URL，这是正常的，不是丢件。** 它们本来就不存在于上游，
身份 = 基座 commit（有 URL）+ 本仓库这份 patch（可读、可校验）+ 再生命令（跑过）。

### 47.6 GB 里，只有 191 MB 不能保证「编出来一模一样」

把镜像拆开看：

| 组成 | 大小 | 怎么来 | 需要存吗 |
|---|---:|---|---|
| `nvidia/cuda:13.0.3-devel-ubuntu24.04` | ~7.4 GB | `docker pull` | ❌ 公开 |
| apt 工具链 + uv | 246 MB | apt / astral.sh | ❌ 公开 |
| `torch==2.13.0` (+vision/audio) | 9.6 GB | `download.pytorch.org/whl/cu130` | ❌ 公开，版本已在 `requirements.freeze.txt` |
| 其余 194 个包 | 13.5 GB | PyPI | ❌ 公开，版本已 freeze |
| fork 源码树（含 260 MB `.git`） | 408 MB | `git clone` [buliaoyin/vllm-backport](https://github.com/buliaoyin/vllm-backport) 后 checkout `c3d3104de0` | ❌ 公开 |
| **5 个编译产物 `.so`** | **191 MB** | 源码编译，但编译器版本没钉死 | ⚠️ **能重编，不保证逐字节相同** |
| **我们那 74 行改动** | **8.3 KB** | patch | ✅ 已在 `patches/` |

191 MB ÷ 47.6 GB = **0.39%**。

**这里要纠正一个说法**：这 191 MB **不是「没有源码所以造不出来」**。源码一直在
（`buliaoyin/vllm-backport` @ `c3d3104de`），任何一台能上网的机器都能重新编出这 5 个 `.so`，
功能完全一样、能跑同样的参数。不保证的只是**逐字节相同**，原因有两个：

1. 基础镜像钉的是 **tag**（`nvidia/cuda:13.0.3-devel-ubuntu24.04`）不是 digest，tag 会被重新发布；
2. `apt` 装的 `build-essential` / `cmake` / `ninja-build` **没写版本号**，跟着 Ubuntu 仓库走 ——
   编译器一换，生成的指令序列就变。

后果不是「跑不起来」，而是**跑起来的数字可能跟我们的基准不一样**。所以那 191 MB 的定位是
**性能基准锚点**（保证历史测速数字继续有效），不是「唯一副本」。这个前提我们**没有实测过**
（没跑过第二次完整构建再对比），属于保险，不是实测结论。

三种做法各有位置，**不是二选一**：

| 做法 | 体积 | 什么时候用 | 换来什么 |
|---|---:|---|---|
| 配方重建（主路径） | **260 KB** | 有外网，40~90 分钟 | 代码一致；可审计、可 diff |
| 配方 + 那 5 个 `.so` | +191 MB | 重建后要沿用历史基准 | 二进制一致，测速数字不作废（**不省编译时间**） |
| 离线整包（应急） | 20~25 GB | 目标机**完全没网** | 导入 10~20 分钟就能起 |

**整包不是「上传到 GitHub」**：`.gitignore` 已排除 `*.tar.zst` / `*.tar` / `engine/bundles/`，
它从头到尾只是一份本地归档。之前文档把它写成「路径 1，首选」，是定位讲错了 ——
它只解决「今天没网、但必须马上起服务」这一种场景。

## 一、为什么不能「官方 vLLM + 我们的补丁」

| 障碍 | 能不能解决 |
|---|---|
| `vllm/models/deepseek_v4_1/` 只存在于 fork 仓库（`buliaoyin/vllm-backport`），`pip install vllm` 装出来的包里没有这一层 | ⚠️ 必须用 fork 源码，绕不过（除非自己移植） |
| 运行中的引擎有 3 个文件共 74 行未提交改动（`DSV41_HOST_STAGING_CACHE`、`cache_decode_interval`），fork commit 里没有 | ✅ 已补齐：`patches/0001-*.patch`，sha256 与镜像内 `git diff` 逐字节相等 |
| 旧引擎的 0001/0002 原来靠容器构建时从别人 GitHub raw 临时下载 | ✅ 已入库；0003/0005 本来就是我们写的 |

准确说法：**官方 CUDA 基础镜像 + 第三方 fork 源码 + 我们的补丁**。这正是 `engine/*/build.sh` 在干的事 ——
三条阻碍里两条已用 patch 解决，剩下一条是「必须用 fork」的硬约束，不是「必须存镜像」。

## 二、三个引擎的真实来源

| 方案 | 镜像 | 镜像 ID（校验用） | 体积 | 代码来源 | commit |
|---|---|---|---:|---|---|
| A · PP5 fast | [`18gogogo/170hx4-256r-dsv41-1017-flash:sm80`](https://hub.docker.com/r/18gogogo/170hx4-256r-dsv41-1017-flash) | `sha256:4e08e1c9231e…` | 47.6 GB | [`buliaoyin/vllm-backport`](https://github.com/buliaoyin/vllm-backport)（fork） | `c3d3104de0` |
| B · PP4 t16 | 同 A | 同 A | 47.6 GB | 同 A | 同 A |
| C · PP5 legacy | `local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102`（**无地址**，`local/` 本机构建） | `sha256:4f7370a2a284…` | 29.7 GB | [`344303947/dsv41-flash-pp5-170hx`](https://github.com/344303947/dsv41-flash-pp5-170hx)（fork） | `d63af5a472` |

两个镜像**不是同一个基础**（这一点之前写错过，2026-11-14 核实后更正）：

- **基础镜像**：V2 是 `nvidia/cuda:13.0.3-devel-ubuntu24.04`（NVIDIA 官方，CUDA 13.0.3 / Ubuntu 24.04.4 /
  NCCL 2.28.3 / 无 cuDNN）；V1 是 `lazymio/vllm-backport:v0.13.0-sm80`（第三方把 `deepseek_v4_1` 合进
  vLLM 0.13.0 的镜像）。两者都不是 `vllm/vllm-openai` 官方镜像。
- **V2 也从没装过 lazymio 的 base**：它是从 CUDA 基础镜像自己 `uv venv` + 编译扩展搭起来的。

差异在叠加层：

- **A/B（新 V2）**：`buliaoyin/vllm-backport` @ `c3d3104de0` 整棵树 `pip install -e .`（editable），
  再装 torch 2.13 + 197 个依赖（23 GB 一层）。**镜像里带 `.git`（260 MB）**，因此镜像本身能自证 commit，
  也能 `git bundle` 导出源码。
- **C（旧 V1）**：把 `344303947/dsv41-flash-pp5-170hx` @ `d63af5a472` 的 `vllm/` 整棵 COPY 覆盖
  到 site-packages（非 editable，所以无 `.git`），另外编译一个 C++ 扩展 `libdsv41_cuda.so`
  （O_DIRECT + Linux AIO 的 engram 后端）。

**逐项改动清单**（拉完基础镜像和源码之后还差什么）见 [`REBUILD.md`](REBUILD.md)。

## 三、补丁清单（已入库）

### 新引擎 V2 · `engine/v2-dspark/patches/`

| 文件 | 大小 | sha256（前 12） | 作者 | 作用 |
|---|---:|---|---|---|
| `0001-host-staging-and-cache-interval.patch` | 8.3 KB | `7448fefe8ccc` | **LocalFrameLab**（Apache-2.0） | 3 文件 +74/−5：`DSV41_HOST_STAGING_CACHE` 主机暂存、`cache_decode_interval` 读取校验、`DSV41_FORCE_LOCAL_CACHE`、CUDA Graph pool 轮换 |
| `upstream-x99-stable.patch` | 8.3 KB | `16f2a4606dae` | LocalFrameLab（原件） | 与 0001 **逐行相同**的证据副本，不参与构建，见 [`PROVENANCE.md`](v2-dspark/PROVENANCE.md) |

**这是当前生产唯一依赖的运行时补丁，且它不是我们写的。** 校验方式：在镜像里 `git diff | sha256sum`
得到的值与文件 sha256 **完全相等**（`7448fefe8ccc05da…`），并且 `git apply -R --check` 能干净反向应用 ——
说明补丁与镜像内工作树逐字节一致。

与上游公开件的差异**只在 patch 元数据**：3 处 `index` 行 hash 缩写（9 位 vs 12 位）+ 5 处行尾空格，
合计 23 字节。每一行代码、每个 hunk 位置都相同。

> **出处**：[`LocalFrameLab/x99-stable.patch`](https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch)
> （Apache-2.0，sha256 `16f2a4606dae…`，2026-11-14 实测 200）。我们这份是镜像里 `git diff` 导出的，
> 与它逐行相同 —— 所以「上游地址还在」就等于「这个补丁还能拿到」。

同目录另有三件重建所需的东西（2026-11-14 补齐）：

| 文件 | 作用 |
|---|---|
| `Containerfile` | 按镜像层历史反推的构建配方（基础镜像 `nvidia/cuda:13.0.3-devel-ubuntu24.04`） |
| `requirements.freeze.txt` | **197 个包的确切版本**，从运行中的镜像导出，不是推测 |
| `build.sh` | 一条命令跑完：克隆 → 校验并应用补丁 → 构建 → 自检 |

### 旧引擎 V1 · `engine/v1-legacy/patches/`

| 文件 | 大小 | sha256（前 12） | 作者 | 进镜像 | 作用 |
|---|---:|---|---|---|---|
| `0001-prime-pp-communicators-before-model-load.patch` | 3.3 KB | `16d8b7554700` | aiultrayang | ✅ | 加载前预热 PP 通信器 |
| `0002-engram-ssd-port.patch` | 39 KB | `5ec439e0d300` | aiultrayang | ✅ | engram 后端改为 SSD（O_DIRECT + Linux AIO） |
| `0005-pp5-lazy-receive.patch` | 4.2 KB | `30b53a49fd64` | **本仓库** | ✅ | `gpu_worker.py` +37/−9：`comm_loader` 延迟接收 + `VLLM_PP5_LAZY_RECEIVE` 开关 + 5 条断言 |
| `0003-dspark-construction-offload.patch` | 5.8 KB | `57c0ef0e3808` | **本仓库**（2026-10-31 r11 实验） | ❌ | dspark 构造阶段 UVA 下放，**不在本镜像**（r11/r12 那条实验线专用） |
| `0004-runtime-cumulative-snapshot.patch` | 52 KB | `f8c6cb49b08a` | 构建工作区快照（含 0001/0002/0005） | ❌ | 含 0003 的历史快照，**用它重建会得到另一个引擎** |

**出处（2026-11-14 实测 HTTP 200，拉下来 sha256 与归档逐字节一致）：**

| 文件 | 上游地址 |
|---|---|
| `0001` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/patches/0001-prime-pp-communicators-before-model-load.patch> |
| `0002` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/patches/engram-ssd-port.patch> ← **上游名没有 `0002-` 前缀** |
| `Containerfile` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/build/Containerfile> ← 原样使用，一字未改 |
| `0003` | **我们改的**（该 commit 只有 2 个补丁，没有它）。原件在 `archive/legacy/patches/`，sha256 相同；也可从 `0004` 精确抽出 |
| `0004` | **我们改的**（2026-11-02 构建工作区快照），只有仓库副本，再生不了 |
| `0005` | **我们改的**。定义 = 镜像内 `gpu_worker.py` − (`d63af5a`+0001+0002) |

这三个的再生命令都写在 **[`LINKS.md`](LINKS.md) 2.1**，逐条都跑过。
它们**不是丢件**，身份 = 上游 commit + 我们自己改的 patch，见 [`LINKS.md`](LINKS.md) 第 0 节。

外加 `engine/v1-legacy/Containerfile`（两阶段：编译 `.so` → 覆盖 `vllm/`）。

⚠️ **2026-11-14 夜间实测纠正（重要）**：镜像里只有 0001 + 0002 + 0005。
那 3 个 dspark 文件的 blob 与 **0003 的 pre-image 完全相等** → 0003 从没进过这个镜像。
所以重建顺序是 **0001 → 0002 → 0005**，不要用 0004。证据与逐文件 blob 表见
[`v1-legacy/SOURCE.md`](v1-legacy/SOURCE.md#2026-11-14-复核镜像--d63af5a--0001--0002--0005)。

## 四、怎么还原（按场景选，不是按优先级排）

先说结论：**正常还原走「源码重建」，它是主路径，不依赖任何镜像存档。**
离线整包只在「这台机器完全没有外网」时才有存在价值 —— 那是应急，不是常态。

| 你手上的条件 | 走哪条 | 耗时 | 产物是否等同原引擎 |
|---|---|---:|---|
| 有外网（GitHub + Docker Hub + PyPI） | ① 源码重建（主路径） | 40~90 分钟（编译无法跳过） | 代码一致；**把那 5 个 `.so` 放回去后二进制一致** |
| 只有目标机，无外网 | ② 离线整包 | 导入 10~20 分钟 | ✅ 完全一致 |
| A/B，且 `18gogogo` 仓库确认公开 | ③ 直接 pull | 下载 47.6 GB | ✅ 完全一致 |

### ① 源码重建（主路径，260 KB）

这是仓库存在的意义。所需一切都在 `engine/` 里，不依赖任何镜像存档：

```bash
engine/v2-dspark/build.sh          # 方案 A / B：克隆 → 校验 patch → 编译 → 自检
engine/v1-legacy/build.sh          # 方案 C
```

它做的事：`git clone` fork @ 钉死的 commit → `git apply` 我们的 patch（带 sha256 校验，
不匹配直接退出）→ 按 `Containerfile` 拉公开基础镜像 → 装 `requirements.freeze.txt` 里的
197 个包 → 编译 → 跑三项自检（版本号 / `.so` 存在 / patch 特征串）。

**唯一的不确定性是那 191 MB 编译产物**：源码逐字节一致，也不代表 nvcc 编出来的 `.so` 一致
（编译期 flag、依赖解析顺序、寄存器分配都可能变），所以重建出的引擎**测速数字不能直接和
100.1 tok/s 混比**。解决办法就是把这 191 MB 存下来，重建完把 `.so` 覆盖回去：

```bash
tools/engine-bundle.sh save-so v2-dspark --out <盘>    # 只导 5 个 .so（191 MB）
tools/engine-bundle.sh load-so v2-dspark.tar.zst --into <重建出的镜像>   # 覆盖后 commit
```

⚠️ 注意这一步**不省编译时间**（`pip install -e .` 一定会跑 setup.py 编译，绕不过去），
它省的是「重建后所有基准数字作废、必须重新立方案」这件事 —— 换上实测过的 `.so` 后，
100.1 tok/s 这些数字可以直接沿用。

### ② 离线整包（仅限无外网应急）

```bash
tools/engine-bundle.sh save v2-dspark --out /path/to/big/disk   # 47.6 GB → 压缩后约 20~25 GB
tools/engine-bundle.sh save v1-legacy --out /path/to/big/disk   # 29.7 GB → 压缩后约 12~15 GB
```

产出 `.tar.zst` + `.sha256` + `manifest.json`。**不进 git**（`.gitignore` 已排除），
放你自己的盘或内网 NAS。换机器：`tools/engine-bundle.sh load <name>.tar.zst`（自动校验 sha256 + 镜像 ID）。

适用场景只有一个：目标机连不上 Docker Hub / PyPI / GitHub，且不能等编译。
除此之外它没有任何优势 —— 它比 260 KB 的配方大 10 万倍，还不可审计。

### ③ 直接 pull（仅 A/B，且未验证）

```bash
docker pull 18gogogo/170hx4-256r-dsv41-1017-flash:sm80
docker image inspect ... --format '{{.Id}}'     # 应为 sha256:4e08e1c9231e…
```

⚠️ **未验证这个仓库是否公开**：本机到 `registry-1.docker.io` 的请求全部超时
（`HTTP 000`），也没有任何 `~/.docker/config.json` 登录信息。既不能确认能匿名拉，
也不能确认它存在。**别把这条当成计划 A。** C 的镜像是 `local/` 前缀，从来没推过，这条路对它无效。

⚠️ 另一层风险：即使能拉，也**必须钉 digest 而不是 tag**（`@sha256:4e08e1c9…`），
tag 会被作者覆盖，覆盖后拉到的是另一个引擎。而且这条路的可复现性完全外包给第三方
Docker Hub 仓库 —— 作者删库、转私有、改 tag，你就没有引擎了。这正是它不能当主路径的原因。

## 五、模型不在引擎里（这是比引擎更大的一块）

| 项 | 值 |
|---|---|
| 来源 | ModelScope `deepseek-ai/DeepSeek-V4.1-Flash-1017`，revision `f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad` |
| HF 对应 revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| 权重总量 | **475.2 GiB**（`model.safetensors.index.json` 的 `total_size`），48 个分片 |
| 磁盘占用 | 约 476 GB；`model-00048-of-00048.safetensors` 实体在 `.shards/`，顶层是符号链接 |
| 是否改动 | 否。旧镜像标签写着 `local.deepseek.checkpoint=official-1017-unchanged` |

部署时通过 `DSV41_MODEL` 挂载，**镜像里不含权重**。换一台机器必须自己准备这 476 GB。

## 六、校验命令（还原完先跑这三条）

```bash
# 1. 镜像 ID 对不对
docker image inspect 18gogogo/170hx4-256r-dsv41-1017-flash:sm80 --format '{{.Id}}'
#   期望 sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157

# 2. 引擎内代码是不是那一版（A/B）
docker run --rm --network none --entrypoint bash <镜像> -c \
  'cd /workspace/vllm-backport && git rev-parse HEAD && git diff | sha256sum'
#   期望 c3d3104de0163c389d26d9fcba0180f55ad7964d / 7448fefe8ccc05da5ae4ce53bbf59866a954946c243bc04b38ac249da0e9e8d6

# 3. 补丁能不能对上（反向应用成功 = 一致）
docker run --rm -v "$PWD/engine/v2-dspark/patches:/p:ro" --entrypoint bash <镜像> -c \
  'cd /workspace/vllm-backport && git apply -R --check /p/0001-host-staging-and-cache-interval.patch && echo OK'
```

第三条对 C 不适用（旧镜像没有 `.git`），改用镜像 ID + 标签核对：
`docker image inspect local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102 --format '{{json .Config.Labels}}'`。

## 七、诚实清单：现在还缺什么

按「影响还原能力」排序，前两条是真缺口：

1. **那 5 个 `.so`（191 MB）还没导出** —— 源码能重编，但重编不保证和现在这份逐字节相同，
   所以它是「性能基准锚点」而不是「唯一副本」，至今只存在于运行中的镜像里。
   导出后应进 GitHub Release（单文件 <2 GB），重建完用 `load-so` 覆盖回去（`build.sh` 无法跳过编译，
   因为 `pip install -e .` 一定会编译），换来的是「性能等同原引擎、历史基准可直接沿用」。
   这是当前最该补的一件事。
2. **两个 `build.sh` 都没有在干净机器上端到端跑通过** —— 配方是按 `docker history` 反推的，
   原作者没留 Dockerfile。逻辑上完整（每个文件、每行改动、每个包版本都有出处），但没有「跑通了」的实证。
   要坐实，需要在能上网的机器上跑一次（40~90 分钟 + 60~80 GB 磁盘）。
3. **`18gogogo` 镜像是否公开可拉，未验证**（本机无 Docker Hub 访问）。要确认，请在能上网的机器上
   `docker manifest inspect 18gogogo/170hx4-256r-dsv41-1017-flash:sm80`。
   注意：即使能拉，它也只算整包的替代品，不能替代「配方 + `.so`」。
4. **完整离线包未导出**（合计压缩后约 32~40 GB）—— 这是应急物资，优先级低于第 1 条。
   跑 `tools/engine-bundle.sh save` 之前它会先算剩余空间，不够会拒绝。
5. ~~**A/B 的 74 行补丁来源不明**~~ → **已查明（2026-11-14）**：它就是
   `LocalFrameLab/deepseek-v41-cmp170hx-x99` 的 `patches/x99-stable.patch`（Apache-2.0），
   **逐行相同，我们一个字没改**。证据见 [`v2-dspark/PROVENANCE.md`](v2-dspark/PROVENANCE.md)。
   镜像里显示「未提交」是因为上游作者构建时直接 COPY 了带改动的工作树，不是没公开。
6. **C 的重建配方刚纠正（2026-11-14 夜）**：正确顺序是 `d63af5a` + 0001 + 0002 + 0005，
   已实测能复现镜像内 9 个 Python 文件的 blob。但 `build.sh` 仍未在干净机器上端到端跑过
   （需 GitHub + 基础镜像 + 40~90 分钟编译）。
7. **方案 C 的 0003 未进镜像这件事，是本次复核才发现的**：说明「归档快照 = 镜像」不能靠推测，
   只能对 blob。A/B 那边是 `git status -uall` 直接读的，可信度更高（无未跟踪文件）。
