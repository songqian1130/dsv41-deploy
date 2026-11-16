# 外部依赖清单（LINKS）

> 这份文件存在的唯一理由：**凡是没写地址的东西，半年后都等于不存在。**
>
> 补丁、配方、镜像、权重，只要没有「完整 URL + 钉死的版本 + sha256」，它就是一段无法追溯的字节。
> 这里把每一条都钉到 commit / digest / sha256，并且可以用一条命令重验：
>
> ```bash
> ./tools/verify-sources.sh
> ```

**最后一次实测：2026-11-15 00:30 (+08:00)**。本机直连下 GitHub / ModelScope / PyPI 可达（10 项全 OK）；
**Docker Hub 与 HuggingFace 直连不通、本地代理 7890 又没启动**（`HTTP 000`，实测见第 6.1 节），
所以镜像和 HF 那几条只能标「未验证」。**这是网络问题，不是链接失效。**

---

## 0. 两类东西：别人的、我们改的

这一步必须先说清，否则「没有地址」会被误读成「丢件了」。

| 类别 | 是什么 | 几个 | 它们的「地址」怎么写 |
|---|---|---|---|
| **别人的** | 别人发布的东西，我们**逐字节复制**过来 | 4 个 + 1 份副本 | 完整 URL + commit/digest + sha256 |
| **我们改的** | 在别人的 commit 基础上**我们自己动的** | 3 补丁 + 1 镜像 + 1 份反推的 Containerfile | 没有第三方 URL，也不该有。身份 = **基座 commit + 本仓库这份 patch + 再生命令** |

> 「别人的」那 1 份副本是派生件：`0001-host-staging-and-cache-interval.patch` 与上游
> `x99-stable.patch` 逐行相同，只差 23 字节 patch 元数据。第 2 节有逐条对照。

**「我们改的」这几个** —— 就是「基于原始仓库以后我们自己尝试的改动」：

| 我们改的 | 基座（这个有 URL） | 谁写的 | 现在在哪 |
|---|---|---|---|
| `0003-dspark-construction-offload.patch` | `d63af5a` | **我们**（2026-10-31 r11 构造 UVA 实验） | 本仓库；`archive/legacy/patches/` 有同日原件，sha256 相同 |
| `0004-runtime-cumulative-snapshot.patch` | `d63af5a` | **我们**（2026-11-02 构建工作区快照） | 只有本仓库这份，不可再生 |
| `0005-pp5-lazy-receive.patch` | `d63af5a` + 0001 + 0002 | **我们**（惰性接收） | 本仓库；可从镜像 diff 再生 |
| C 的 `local/...` 镜像 | 上面三者 + 0002 | **我们**（本机 build） | 只有本机 docker，靠离线包搬运 |

所以正确的说法不是「这四个没有地址」，而是：**它们的地址 = 上游 commit 的 URL + 本仓库里这份 patch。**
基座 `d63af5a` 是有 URL 的（第 1 节第一行），改动部分在本仓库里可读、可校验、可重建。

> 只有一种情况才算真丢件：**既没有上游 URL，也没有本地副本，还没有再生命令**。目前一个都没有。

## 0.1 速查表

| 你要找的 | 地址 | 钉住的版本 |
|---|---|---|
| 方案 A/B 的引擎源码 | <https://github.com/buliaoyin/vllm-backport> | `c3d3104de0163c389d26d9fcba0180f55ad7964d` |
| A/B 那 74 行补丁的**原件** | <https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch> | Apache-2.0，sha256 `16f2a4606dae…` |
| 方案 C 的引擎源码 | <https://github.com/344303947/dsv41-flash-pp5-170hx> | `d63af5a472dc76b12d7a73d50a5af142844c15d1` |
| C 的 0001 / 0002 / Containerfile 出处 | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/> | `29b03af5eedfac9af12dfa14e8cfa7e2eafc852a` |
| 模型权重 | <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017> | rev `f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad` |
| A/B 运行的镜像 | `docker pull 18gogogo/170hx4-256r-dsv41-1017-flash:sm80` | ID `sha256:4e08e1c9…`⚠️未验证 |
| C 运行的镜像 | **没有任何地址**（`local/` 前缀，本机构建、从未推送） | ID `sha256:4f7370a2…` |

---

## 1. 源码仓库（5 个，全部公开可达）

| 仓库 | 地址 | 钉住的版本 | 用途 | 2026-11-14 |
|---|---|---|---|---|
| `buliaoyin/vllm-backport` | <https://github.com/buliaoyin/vllm-backport> | commit `c3d3104de0163c389d26d9fcba0180f55ad7964d` | A/B 的 `vllm/models/deepseek_v4_1/` 整棵模型树 | HTTP 200 ✅ |
| `LocalFrameLab/deepseek-v41-cmp170hx-x99` | <https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99> | `main`（补丁文件稳定，仍建议按 sha256 校验） | A/B 的 74 行补丁原件、4 卡基准 | HTTP 200 ✅ |
| `344303947/dsv41-flash-pp5-170hx` | <https://github.com/344303947/dsv41-flash-pp5-170hx> | commit `d63af5a472dc76b12d7a73d50a5af142844c15d1` | C 的源码基线 | HTTP 200 ✅ |
| `aiultrayang/deepseek-v41-flash-cmp170hx` | <https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx> | commit `29b03af5eedfac9af12dfa14e8cfa7e2eafc852a`，目录 `pp6-ssd/` | C 的 0001 / 0002 / Containerfile 出处 | HTTP 200 ✅ |
| `zebgop-ops/dsv41flash-pp` | <https://github.com/zebgop-ops/dsv41flash-pp> | commit `e29e9fdc9bbc419b34301fc581c2da9f9062164a`（2026-11-07） | **L 的全部源码**：`overlay/` 37 个 `patch_*.py` + `hybrid/cpu_moe.cpp` + `engram_ssd/row_store.cpp` + 启动脚本 | HTTP 200 ✅ · **Apache-2.0** |

**取法（两个都是只取那一个 commit，不要 clone 默认分支）：**

```bash
# C 的源码基线
curl -L -o src.tgz https://codeload.github.com/344303947/dsv41-flash-pp5-170hx/tar.gz/d63af5a472dc76b12d7a73d50a5af142844c15d1

# A/B 的源码（build.sh 里用的就是这个）
git clone https://github.com/buliaoyin/vllm-backport.git && git -C vllm-backport checkout c3d3104de0
```

> ⚠️ **两个分支名指向同一处，但都已经前进了**：`codex/deepseek-v41-cpu-hybrid` 和 `sm80-optimize`
> 在 2026-11-14 的 HEAD 都是 `12d7c186`，不是 `c3d3104de`。**跟分支 = 拿到另一个引擎**，必须按 commit 取。

---

## 2. 补丁与配方文件：逐个的出处

仓库里一共 **7 个补丁文件**（V1 五个 + V2 两个），另有 1 个 `Containerfile` 来自上游。
**能从公网直接取到内容的只有 4 条**（V1 的 0001 / 0002、Containerfile、V2 的 `upstream-x99-stable.patch`），
其余必须靠仓库内副本或按 2.1 节再生。

> 每个 `patches/` 目录里还各有一份纯文本 **`ORIGIN.txt`**（`v1-legacy/patches/`、`v2-dspark/patches/`），
> 内容是同样的地址 + sha256。它的用途是：**只把几个 `.patch` 拷给别人时，出处也一起走了**。

| 本地文件 | 出处（完整地址） | 校验 | 状态 |
|---|---|---|---|
| `engine/v1-legacy/patches/0001-prime-pp-communicators-before-model-load.patch` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/patches/0001-prime-pp-communicators-before-model-load.patch> | `16d8b7554700…` 与上游**逐字节相同** | 200 ✅ |
| `engine/v1-legacy/patches/0002-engram-ssd-port.patch` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/patches/engram-ssd-port.patch> | `5ec439e0d300…` 与上游**逐字节相同** | 200 ✅ |
| `engine/v1-legacy/Containerfile` | <https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd/build/Containerfile> | `5af7be182795…` 与上游**逐字节相同**（我们没改过它） | 200 ✅ |
| `engine/v2-dspark/patches/upstream-x99-stable.patch` | <https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch> | `16f2a4606dae…`（上游原件） | 200 ✅ |
| `engine/v2-dspark/patches/0001-host-staging-and-cache-interval.patch` | 同上（内容相同，仅 `index` 缩写与行尾空格不同，见 [`PROVENANCE.md`](v2-dspark/PROVENANCE.md)） | `7448fefe8ccc…` | 派生 |
| `engine/v1-legacy/patches/0003-dspark-construction-offload.patch` | **我们改的**（2026-10-31 r11 构造 UVA 实验；上游该 commit 只有 0001 与 engram-ssd-port）。基座：<https://github.com/344303947/dsv41-flash-pp5-170hx> @ `d63af5a` | `57c0ef0e3808…` | 再生 ↓ |
| `engine/v1-legacy/patches/0004-runtime-cumulative-snapshot.patch` | **我们改的**（2026-11-02 构建工作区的工作树）。基座同上，内容 = 0001+0002+0003+0005 的合并 | `f8c6cb49b08a…` | 仅存档 |
| `engine/v1-legacy/patches/0005-pp5-lazy-receive.patch` | **我们改的**（惰性接收，`+37/−9`）。基座 = `d63af5a`+0001+0002，定义式见 2.1 | `30b53a49fd64…` | 再生 ↓ |

### 2.1 「我们改的」那三个：从哪来、怎么重建

**0003** —— 能从 `0004` 精确抽出（`0001/0002/0005` 都不碰那 3 个 dspark 文件，所以 `0004` 里这 3 个文件的 hunk 就是 `0003`）：

```bash
python3 - <<'PY'
import re
src = open('engine/v1-legacy/patches/0004-runtime-cumulative-snapshot.patch', encoding='utf-8', newline='').read()
keep = ('qwen3_dspark.py', 'nvidia/dspark.py', 'spec_decode/dspark/utils.py')
blocks = re.split(r'(?m)^(?=diff --git )', src)
sel = [b for b in blocks if b.startswith('diff --git') and any(k in b.split('\n')[0] for k in keep)]
open('0003.patch', 'w', encoding='utf-8', newline='').write(''.join(sel))   # newline='' 必须写，否则 Windows 下变 CRLF
PY
sha256sum 0003.patch   # 期望 57c0ef0e380870336f8fd69e05b06fc2fdb71634942ecbae599f9e1e37f7d9ec
```

> 2026-11-14 实测：**逐字节相同**。注意 `newline=''`，否则 Python 在 Windows 上会把 LF 写成 CRLF，
> 文件从 5805 B 变成 5929 B（多 124 字节），哈希对不上。

**0004** —— **无法再生**，只能靠仓库里这份。它是历史件，**不进镜像**，唯一用途就是上面那条（抽 0003）。

**0005** —— 从「镜像内的文件」减去「`d63af5a + 0001 + 0002` 的文件」得到，这是它唯一的定义：

```bash
# 1) 干净源码 + 0001 + 0002（autocrlf 必须关，否则行尾会被改）
git -c core.autocrlf=false clone https://github.com/344303947/dsv41-flash-pp5-170hx.git src
git -C src checkout d63af5a472dc76b12d7a73d50a5af142844c15d1
for p in 0001-prime-pp-communicators-before-model-load.patch 0002-engram-ssd-port.patch; do
  git -C src -c core.autocrlf=false apply -p1 "$PWD/engine/v1-legacy/patches/$p"
done

# 2) 从镜像取出实际在跑的那份
docker run --rm --network none --entrypoint cat \
  local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102 \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py > img_gpu_worker.py

# 3) 两边摆成 a/ b/ 目录做 no-index diff（这样 -p1 能直接 apply）
mkdir -p mk/a/vllm/v1/worker mk/b/vllm/v1/worker
cp src/vllm/v1/worker/gpu_worker.py mk/a/vllm/v1/worker/
cp img_gpu_worker.py                mk/b/vllm/v1/worker/gpu_worker.py
(cd mk && git diff --no-index --no-prefix a/vllm/v1/worker/gpu_worker.py b/vllm/v1/worker/gpu_worker.py) \
  > 0005-pp5-lazy-receive.patch
sha256sum 0005-pp5-lazy-receive.patch   # 期望 30b53a49fd6401aaaf4c7b011d2c7f670927f9dce7e602f47b571ba90601773e
```

---

## 3. 镜像（5 个 tag）

| tag | 地址 | 镜像 ID | 2026-11-14 |
|---|---|---|---|
| `18gogogo/170hx4-256r-dsv41-1017-flash:sm80` | Docker Hub：<https://hub.docker.com/r/18gogogo/170hx4-256r-dsv41-1017-flash> | `sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157` | ⚠️ 页面 000（**本机代理未运行**，非链接失效），是否公开未验证 |
| `local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102` | **我们改的**（`local/` 前缀，从未推送）。身份 = 本仓库的 `engine/v1-legacy/`（配方 + 补丁）+ `Containerfile` | `sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1` | 源码重建 / 离线包 |
| `vllm/vllm-openai:deepseekv41-flash-0909`（**L 用**） | Docker Hub 官方组织镜像 tag；身份 = 官方 vLLM + `zebgop-ops` overlay 挂载（**不改镜像**） | `sha256:00d577a6a632…`（30.4 GB，vLLM `0.1.dev20904+g179dd0fa9`） | ⚠️ 未验证（本机 Docker Hub 直连不通） |
| `lazymio/vllm-backport:v0.13.0-sm80`（C 的基础镜像） | <https://hub.docker.com/r/lazymio/vllm-backport> | digest `sha256:bfe237ee0886d9b4b2357b84bfbe09805678cc80b2182dd2a87cefb89497342d` | ⚠️ 未验证（原机上还在） |
| `nvidia/cuda:13.0.3-devel-ubuntu24.04`（A/B 的基础镜像） | <https://hub.docker.com/r/nvidia/cuda> | tag 名是**推断**（实测环境变量 CUDA 13.0.3 / Ubuntu 24.04.4） | ⚠️ 未验证 |

验证镜像真身（不需要 pull）：

```bash
docker manifest inspect 18gogogo/170hx4-256r-dsv41-1017-flash:sm80 | head -20
```

---

## 4. 模型权重

| 项 | 值 |
|---|---|
| 仓库 | <https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017> |
| revision | `f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad` |
| 许可 | MIT（非 gated） |
| 体积 | **475.2 GiB / 48 分片**（= 510.2 GB，两个数字是同一件事，单位不同） |
| HF 对应 | <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017>，rev `1bae0e1088993b9b82de08f59de5be5a34e42170`（本机 000，未验证） |

```bash
modelscope download --model deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  --revision f04a978641fa9bfe6f448ec5da3b2e1c2520a6ad \
  --local_dir /run/media/xpz/models/DeepSeek-V4.1-Flash-1017-official
```

---

## 5. Python 依赖

| 项 | 地址 / 文件 | 校验 |
|---|---|---|
| torch 2.13.0（cu130 索引） | <https://download.pytorch.org/whl/cu130> | 200 ✅ |
| 其余 197 个包 | `engine/v2-dspark/requirements.freeze.txt`（200 行） | sha256 `7c823eb81562…` |
| PyPI | <https://pypi.org/simple/> | — |

版本全部冻结在 freeze 文件里，**不要用 `pip install -r requirements.txt` 重新解析**——解析结果会漂。

---

## 6. 一键复核

```bash
./tools/verify-sources.sh          # 逐条抓取 + 比 sha256
./tools/verify-sources.sh --quiet  # 只看结论
```

| 状态 | 含义 | 该做什么 |
|---|---|---|
| `OK` | 200，且 sha256 与归档一致 | 什么都不用做 |
| `DRIFT` | 拉到了，但 sha256 变了 | **上游改过内容**。不要直接用，先看 diff 再决定是否更新归档 |
| `GONE` | 404 / 410 | 地址失效。补丁靠仓库副本，镜像靠离线包 |
| `NET` | 000 / 超时 | 只是本机连不上，不是地址的问题。换台机器再验 |
| `SKIP` | 需要 docker，本脚本不碰 | 用第 3 节的 `docker manifest inspect` 手工验 |

> 仓库与 commit 的检查走 `api.github.com/repos/<repo>/commits/<sha>`，不走网页 ——
> 2026-11-14 夜间 `github.com` 的 HTML 走不通（`CONNECT tunnel failed 502`），
> 但 `api.github.com` 与 `raw.githubusercontent.com` 正常。所以「网页打不开」**不等于**「链接失效」，
> 脚本的 `NET` 状态就是用来区分这两件事的。

### 6.1 为什么本机连不上 Docker Hub / HuggingFace（2026-11-15 00:30 实测）

**不是「被代理拦截」，是这两个域名直连不通、而代理又没启动。** 实测三种走法：

| 目标 | 直连（`--noproxy '*'`） | 走 `9455` | 走 `7890` |
|---|---|---|---|
| `api.github.com` | **200** ✅ | 502 / 000 | 无服务 |
| `registry-1.docker.io/v2/` | **000** ❌ | 000 | 无服务 |
| `hub.docker.com` | **000** ❌ | 000 | 无服务 |
| `huggingface.co` | **000** ❌ | 000 | 无服务 |

端口探测：

| 检查项 | 结果 |
|---|---|
| `7890`（Clash / FlClash 默认混合端口） | **无服务监听** |
| `7891` / `7897` / `1080` / `10808` / `10809` / `8889` | 全部无服务监听 |
| `9455` | 有服务在听，**但那是 WorkBuddy 注入的沙箱代理**（环境变量 `http_proxy`），只放行白名单域名，到不了 Docker Hub |
| 进程 | 只有 `FlClashHelperService`（PID 6024）—— FlClash 的**后台助手服务**，主程序没启动 |
| 服务器直连 `registry-1.docker.io` | 超时（`000`）—— 该机房到 Docker Hub 也不通 |

**所以**：GitHub 能直连（这就是为什么补丁全都能拉到、sha256 全对），
Docker Hub 和 HuggingFace 必须过代理，而 7890 现在没起来。

要验镜像是否公开，先启动 FlClash 主程序，然后：

```bash
curl --proxy http://127.0.0.1:7890 -sS -o /dev/null -w "%{http_code}\n" https://hub.docker.com/
docker manifest inspect 18gogogo/170hx4-256r-dsv41-1017-flash:sm80   # docker 读系统代理，不读 shell 变量
```

`tools/verify-sources.sh` 现在会自动探测 7890/7897/1080/10809/7891/8889，
探测到就用，探测不到才直连；也可 `DSV41_PROXY=none` 强制直连、`DSV41_PROXY=http://...` 指定。

---

## 附：这次核对发现的四个坑（都已被上面钉住）

1. **上游 `0002` 的文件名没有 `0002-` 前缀**，就叫 `engram-ssd-port.patch`。
   照着我们改过的名字去拼 URL 会 404。
2. **`0003` 在上游不存在 —— 因为它本来就是我们写的**：`29b03af5` 的 `pp6-ssd/patches/` 目录只有 2 个文件
   （2026-11-14 用 API 列过）。原件是 `archive/legacy/patches/dspark-construction-offload.patch`
   （时间戳 **2026-10-31 13:21**，sha256 与入库版本完全相同），配套件是
   `archive/legacy/image-build/pp5-dspark-r11-construction-uva.Containerfile` + `probes/patch-dspark-construction-uva.py`（11-01 07:17）。
   **归属：本仓库 r11 实验，不是 aiultrayang。**早前文档写成 aiultrayang 是错的，已改。
3. **`engine/v1-legacy/Containerfile` 是上游原件的逐字节副本**（`5af7be18…`），不是我们写的。
   之前文档没写这点，容易被误认成本仓库的产物。
4. **`archive/legacy/image-build/reference/` 是空的** —— 当初从 raw 下载的三份文件没有留下来，
   只剩我们后来归档的副本。这正是 sha256 必须写进文档的原因：原件没了以后，哈希是唯一能证明
   「我们手上这份就是当初那份」的东西。
