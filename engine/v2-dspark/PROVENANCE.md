# 补丁归属：这 74 行是谁写的

> **结论：不是我们写的。它和 `LocalFrameLab/x99-stable.patch` 内容逐行相同，一字不差。**
> 2026-11-14 核实，此前文档写的「来源不明」是错的，已更正。

## 证据

| | 文件 | 大小 | sha256 |
|---|---|---:|---|
| 从镜像里导出的 | `patches/0001-host-staging-and-cache-interval.patch` | 8304 B | `7448fefe8ccc05da…` |
| 从 GitHub 下载的 | `patches/upstream-x99-stable.patch` | 8281 B | `16f2a4606dae0fc3…` |

来源：`https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches/x99-stable.patch`
（2026-11-14 23:32 下载，Apache-2.0）

逐行 diff 归一化后**只有两类差异，都在 patch 元数据，不在代码里**：

| 差异 | 数量 | 原因 |
|---|---:|---|
| `index 7b3ed95..0d1a121` vs `index 7b3ed95fc4..0d1a121e8b` | 3 处 | git `core.abbrev` 配置不同（9 位 vs 12 位） |
| 空 context 行的行尾空格（`" "` vs `""`） | 5 处 | 我们的从 `git diff` 直接导出；上游文件可能被编辑器去过行尾空白 |

**每一行代码、每一个 hunk 的起止位置、每一处增删都完全相同。** 23 字节的总差异 =
5 个行尾空格 + 3 个 index 行各多 3 字符 + 其余格式差异。

复核命令：

```bash
diff -u <(sed 's/^index .*/index X/' patches/upstream-x99-stable.patch) \
        <(sed 's/^index .*/index X/' patches/0001-host-staging-and-cache-interval.patch)
# 只会剩行尾空白的差异
```

## 三个文件改了什么（原作者的话）

LocalFrameLab 的 README 原话：**「相对上游固定提交的三个 Python 文件修改，包含本机原有本卡缓存开关」**。

| 文件 | 改动 | 作用 |
|---|---|---|
| `expert_cache.py` | +43 −4 | `DSV41_HOST_STAGING_CACHE=1` 时走 pinned 主机暂存（固定地址，跨 graph replay 不变） |
| `hybrid.py` | +11 −0 | `cache_decode_interval` 必须为正整数；`DSV41_FORCE_LOCAL_CACHE` 把非本卡容量清零 |
| `hybrid_runtime.py` | +4 +16 | `decode_interval` 改为读配置；`_clear_hybrid_graphs_and_cache` 里轮换 CUDA Graph pool |

三个文件合计 **+74 / −5**，正是我们一直在说的「74 行」。

## 为什么两份都留着

- `0001-*.patch` 是**我们实际使用的那一份**（从运行中的镜像 `git diff` 导出，sha256 与镜像内工作树
  逐字节相等，`git apply -R --check` 通过）。`build.sh` 校验的是它的 sha256。
- `upstream-x99-stable.patch` 是**上游原件**，用来证明「同一份东西」，也是万一镜像和我们的导出都丢了
  时的独立副本。

两份内容等价，`build.sh` **只用 0001**；upstream 那份不参与构建，只作证据。

## 合规

- 上游 `LocalFrameLab/deepseek-v41-cmp170hx-x99` 以 **Apache-2.0** 发布，允许再分发与修改，
  要求保留版权与许可声明。本仓库保留原文件与来源 URL 即满足该要求。
- 我们**没有修改**这 74 行的内容。如果将来需要改（例如适配 5 卡），应当另开一个补丁叠加，
  而不是改 0001 —— 否则 sha256 变化会让 `build.sh` 的自校验失去意义，也会让「和上游一致」这个
  可核对的性质消失。

## 相关文件

- [`SOURCE.md`](SOURCE.md) —— 镜像、commit、构建层序
- [`../../engine/REBUILD.md`](../../engine/REBUILD.md) —— 从源码重建要做的 4 件事
- 上游仓库：`https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99`
