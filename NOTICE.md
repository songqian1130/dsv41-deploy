# NOTICE · 第三方代码与许可

本仓库**不是**从零实现的推理引擎。它打包、参数化并实测了别人的成果。
所有第三方文件的版权归原作者，本仓库只做原样存档与复现说明。

## 1. 补丁与镜像

| 文件 | 原作者 | 许可 | 本仓库的处理 |
|---|---|---|---|
| `engine/v2-dspark/patches/upstream-x99-stable.patch`<br>`engine/v2-dspark/patches/0001-host-staging-and-cache-interval.patch` | [LocalFrameLab](https://github.com/LocalFrameLab/deepseek-v41-cmp170hx-x99) | **Apache-2.0** | 原样存档。许可证全文见同目录 `LICENSE` |
| `engine/v1-legacy/patches/0001-prime-pp-communicators-before-model-load.patch`<br>`engine/v1-legacy/patches/0002-engram-ssd-port.patch` | [aiultrayang](https://github.com/aiultrayang/deepseek-v41-flash-cmp170hx) | **上游未声明** | 原样存档，仅为重建可行性 |
| `engine/v1-legacy/Containerfile` | 同上 | **上游未声明** | 逐字节副本 |

> ⚠️ 上表第 2、3 行的文件，**上游仓库没有 LICENSE 文件**。本仓库不以任何方式对它们再许可，
> 也不主张任何权利。存档目的是让「这个引擎怎么造出来的」可被审计与复现。
> 如果原作者认为不妥，提出即可删除。

## 2. 引擎源码

| 项目 | 地址 | 许可 |
|---|---|---|
| vLLM | <https://github.com/vllm-project/vllm> | Apache-2.0 |
| `deepseek_v4_1` 模型实现 | <https://github.com/buliaoyin/vllm-backport> | 上游未声明 |
| 旧引擎 5 卡 PP 方案 | <https://github.com/344303947/dsv41-flash-pp5-170hx> | 上游未声明 |

**本仓库不包含上述源码的任何副本** —— 只有 commit 地址和补丁，构建时按地址拉取。

## 3. 模型权重

[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://modelscope.cn/models/deepseek-ai/DeepSeek-V4.1-Flash-1017)，MIT。
**权重不在本仓库里**（475 GiB），部署时自行准备。

## 4. 本仓库自己的部分

本仓库原创的是：三个方案的参数与实测数据、`bench/` 与 `tools/` 下的脚本、文档、
以及旧引擎的 `0003` / `0004` / `0005` 三个补丁。

这部分**暂未声明开源许可证**。在作者明确之前，默认保留所有权利；
如需引用数据或脚本，请先开 issue。
