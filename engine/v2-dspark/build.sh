#!/usr/bin/env bash
# build.sh — 从零重建新引擎 V2 镜像（方案 A / B）
#
# 说明：这是**主路径** —— 仓库里只有 236 KB 的配方（源地址 + patch + freeze + Containerfile），
#       重建不依赖任何镜像存档。离线整包（tools/engine-bundle.sh save）只是无外网时的应急副本。
#       ⚠️ 唯一不在仓库里的是 191 MB 编译产物；重建后可用 load-so 覆盖成实测那一份，
#          这样 100.1 tok/s 等基准数字仍可沿用（编译时间不会因此减少）。
#       ⚠️ 没有在干净机器上端到端验证过 —— 原镜像没有 Dockerfile，Containerfile 是
#       按 docker history 反推 + 从运行中的镜像导出的依赖清单写出来的。
#
# 与原始作者的构建方式只差一处：他把带未提交改动的整棵树 COPY 进镜像（补丁是「烤」进去的），
# 我们这里显式 git apply，效果相同，但**可审计**。
#
# 用法：  ./build.sh [输出镜像名]
# 依赖：  docker、git、外网（GitHub + Docker Hub + PyPI + download.pytorch.org）
# 磁盘：  构建缓存 + 基础镜像 + 依赖约需 60~80 GB 空闲
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${DSV41_BUILD_ROOT:-$HERE/.build}
IMAGE=${1:-local/dsv41-v2-dspark:rebuilt-$(date +%Y%m%d)}

SRC_URL=https://github.com/buliaoyin/vllm-backport.git
SRC_COMMIT=c3d3104de0163c389d26d9fcba0180f55ad7964d
BASE_IMAGE=${DSV41_V2_BASE:-nvidia/cuda:13.0.3-devel-ubuntu24.04}
PATCH=$HERE/patches/0001-host-staging-and-cache-interval.patch
PATCH_SHA=7448fefe8ccc05da5ae4ce53bbf59866a954946c243bc04b38ac249da0e9e8d6

log() { printf '\033[36m[build]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[build] 注意：\033[0m %s\n' "$*"; }
die() { printf '\033[31m[build] 失败：\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "没有 docker"
command -v git    >/dev/null || die "没有 git"

# ---------- 1. 取源码 ----------
mkdir -p "$ROOT"
if [ ! -d "$ROOT/src/.git" ]; then
  log "克隆 $SRC_URL（完整克隆，不用 --filter：.git 要随上下文进镜像给 setuptools_scm 算版本号）"
  git clone "$SRC_URL" "$ROOT/src" \
    || die "克隆失败。仓库是第三方的，若已删除，本脚本无法工作 —— 只能用离线镜像包"
fi
if ! git -C "$ROOT/src" cat-file -e "${SRC_COMMIT}^{commit}" 2>/dev/null; then
  log "拉取指定 commit"
  git -C "$ROOT/src" fetch origin "$SRC_COMMIT" || die "fetch $SRC_COMMIT 失败"
fi
git -C "$ROOT/src" checkout -q --force --detach "$SRC_COMMIT"
git -C "$ROOT/src" clean -qfdx -e .venv      # 保留就地创建的 venv（如果复用构建目录）
[ "$(git -C "$ROOT/src" rev-parse HEAD)" = "$SRC_COMMIT" ] || die "commit 不对"

# ---------- 2. 校验并应用补丁 ----------
log "校验补丁 sha256"
echo "$PATCH_SHA  $PATCH" | sha256sum -c - >/dev/null || die "补丁被改动过，sha256 不匹配"

log "应用 74 行补丁（3 文件：expert_cache / hybrid / hybrid_runtime）"
git -C "$ROOT/src" apply --check "$PATCH" \
  || die "补丁应用失败：源码树不是 $SRC_COMMIT 的干净状态？"
git -C "$ROOT/src" apply "$PATCH"

# 三项自证：缺任何一项就不是测出 82.5 tok/s 的那个引擎
grep -q 'DSV41_HOST_STAGING_CACHE' "$ROOT/src/vllm/models/deepseek_v4_1/expert_cache.py" \
  || die "缺 host staging 改动"
grep -q 'DSV41_FORCE_LOCAL_CACHE' "$ROOT/src/vllm/models/deepseek_v4_1/hybrid.py" \
  || die "缺 cache_decode_interval / FORCE_LOCAL_CACHE 改动"
grep -q 'decode_interval' "$ROOT/src/vllm/models/deepseek_v4_1/hybrid_runtime.py" \
  || die "缺 decode_interval 读配置的改动"

# ---------- 3. 构建上下文 ----------
log "准备构建上下文（含 .git，版本号依赖它）"
rm -rf "$ROOT/context"
mkdir -p "$ROOT/context"
cp -a "$ROOT/src" "$ROOT/context/vllm-backport"
rm -rf "$ROOT/context/vllm-backport/.venv"
cp "$HERE/Containerfile" "$ROOT/context/Containerfile"
cp "$HERE/requirements.freeze.txt" "$ROOT/context/requirements.freeze.txt"

# ---------- 4. 构建 ----------
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  log "基础镜像不在本地，拉取 $BASE_IMAGE（约 9 GB）"
  docker pull "$BASE_IMAGE"
}

log "开始构建 $IMAGE —— 预计 40~90 分钟（编译 5 个 CUDA 扩展）"
docker build --pull=false \
  --tag "$IMAGE" \
  --label dsv41.engine=v2-dspark \
  --label dsv41.source="$SRC_URL@$SRC_COMMIT" \
  --label dsv41.patch="$PATCH_SHA" \
  --label dsv41.model=official-1017-unchanged \
  --label dsv41.rebuilt-by=engine/v2-dspark/build.sh \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --file "$ROOT/context/Containerfile" "$ROOT/context"

# ---------- 5. 自检 ----------
log "自检"
EXPECT_VER="0.1.dev21380+gc3d3104de"
docker run --rm --network none --entrypoint /workspace/vllm-backport/.venv/bin/python "$IMAGE" -c "
import sys, vllm
print('vllm =', vllm.__version__)
assert vllm.__version__.startswith('$EXPECT_VER'), '版本号与基准镜像不一致，测速数字不可比'
import pathlib
so = sorted(p.name for p in pathlib.Path(vllm.__file__).parent.glob('*.so'))
print('编译产物:', so)
assert '_C_stable_libtorch.abi3.so' in so and '_moe_C_stable_libtorch.abi3.so' in so
" || die "自检失败。若 .so 缺失，通常是 TORCH_CUDA_ARCH_LIST 或 ninja 没生效；若版本号是 0.0.0，是 .git 没进上下文"

log "完成：$IMAGE"
log "部署时用 DSV41_IMAGE=$IMAGE 覆盖默认镜像名"
warn "重建产物不保证与原镜像字节一致；用它测出来的数字请按新方案记录，不要与 82.5 tok/s 混比"
