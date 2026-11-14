#!/usr/bin/env bash
# build.sh — 从零重建旧引擎 V1 镜像（方案 C）
#
# 说明：这是**主路径** —— 仓库里只有 150 KB 的配方（源地址 + 5 个 patch + Containerfile），
#       重建不依赖任何镜像存档。离线整包（tools/engine-bundle.sh save）只是无外网时的应急副本。
#       本脚本没有在干净机器上端到端验证过，但每一步都与当初的构建脚本
#       archive/legacy/image-build/pp5-ssd-async-build.sh 对应，差异有三处：
#         1) 补丁来源改为本目录（不再从 GitHub raw 下载）
#         2) 应用 0001 + 0002 + 0005（实测配方；0003 不在镜像里，0004 含 0003 已弃用）
#         3) 自检增加 git blob 校验，确保产物与 11-02 那个镜像的源码逐字节相同
#
# 用法：  ./build.sh [输出镜像名]
# 依赖：  docker、git、g++（在镜像内安装）、能访问 GitHub 与基础镜像
set -Eeuo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${DSV41_BUILD_ROOT:-$HERE/.build}
IMAGE=${1:-local/deepseek-v41-official-pp5:rebuilt-$(date +%Y%m%d)}

SRC_URL=https://github.com/344303947/dsv41-flash-pp5-170hx.git
SRC_COMMIT=d63af5a472dc76b12d7a73d50a5af142844c15d1
BASE_IMAGE=lazymio/vllm-backport@sha256:bfe237ee0886d9b4b2357b84bfbe09805678cc80b2182dd2a87cefb89497342d
REFERENCE_COMMIT=29b03af5eedfac9af12dfa14e8cfa7e2eafc852a

log() { printf '\033[36m[build]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[build] 失败：\033[0m %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "没有 docker"
command -v git    >/dev/null || die "没有 git"

# ---------- 1. 取源码 ----------
mkdir -p "$ROOT"
if [ ! -d "$ROOT/src/.git" ]; then
  log "克隆 $SRC_URL"
  git clone --filter=blob:none "$SRC_URL" "$ROOT/src"
fi
if ! git -C "$ROOT/src" cat-file -e "${SRC_COMMIT}^{commit}" 2>/dev/null; then
  log "拉取指定 commit"
  git -C "$ROOT/src" fetch --depth 1 origin "$SRC_COMMIT"
fi
git -C "$ROOT/src" checkout -q --force --detach "$SRC_COMMIT"
git -C "$ROOT/src" clean -qfd
[ "$(git -C "$ROOT/src" rev-parse HEAD)" = "$SRC_COMMIT" ] || die "commit 不对"

# ---------- 2. 校验补丁 ----------
log "校验补丁 sha256"
(cd "$HERE/patches" && sha256sum -c <<'EOF'
16d8b755470052a7e939a10c97e6877396b55d06b24b548c166c7e5974dd5af7  0001-prime-pp-communicators-before-model-load.patch
5ec439e0d300cee47fa5ba9ee99c393d71161cae95131298ada31b0a51c658c1  0002-engram-ssd-port.patch
57c0ef0e380870336f8fd69e05b06fc2fdb71634942ecbae599f9e1e37f7d9ec  0003-dspark-construction-offload.patch
f8c6cb49b08a8af57996d1963edf1694a1dc5ad5e7baadad3af966c7d5fe11c2  0004-runtime-cumulative-snapshot.patch
30b53a49fd6401aaaf4c7b011d2c7f670927f9dce7e602f47b571ba90601773e  0005-pp5-lazy-receive.patch
EOF
) || die "补丁文件被改动过"

# ---------- 3. 应用补丁 ----------
# ⚠ 2026-11-14 实测纠正：镜像里只有 0001 + 0002 + 0005。
#   0003（dspark 构造下放）**不在镜像里**：镜像内那 3 个文件的 git blob
#   与 0003 的 pre-image 完全相等。0004 是含 0003 的历史快照，
#   只应用 0004 会重建出「比镜像多 3 个文件改动」的另一个引擎。
#   顺序必须是 0001 → 0002 → 0005（0005 是相对 0001 之后状态的增量）。
log "应用 0001 / 0002 / 0005"
for p in 0001-prime-pp-communicators-before-model-load.patch \
         0002-engram-ssd-port.patch \
         0005-pp5-lazy-receive.patch; do
  git -C "$ROOT/src" apply --check "$HERE/patches/$p" \
    || die "$p 应用失败：源码树不是 d63af5a 的干净状态？"
  git -C "$ROOT/src" apply "$HERE/patches/$p"
done

[ -s "$ROOT/src/csrc/cpu/dsv41/engram_ssd.cpp" ] || die "缺 engram_ssd.cpp"
grep -q '_pp_communicators_primed' "$ROOT/src/vllm/v1/worker/gpu_worker.py" || die "通信器预热改动缺失"
grep -q 'VLLM_PP5_LAZY_RECEIVE' "$ROOT/src/vllm/v1/worker/gpu_worker.py" || die "惰性接收改动缺失"

# ---------- 4. 构建上下文 ----------
log "准备构建上下文"
rm -rf "$ROOT/context"
mkdir -p "$ROOT/context/native"
cp -a "$ROOT/src/vllm" "$ROOT/context/vllm"
cp "$ROOT/src/csrc/cpu/dsv41/engram_ssd.cpp" "$ROOT/context/native/engram_ssd.cpp"
cp "$HERE/Containerfile" "$ROOT/context/Containerfile"

docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || {
  log "基础镜像不在本地，尝试拉取（需要网络）"
  docker pull "$BASE_IMAGE"
}
PKG_DIR=$(docker run --rm --network none --entrypoint python3 "$BASE_IMAGE" \
            -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
[ -n "$PKG_DIR" ] || die "无法确定 Python 包路径"

# ---------- 5. 构建镜像 ----------
log "开始构建 $IMAGE（约 5~10 分钟，编译 C++ 扩展 + 覆盖 30 GB 层）"
docker build --pull=false \
  --tag "$IMAGE" \
  --label local.deepseek.checkpoint=official-1017-unchanged \
  --label local.deepseek.pp=5 \
  --label local.deepseek.pp-layer-partition=8,8,8,8,8 \
  --label local.deepseek.engram=aiultrayang-O_DIRECT-Linux-AIO \
  --label local.deepseek.pp5-performance=lazy-receive \
  --label local.deepseek.reference-commit="$REFERENCE_COMMIT" \
  --label local.deepseek.rebuilt-from="$SRC_COMMIT" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg PKG_DIR="$PKG_DIR" \
  --file "$ROOT/context/Containerfile" "$ROOT/context"

# ---------- 6. 自检 ----------
log "自检：加载 C++ 扩展"
docker run --rm --network none --entrypoint python3 "$IMAGE" -c \
  'import ctypes, pathlib, vllm
p = pathlib.Path(vllm.__file__).parent / "libdsv41_cuda.so"
lib = ctypes.CDLL(str(p))
lib.dsv41_engram_ssd_version.restype = ctypes.c_char_p
print("engram_ssd_version =", lib.dsv41_engram_ssd_version())
print("vllm =", vllm.__version__)' || die "扩展加载失败"

# 与原镜像对齐的硬校验：blob 是内容哈希，与行尾/路径无关。
# qwen3_dspark.py 这一条尤其关键 —— 它一旦变成 ff2413d4，说明 0003 被误应用了。
log "自检：源码 blob 必须与 11-02 镜像一致"
docker run --rm --network none --entrypoint python3 "$IMAGE" -c '
import hashlib, pathlib, sys, vllm
root = pathlib.Path(vllm.__file__).parent
want = {
 "v1/worker/gpu_worker.py": "831a4f34decb278bfe75d838786f815700884722",
 "config/engram.py": "5a18015129be78abf790e0b82664765c231dc8a0",
 "model_executor/models/qwen3_dspark.py": "1ecdf5fa4096e5271101549a3b35529211ab47fa",
}
bad = 0
for f, exp in want.items():
    d = (root / f).read_bytes()
    got = hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest()
    print(("OK  " if got == exp else "BAD "), got[:12], f)
    bad += got != exp
sys.exit(1 if bad else 0)' || die "源码 blob 与原镜像不一致 —— 检查是否误用了 0003/0004"

log "完成：$IMAGE"
log "用 ./deploy.sh pp5-legacy 部署时，设 DSV41_IMAGE=$IMAGE 覆盖默认镜像名"
