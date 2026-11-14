#!/usr/bin/env bash
# engine-bundle.sh — 引擎镜像的离线打包 / 还原（整包，或只打包 191 MB 编译产物）
#
#   ./tools/engine-bundle.sh list
#   ./tools/engine-bundle.sh save v2-dspark --out /run/media/xpz/models/engine-bundles
#   ./tools/engine-bundle.sh save v1-legacy --out /mnt/usb
#   ./tools/engine-bundle.sh load /path/v2-dspark-....tar.zst
#   ./tools/engine-bundle.sh verify v2-dspark
#   ./tools/engine-bundle.sh save-so v2-dspark --out ./engine/bundles   # 只导 .so（~191 MB）
#   ./tools/engine-bundle.sh load-so x.tar.zst --into <镜像>            # 覆盖 .so 后 commit
#
# 定位：整包（save/load）是**无外网时的应急副本**，20~25 GB，永不进 git。
#       真正的可复现主路径是 engine/<引擎>/build.sh（源码 + patch + freeze，236 KB）。
#       save-so 补的是唯一不可再生的那 191 MB 编译产物 —— 它不省编译时间，
#       但能让重建出来的引擎在性能上等同原引擎，历史基准数字可直接沿用。
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

declare -A IMAGE=(
  [v2-dspark]="18gogogo/170hx4-256r-dsv41-1017-flash:sm80"
  [v1-legacy]="local/deepseek-v41-official-pp5:aiultrayang-async-ssd-pp5-lazy-20261102"
)
declare -A IMAGE_ID=(
  [v2-dspark]="sha256:4e08e1c9231ea62355e58a29702970c2c993dc7f2b6993b191efea1ddf100157"
  [v1-legacy]="sha256:4f7370a2a284f3def000accec49ed92be7d082e7e9a5e79c1f12e82b03b097b1"
)
declare -A SCHEMES=(
  [v2-dspark]="pp5-fast pp4-t16"
  [v1-legacy]="pp5-legacy"
)

log()  { printf '\033[36m[bundle]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[bundle]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[bundle] 失败：\033[0m %s\n' "$*" >&2; exit 1; }

need_docker() { command -v docker >/dev/null || die "没有 docker"; }

usage() {
  sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

[ $# -ge 1 ] || usage
CMD=$1; shift

case "$CMD" in

list)
  printf '%-12s %-18s %-12s %s\n' 引擎 用于方案 体积 镜像
  printf '%-12s %-18s %-12s %s\n' ---- -------- ---- ----
  for k in v2-dspark v1-legacy; do
    sz="—"
    if command -v docker >/dev/null && docker image inspect "${IMAGE[$k]}" >/dev/null 2>&1; then
      sz=$(docker image inspect "${IMAGE[$k]}" --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}')
    fi
    printf '%-12s %-18s %-12s %s\n' "$k" "${SCHEMES[$k]}" "$sz" "${IMAGE[$k]}"
  done
  echo
  echo "说明："
  echo "  v2-dspark  方案 A/B 共用；第三方 fork 镜像，是否公开可拉未验证"
  echo "  v1-legacy  方案 C；local/ 前缀，从未推送，重建走 engine/v1-legacy/build.sh"
  echo "  来源与补丁：engine/<引擎>/SOURCE.md"
  echo "  可复现主路径：engine/<引擎>/build.sh（236 KB 配方，不依赖镜像存档）"
  echo "  本脚本的 save/save-so 只是把「无外网时的应急副本」和「唯一不可再生的 191 MB .so」留一份"
  ;;

save)
  [ $# -ge 1 ] || die "用法：save <v2-dspark|v1-legacy> [--out DIR] [--level 1-19]"
  name=$1; shift
  [ -n "${IMAGE[$name]:-}" ] || die "未知引擎 $name（可选：v2-dspark v1-legacy）"
  out=${DSV41_BUNDLE_DIR:-$ROOT/engine/bundles}
  level=3
  while [ $# -gt 0 ]; do
    case "$1" in
      --out)   out=${2:?--out 需要目录}; shift ;;
      --level) level=${2:?--level 需要 1-19}; shift ;;
      *) die "未知参数 $1" ;;
    esac
    shift
  done

  need_docker
  img=${IMAGE[$name]}
  docker image inspect "$img" >/dev/null 2>&1 || die "本机没有镜像 $img"
  actual=$(docker image inspect "$img" --format '{{.Id}}')
  if [ "$actual" != "${IMAGE_ID[$name]}" ]; then
    warn "镜像 ID 与记录不一致！"
    warn "  记录 ${IMAGE_ID[$name]}"
    warn "  实际 $actual"
    warn "  —— 继续打包，但打出来的东西不是我们测过的那一版"
  fi

  mkdir -p "$out" || die "无法创建 $out"
  avail=$(df -P -B1 "$out" | awk 'NR==2{print $4}')
  size=$(docker image inspect "$img" --format '{{.Size}}')
  # zstd -3 对这类镜像通常压到 50~60%，留 2 GB 余量
  need=$(awk -v s="$size" 'BEGIN{printf "%d", s*0.65 + 2*1024^3}')
  if [ "$avail" -lt "$need" ]; then
    die "空间不够：$out 剩余 $(awk -v a="$avail" 'BEGIN{printf "%.1f GB", a/1e9}')，预计需要 $(awk -v n="$need" 'BEGIN{printf "%.1f GB", n/1e9}')。换 --out 指到更大的盘。"
  fi

  stamp=$(date +%Y%m%d-%H%M)
  base="$name-$stamp"
  tar="$out/$base.tar.zst"

  if command -v zstd >/dev/null; then
    log "导出 $img → $tar（zstd -$level -T0，约 10~25 分钟）"
    docker save "$img" | zstd -q -T0 -"$level" -o "$tar"
  else
    warn "没有 zstd，退回 gzip（更慢）"
    tar="$out/$base.tar.gz"
    docker save "$img" | gzip -1 > "$tar"
  fi

  log "计算 sha256"
  ( cd "$out" && sha256sum "$(basename "$tar")" > "$(basename "$tar").sha256" )
  sha=$(sha256sum "$tar" | awk '{print $1}')
  bytes=$(stat -c%s "$tar" 2>/dev/null || stat -f%z "$tar")

  cat > "$out/$base.manifest.json" <<JSON
{
  "engine": "$name",
  "image": "$img",
  "expected_image_id": "${IMAGE_ID[$name]}",
  "exported_image_id": "$actual",
  "image_id_match": $([ "$actual" = "${IMAGE_ID[$name]}" ] && echo true || echo false),
  "used_by_schemes": "$(echo ${SCHEMES[$name]})",
  "archive": "$(basename "$tar")",
  "archive_bytes": $bytes,
  "archive_sha256": "$sha",
  "compression": "$(command -v zstd >/dev/null && echo zstd-$level || echo gzip-1)",
  "exported_at": "$(date -Iseconds)",
  "exported_from": "$(hostname)"
}
JSON

  log "完成"
  echo "  包体  $tar"
  echo "  校验  $(basename "$tar").sha256"
  echo "  清单  $base.manifest.json"
  echo
  echo "换机器：./tools/engine-bundle.sh load $tar"
  ;;

save-so)
  [ $# -ge 1 ] || die "用法：save-so <v2-dspark|v1-legacy> [--out DIR]"
  name=$1; shift
  [ -n "${IMAGE[$name]:-}" ] || die "未知引擎 $name（可选：v2-dspark v1-legacy）"
  out=${DSV41_BUNDLE_DIR:-$ROOT/engine/bundles}
  while [ $# -gt 0 ]; do
    case "$1" in
      --out) out=${2:?--out 需要目录}; shift ;;
      *) die "未知参数 $1" ;;
    esac
    shift
  done
  need_docker
  img=${IMAGE[$name]}
  docker image inspect "$img" >/dev/null 2>&1 || die "本机没有镜像 $img"
  mkdir -p "$out" || die "无法创建 $out"

  stamp=$(date +%Y%m%d-%H%M)
  base="$name-so-$stamp"
  tar="$out/$base.tar.zst"

  log "从 $img 导出编译产物（只取 vllm 包内的 *.so）"
  # 容器内定位 vllm 包，把 .so 原样 tar 到 stdout；路径用绝对路径记进清单，
  # 这样 load-so 能精确放回同一位置，不需要猜 site-packages / editable 的区别。
  docker run --rm --network none --entrypoint bash "$img" -c '
    set -euo pipefail
    PY=""
    for c in python python3 /workspace/vllm-backport/.venv/bin/python; do
      if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
    done
    [ -n "$PY" ] || { echo "容器内找不到 python" >&2; exit 1; }
    VDIR=$("$PY" -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
    echo "[容器] vllm 包目录 $VDIR" >&2
    cd "$(dirname "$VDIR")"
    list=$(find "$(basename "$VDIR")" -name "*.so" -type f | sort)
    [ -n "$list" ] || { echo "没找到 .so" >&2; exit 1; }
    echo "$list" | while read -r f; do
      printf "[容器] %8.1f MB  %s\n" "$(awk -v b="$(stat -c%s "$f")" "BEGIN{print b/1048576}")" "$f" >&2
    done
    # 清单：size  sha256  绝对路径（3 列，空格分隔；.so 路径不含空格）
    echo "$list" | while read -r f; do
      printf "%s %s %s\n" "$(stat -c%s "$f")" "$(sha256sum "$f" | cut -d" " -f1)" "$(pwd)/$f"
    done > SO-MANIFEST.txt
    tar -cf - -T <(echo "$list"; echo SO-MANIFEST.txt)
  ' | zstd -q -T0 -19 -o "$tar" || die "导出失败"

  log "计算 sha256"
  ( cd "$out" && sha256sum "$(basename "$tar")" > "$(basename "$tar").sha256" )
  sha=$(sha256sum "$tar" | awk '{print $1}')
  bytes=$(stat -c%s "$tar" 2>/dev/null || stat -f%z "$tar")

  cat > "$out/$base.manifest.json" <<JSON
{
  "engine": "$name",
  "image": "$img",
  "expected_image_id": "${IMAGE_ID[$name]}",
  "used_by_schemes": "$(echo ${SCHEMES[$name]})",
  "kind": "shared-objects-only",
  "contains": "vllm 包内全部 *.so + SO-MANIFEST.txt（size/sha256/绝对路径）",
  "archive": "$(basename "$tar")",
  "archive_bytes": $bytes,
  "archive_sha256": "$sha",
  "note": "不省编译时间；作用是让重建产物在性能上等同原引擎，历史基准可直接沿用",
  "exported_at": "$(date -Iseconds)",
  "exported_from": "$(hostname)"
}
JSON

  log "完成"
  echo "  包体  $tar（$(awk -v b="$bytes" 'BEGIN{printf "%.0f MB", b/1048576}')）"
  echo "  校验  $(basename "$tar").sha256"
  echo "  清单  $base.manifest.json"
  echo
  echo "还原：./tools/engine-bundle.sh load-so $tar --into <build.sh 重建出的镜像>"
  ;;

load-so)
  [ $# -ge 1 ] || die "用法：load-so <xxx.tar.zst> --into <镜像> [--tag 新镜像名]"
  file=$1; shift
  into=""; newtag=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --into) into=${2:?--into 需要镜像名}; shift ;;
      --tag)  newtag=${2:?--tag 需要镜像名}; shift ;;
      *) die "未知参数 $1" ;;
    esac
    shift
  done
  [ -f "$file" ] || die "找不到 $file"
  [ -n "$into" ] || die "--into 必填：要覆盖哪个镜像（一般是 build.sh 刚重建出来的那个）"
  need_docker
  docker image inspect "$into" >/dev/null 2>&1 || die "本机没有镜像 $into"

  sumfile="$file.sha256"
  if [ -f "$sumfile" ]; then
    log "校验包体 sha256"
    ( cd "$(dirname "$file")" && sha256sum -c "$(basename "$sumfile")" ) || die "包体 sha256 不匹配"
  else
    warn "没有 .sha256，跳过包体校验"
  fi

  tmp=$(mktemp -d)
  cid=""
  cleanup() { [ -n "$cid" ] && docker rm -f "$cid" >/dev/null 2>&1 || true; rm -rf "$tmp"; }
  trap cleanup EXIT

  log "解开 $file"
  case "$file" in
    *.tar.zst) command -v zstd >/dev/null || die "没有 zstd"; zstd -dc "$file" | tar -xf - -C "$tmp" ;;
    *.tar.gz)  gzip -dc "$file" | tar -xf - -C "$tmp" ;;
    *.tar)     tar -xf "$file" -C "$tmp" ;;
    *) die "不认识的扩展名：$file" ;;
  esac
  [ -f "$tmp/SO-MANIFEST.txt" ] || die "包里没有 SO-MANIFEST.txt，不是 save-so 产物"

  log "逐个校验 .so"
  bad=0
  while read -r sz sum path; do
    case "${sz:-}" in ''|\#*) continue ;; esac
    if [ ! -f "$tmp/$path" ]; then warn "缺文件 $path"; bad=1; continue; fi
    got=$(sha256sum "$tmp/$path" | awk '{print $1}')
    [ "$got" = "$sum" ] || { warn "sha256 不符 $path"; bad=1; }
  done < "$tmp/SO-MANIFEST.txt"
  [ "$bad" = 0 ] || die "校验失败，包损坏或被改动"

  newtag=${newtag:-${into%:*}-with-so:$(date +%Y%m%d)}
  log "覆盖到 $into 的副本 → $newtag"
  cid=$(docker create "$into")
  while read -r sz sum path; do
    case "${sz:-}" in ''|\#*) continue ;; esac
    docker cp "$tmp/$path" "$cid:$(dirname "$path")/" || die "docker cp 失败：$path"
  done < "$tmp/SO-MANIFEST.txt"

  docker commit --change 'LABEL dsv41.so-from="engine-bundle save-so"' "$cid" "$newtag" >/dev/null
  cleanup; cid=""

  log "完成：$newtag"
  echo "  启动时用 DSV41_IMAGE=$newtag 覆盖默认镜像名"
  warn "这是「换掉编译产物」得到的镜像，镜像 ID 必然与原始镜像不同；"
  warn "核对的是 .so 的 sha256（上面已逐个校验），不是 image digest。"
  ;;

load)
  [ $# -ge 1 ] || die "用法：load <xxx.tar.zst|xxx.tar.gz>"
  file=$1
  [ -f "$file" ] || die "找不到 $file"
  need_docker

  dir=$(cd "$(dirname "$file")" && pwd)
  base=$(basename "$file")
  sumfile="$file.sha256"

  if [ -f "$sumfile" ]; then
    log "校验 sha256（$(basename "$sumfile")）"
    ( cd "$dir" && sha256sum -c "$(basename "$sumfile")" ) || die "sha256 不匹配，包可能损坏"
  else
    warn "没有 .sha256 文件，跳过完整性校验"
  fi

  case "$file" in
    *.tar.zst) command -v zstd >/dev/null || die "没有 zstd，无法解压"; log "导入中（约 10~20 分钟）"; zstd -dc "$file" | docker load ;;
    *.tar.gz)  log "导入中"; gzip -dc "$file" | docker load ;;
    *.tar)     log "导入中"; docker load -i "$file" ;;
    *) die "不认识的扩展名：$file" ;;
  esac

  log "导入完成，请核对镜像 ID："
  for k in v2-dspark v1-legacy; do
    if docker image inspect "${IMAGE[$k]}" >/dev/null 2>&1; then
      id=$(docker image inspect "${IMAGE[$k]}" --format '{{.Id}}')
      [ "$id" = "${IMAGE_ID[$k]}" ] \
        && printf '  \033[32m✔\033[0m %-10s %s\n' "$k" "$id" \
        || printf '  \033[33m!\033[0m %-10s %s（记录值 %s）\n' "$k" "$id" "${IMAGE_ID[$k]}"
    fi
  done
  ;;

verify)
  [ $# -ge 1 ] || die "用法：verify <v2-dspark|v1-legacy>"
  name=$1
  [ -n "${IMAGE[$name]:-}" ] || die "未知引擎 $name"
  need_docker
  img=${IMAGE[$name]}
  docker image inspect "$img" >/dev/null 2>&1 || die "本机没有镜像 $img"
  id=$(docker image inspect "$img" --format '{{.Id}}')
  echo "镜像    $img"
  echo "当前 ID $id"
  echo "记录 ID ${IMAGE_ID[$name]}"
  if [ "$id" = "${IMAGE_ID[$name]}" ]; then
    printf '\033[32m结果    一致 ✔\033[0m\n'
  else
    printf '\033[33m结果    不一致 ✘ —— 这不是测出那份基准的镜像\033[0m\n'
    exit 1
  fi
  ;;

*) usage ;;

esac
