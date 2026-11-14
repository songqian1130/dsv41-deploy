#!/usr/bin/env bash
# verify-sources.sh — 逐条复验 engine/LINKS.md 里记录的外部地址
#
#   ./tools/verify-sources.sh            # 全部检查
#   ./tools/verify-sources.sh --quiet    # 只打印汇总
#
# 状态：
#   OK      HTTP 200，且 sha256 与仓库归档一致
#   DRIFT   拉到了，但 sha256 变了 —— 上游改过内容，必须人工判断
#   GONE    HTTP 404/410 —— 地址失效
#   NET     连不上（000/超时/代理拦截）—— 不是地址本身的问题
#   SKIP    需要 docker 才能验，本脚本不碰
#
# 退出码：0 = 没有 DRIFT/GONE；1 = 有（NET 不计入失败）。
set -uo pipefail   # 故意不用 -e：单条失败也要跑完全部

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1

TMP=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP" 2>/dev/null || true' EXIT

fail=0; ok=0; drift=0; gone=0; net=0

say()  { [[ $QUIET -eq 1 ]] || printf '%s\n' "$*"; }
line() { say "-------------------------------------------------------------------------------"; }

# ---------- 代理 ----------
# 这台机器上 WorkBuddy 会注入 http_proxy=127.0.0.1:9455（沙箱代理，只放行白名单域名，
# 到不了 Docker Hub / HuggingFace）。用户自己的 Clash / FlClash 是 7890，但**要手动启动**。
# 这里自动探测常见本地代理端口，探测到就用它；可用 DSV41_PROXY 强制指定。
#   DSV41_PROXY=none     → 完全直连（--noproxy '*'）
#   DSV41_PROXY=http://127.0.0.1:7890  → 指定
PROXY="${DSV41_PROXY:-}"
proxy_opt=()
if [[ "$PROXY" == "none" ]]; then
  proxy_opt=(--noproxy '*')
elif [[ -n "$PROXY" ]]; then
  proxy_opt=(--proxy "$PROXY")
else
  for p in 7890 7897 1080 10809 7891 8889; do
    if (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null; then
      exec 3<&- 3>&- 2>/dev/null || true
      PROXY="http://127.0.0.1:$p"
      proxy_opt=(--proxy "$PROXY")
      break
    fi
  done
fi
# 环境变量里的 9455 是沙箱代理，会干扰 curl；显式清掉，改由 proxy_opt 决定走向
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy 2>/dev/null || true

if [[ ${#proxy_opt[@]} -gt 0 ]]; then
  say ""; say "代理：${proxy_opt[*]}"
else
  say ""; say "代理：未探测到本地代理（7890/7897/1080/10809/7891/8889 均未监听）→ 直连"
  say "      Docker Hub / HuggingFace 会显示 NET —— 那是网络问题，不是链接失效"
fi

# check_sha <名字> <URL> <期望 sha256 前 12 位>
check_sha() {
  local name=$1 url=$2 want=${3:0:12} out code got
  out="$TMP/$(printf '%s' "$name" | tr -c 'A-Za-z0-9.-' '_')"
  code=$(curl -sSL "${proxy_opt[@]}" -o "$out" -w '%{http_code}' --max-time 45 "$url" 2>/dev/null) || code=000
  case "$code" in
    200|206)
      # 从 stdin 读：直接传文件名时，GNU coreutils 遇到含反斜杠的 Windows 路径
      # 会在行首加一个转义 `\`，把哈希挤偏一位
      got=$(sha256sum < "$out" | cut -c1-12)
      if [[ "$got" == "$want" ]]; then
        say "$(printf '%-6s' OK) $name  ${got} == 归档"; ok=$((ok+1))
      else
        say "$(printf '%-6s' DRIFT) $name  上游 ${got} != 归档 ${want}"; drift=$((drift+1)); fail=1
      fi ;;
    000|"")
      say "$(printf '%-6s' NET) $name  连不上（代理/DNS），未判定"; net=$((net+1)) ;;
    *)
      say "$(printf '%-6s' GONE) $name  HTTP $code"; gone=$((gone+1)); fail=1 ;;
  esac
}

# check_http <名字> <URL>  —— 只要可达即可（仓库页 / 索引页）
check_http() {
  local name=$1 url=$2 code
  code=$(curl -sSL "${proxy_opt[@]}" -o /dev/null -w '%{http_code}' --max-time 30 "$url" 2>/dev/null) || code=000
  case "$code" in
    200)
      say "$(printf '%-6s' OK) $name  可达（200）"; ok=$((ok+1)) ;;
    000|"")
      say "$(printf '%-6s' NET) $name  连不上（代理/DNS），未判定"; net=$((net+1)) ;;
    *)
      say "$(printf '%-6s' GONE) $name  HTTP $code"; gone=$((gone+1)); fail=1 ;;
  esac
}

skip() { say "$(printf '%-6s' SKIP) $1  $2"; }

AIU=https://raw.githubusercontent.com/aiultrayang/deepseek-v41-flash-cmp170hx/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a/pp6-ssd
LFL=https://raw.githubusercontent.com/LocalFrameLab/deepseek-v41-cmp170hx-x99/main/patches

say ""
say "外部依赖复验 —— 依据 engine/LINKS.md"
line
say "[1] 补丁与配方（必须 sha256 一致）"
check_sha "0001-prime-pp-communicators" "$AIU/patches/0001-prime-pp-communicators-before-model-load.patch" 16d8b7554700
check_sha "engram-ssd-port（上游名无 0002 前缀）" "$AIU/patches/engram-ssd-port.patch" 5ec439e0d300
check_sha "Containerfile（v1-legacy 上游原件）" "$AIU/build/Containerfile" 5af7be182795
check_sha "x99-stable.patch（A/B 的 74 行原件）" "$LFL/x99-stable.patch" 16f2a4606dae

line
say "[2] 钉住的 commit 是否还取得到（走 api.github.com，github.com 的 HTML 直连不通）"
check_http "buliaoyin/vllm-backport @ c3d3104de" \
  https://api.github.com/repos/buliaoyin/vllm-backport/commits/c3d3104de0163c389d26d9fcba0180f55ad7964d
check_http "344303947/dsv41-flash-pp5-170hx @ d63af5a47" \
  https://api.github.com/repos/344303947/dsv41-flash-pp5-170hx/commits/d63af5a472dc76b12d7a73d50a5af142844c15d1
check_http "aiultrayang/...-cmp170hx @ 29b03af5e" \
  https://api.github.com/repos/aiultrayang/deepseek-v41-flash-cmp170hx/commits/29b03af5eedfac9af12dfa14e8cfa7e2eafc852a
check_http "LocalFrameLab/...-x99（按 main）" \
  https://api.github.com/repos/LocalFrameLab/deepseek-v41-cmp170hx-x99
check_http "modelscope.cn/.../DeepSeek-V4.1-Flash-1017" https://modelscope.cn/api/v1/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
check_http "pytorch cu130 index" https://download.pytorch.org/whl/cu130

line
say "[3] 需要 docker 的（本脚本不验）"
skip "18gogogo/170hx4-256r-dsv41-1017-flash:sm80" "docker manifest inspect ..."
skip "local/deepseek-v41-official-pp5:..."          "无 registry，只能 engine-bundle.sh load"
skip "lazymio/vllm-backport:v0.13.0-sm80"           "docker manifest inspect ..."
skip "nvidia/cuda:13.0.3-devel-ubuntu24.04"         "tag 名为推断"

line
printf '%s\n' "汇总：OK ${ok}   DRIFT ${drift}   GONE ${gone}   NET ${net}   （SKIP 不计）"
if [[ $fail -eq 0 ]]; then
  printf '%s\n' "结论：所有可验证的地址都还指向当初那份内容。"
else
  printf '%s\n' "结论：有地址失配，别直接用 —— 先看 engine/LINKS.md 第 6 节再决定。"
fi
[[ $QUIET -eq 1 ]] || printf '\n'
exit $fail
