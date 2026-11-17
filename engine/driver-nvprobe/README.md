# 驱动重建 + NVPROBE 死机探针（2026-11-17）

> 回答：怎么从 cmpunlocker 构建管线重建内核驱动，以及探针补丁怎么加。
> 这是 11-17 死机根因调查的产物：死机形态 = 零内核输出瞬死（疑似 CPU 卡死在 MMIO
> 读取上），被动观测抓不到——**驱动内 printk 的"最后一行"就是楔死现场的函数级坐标**。

## 一、构建管线（cmpunlocker 官方路线）

服务器上唯一的驱动重建入口：

```bash
sudo CMPUNLOCKER_DRIVER_VERSION=610.43.03 NO_COLOR=1 bash /var/tmp/cmpunlocker/driver/build.sh
```

管线做的事（`/var/tmp/cmpunlocker/driver/build.sh`）：
1. 取 `open-gpu-kernel-modules-610.43.03.tar.gz`（NVIDIA 官方 tarball，已缓存在 `.build/`）
2. 按 `PATCH_ORDER` 依次 `patch -p1` 打 15 个解锁补丁（sec2-postbl / booter-verify /
   bar1-resize-unlock / cmp-sku-mask / ecc-\* / name-string / …）
3. python 重写 `kernel_gsp.c` 的 CFG1/LMR/FB（8gb profile → 64GB geometry）
4. `make -j$(nproc) modules` + 安装到 `/lib/modules/$(uname -r)/updates/cmpunlocker/`
   + `depmod` + initramfs

**坑**：每个补丁必须在 `/var/tmp/cmpunlocker/common/constants.yaml` 声明
（`<名字>: {patch: ..., registers: {}}`），否则构建前置校验直接 FAIL。

**装完即生效吗**：安装只是替换磁盘上的 `.ko` 文件，**重启后才加载新驱动**
（运行中模块不受影响，挂了之后下次开机自动带上探针）。

回滚：`cp nvidia.ko.bak-pre-nvprobe nvidia.ko`（同目录）+ 重启。

## 二、NVPROBE 探针（`nvprobe-trace.patch`）

加在 PATCH_ORDER 末尾，改动仅 `kernel-open/nvidia/nv.c` 一个文件（15 处点）：

| 探针 | 覆盖什么 |
|---|---|
| `OPEN begin/end` / `OPEN ctl end` / `OPEN defer-*` | nvidia_open 全路径（含延迟打开分支） |
| `OPEN-DEV begin/end` | nv_open_device（设备拉起） |
| `OPEN-DEFER run` + `llock wait-begin/acquired` | 后台打开线程 + ldata_lock 等待 |
| `CLOSE begin/end` / `CLOSE ctl end` / `CLOSE-CB begin` | nvidia_close 全路径 |
| `CLOSE-DEV teardown begin/end` | nv_close_device（设备拆卸） |
| `CLOSE-CB llock wait-begin/acquired` | close 侧 ldata_lock 等待（#968 close 一侧） |
| `FIND-MINOR/FIND-GPUID/FIND-UUID llock wait-begin/acquired` | find_\* 锁等待 |
| `RMCMD begin cmd=0x…`（5s 限频）/ `RMCMD slow took=…ms`（>100ms） | RM 控制入口/慢调用 |

**频控设计**：锁等待用 `down_trylock` 探测——无争用零输出，有争用必留痕
（即使之后永久卡死，wait-begin 行也已经在日志里）；RM 入口 5s 限频；慢调用 >100ms 才记。

**判读方法**：
```
死机后 grep NVPROBE /var/log/dsv41-blackbox/kmsg.log（或 Windows 侧 netconsole-*.log）
看最后几行：
- 最后一条 wait-begin 无 acquired → 卡死在这把锁上（谁等的一目了然）
- 最后一条 OPEN-DEV begin 无 end → 卡死在设备拉起（MMIO 楔死最可能的位置）
- 最后一条 RMCMD begin cmd=0x… → 卡死在该 RM 控制调用里
- CLOSE-CB begin 后无 teardown → 卡死在 close 侧清理
```

## 三、验证清单（重启后必做）

1. `nvidia-smi` 显示 5 卡、显存 ~63.4GB/卡（unlock geometry 没丢）
2. `dmesg | grep NVPROBE | head` 有行（nvidia-smi 一次触发 OPEN/CLOSE 对）
3. 引擎容器起来、`Expert replacements` 正常（dmesg 无报错）
4. `cat /lib/modules/$(uname -r)/updates/cmpunlocker/card_profile` = 8gb

## 四、本次构建参数

- VERSION=610.43.03（与安装版一致）、profile=8gb（gpu_inventory 五卡 65536MB 与之一致）
- 旧驱动备份：`/lib/modules/$(uname -r)/updates/cmpunlocker/nvidia.ko.bak-pre-nvprobe`
- 构建日志：服务器 `/tmp/nvprobe-build.log`
