# dsv41 看门狗（Windows 侧）——检测引擎断连 → 智能插座断电重启
# 用法: powershell -File dsv41_watchdog.ps1
# 前置: 服务器 BIOS 已设 Restore-on-AC-Power-On（断电复电自动开机，已验证）
# 插座: 待定品牌后把 <PLUG_API_ON>/<PLUG_API_OFF> 换成实际控制接口

$EngineIP = "192.168.3.44"
$EnginePort = 9010
$FailThreshold = 6          # 连续失败次数（6 x 15s = 90s 断连即判定死机）
$IntervalSec = 15
$OffSeconds = 12            # 断电时长

$failCount = 0
$cycled = $false

while ($true) {
    $ok = $false
    try {
        $tcp = New-Object Net.Sockets.TcpClient
        $r = $tcp.BeginConnect($EngineIP, $EnginePort, $null, $null)
        $ok = $r.AsyncWaitHandle.WaitOne(4000) -and $tcp.Connected
        $tcp.Close()
    } catch { $ok = $false }

    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($ok) {
        if ($failCount -gt 0 -or $cycled) { Write-Host "[$ts] engine OK (failCount was $failCount)" }
        $failCount = 0
        $cycled = $false
    } else {
        $failCount++
        Write-Host "[$ts] engine DOWN ($failCount/$FailThreshold)"
        if ($failCount -ge $FailThreshold -and -not $cycled) {
            Write-Host "[$ts] *** TRIGGER: power cycle via smart plug ***"
            # ---- 插座断电（换成实际品牌 API）----
            # 小米: 米家开放平台 API / 涂鸦: cloud API / eWeLink: REST
            Invoke-RestMethod -Uri "<PLUG_API_OFF>" -Method Get -TimeoutSec 10
            Start-Sleep -Seconds $OffSeconds
            Invoke-RestMethod -Uri "<PLUG_API_ON>" -Method Get -TimeoutSec 10
            Write-Host "[$ts] power restored, waiting for BIOS auto-boot..."
            $cycled = $true
            $failCount = 0
            # 等机器起来（最长 5 分钟）
            $bootWait = 0
            while ($bootWait -lt 300) {
                Start-Sleep -Seconds 20; $bootWait += 20
                try {
                    $tcp = New-Object Net.Sockets.TcpClient
                    $r = $tcp.BeginConnect($EngineIP, $EnginePort, $null, $null)
                    if ($r.AsyncWaitHandle.WaitOne(4000) -and $tcp.Connected) { $tcp.Close(); break }
                    $tcp.Close()
                } catch { }
            }
            Write-Host "[$ts] engine back online after $($bootWait)s"
        }
    }
    Start-Sleep -Seconds $IntervalSec
}
