$ErrorActionPreference = 'SilentlyContinue'
$log = "g:\GitHub\DiT\dwm_monitor_log.csv"
$intervalSec = 60

# header
if (-not (Test-Path $log)) {
    "Timestamp, DwmPID, DwmWorkingSet_MB, ActiveDisplays, PrimaryDisplayGPU, GameViewerEnabled" | Out-File -FilePath $log -Encoding utf8
}

Write-Host "Monitoring dwm.exe every $intervalSec s -> $log  (Ctrl+C to stop)"

function Get-PrimaryGpuName {
    # crude: read which adapter holds the primary path via EnumDisplayDevices
    Add-Type @'
    using System; using System.Runtime.InteropServices;
    public class P {
      [DllImport("user32.dll")] public static extern bool EnumDisplayDevices(string d, uint n, ref DD dd, uint f);
      [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
      public struct DD { public int cb; public int sf; [MarshalAs(UnmanagedType.ByValTStr,SizeConst=32)] public string dn; [MarshalAs(UnmanagedType.ByValTStr,SizeConst=128)] public string ds; [MarshalAs(UnmanagedType.ByValTStr,SizeConst=128)] public string di; [MarshalAs(UnmanagedType.ByValTStr,SizeConst=128)] public string dk; }
      public static string Go() {
        DD dd = new DD(); dd.cb = Marshal.SizeOf(dd); uint i=0; string prim="";
        while (EnumDisplayDevices(null,i,ref dd,1)) {
          if ((dd.sf & 4)!=0 && (dd.sf & 1)!=0) { prim = dd.ds; }
          dd = new DD(); dd.cb = Marshal.SizeOf(dd); i++;
        }
        return prim;
      }
    }
'@
    try { return [P]::Go() } catch { return "err" }
}

while ($true) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $dwm = Get-Process dwm -ErrorAction SilentlyContinue
    $ws = if ($dwm) { [math]::Round($dwm.WorkingSet/1MB,1) } else { 0 }
    $pidv = if ($dwm) { $dwm.Id } else { 0 }

    $disps = Get-CimInstance Win32_VideoController | Where-Object { $_.CurrentHorizontalResolution -gt 0 }
    $active = $disps.Count
    $prim = Get-PrimaryGpuName

    $gv = Get-PnpDevice -FriendlyName '*GameViewer*' -ErrorAction SilentlyContinue
    $gvState = if ($gv) { $gv.Status } else { "absent" }

    $line = "{0}, {1}, {2}, {3}, {4}, {5}" -f $ts, $pidv, $ws, $active, $prim, $gvState
    $line | Out-File -FilePath $log -Append -Encoding utf8
    Write-Host $line
    Start-Sleep -Seconds $intervalSec
}
