$ErrorActionPreference = 'SilentlyContinue'

Write-Host "=== Primary display (DWM binds to this GPU) ==="
# The display marked as primary in the current display config
$path = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Explorer\Multimon'
if (Test-Path $path) {
    $v = Get-ItemProperty -Path $path -Name 'PrimaryDisplay' -ErrorAction SilentlyContinue
    if ($v) { Write-Host ("PrimaryDisplay (device id) = {0}" -f $v.PrimaryDisplay) }
}

Write-Host ""
Write-Host "=== Per-display info via EnumDisplayDevices (primary flag) ==="
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class Disp {
  [DllImport("user32.dll")] public static extern bool EnumDisplayDevices(string lpDevice, uint iDevNum, ref DISPLAY_DEVICE dd, uint dwFlags);
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Auto)]
  public struct DISPLAY_DEVICE {
    public int cb; public int StateFlags;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=32)] public string DeviceName;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string DeviceString;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string DeviceID;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=128)] public string DeviceKey;
  }
  public static void Run() {
    DISPLAY_DEVICE dd = new DISPLAY_DEVICE(); dd.cb = Marshal.SizeOf(dd);
    uint i=0;
    while (EnumDisplayDevices(null, i, ref dd, 1)) {
      bool primary = (dd.StateFlags & 0x00000004) != 0;  // DISPLAY_DEVICE_PRIMARY
      bool attached = (dd.StateFlags & 0x00000001) != 0; // ATTACHED_TO_DESKTOP
      Console.WriteLine("Dev{0}: {1} | primary={2} attached={3} | {4}", i, dd.DeviceString, primary, attached, dd.DeviceID);
      dd = new DISPLAY_DEVICE(); dd.cb = Marshal.SizeOf(dd);
      i++;
    }
  }
}
'@
[Disp]::Run()

Write-Host ""
Write-Host "=== Display topology (which GPU each screen maps to) ==="
Get-CimInstance Win32_VideoController | ForEach-Object {
    if ($_.CurrentHorizontalResolution -gt 0) {
        Write-Host ("{0}: {1}x{2} @ {3}Hz" -f $_.Name, $_.CurrentHorizontalResolution, $_.CurrentVerticalResolution, $_.CurrentRefreshRate)
    }
}
