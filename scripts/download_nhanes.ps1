# Download NHANES 2005-2018 files (28 components x 7 cycles = 196 files) into data/raw/.
# Safe to re-run: files that are already present and valid are skipped.
# Run from the repository root:   powershell -ExecutionPolicy Bypass -File scripts\download_nhanes.ps1

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster

$rawDir = Join-Path $PSScriptRoot "..\data\raw"
New-Item -ItemType Directory -Force -Path $rawDir | Out-Null

$cycles = [ordered]@{ D = 2005; E = 2007; F = 2009; G = 2011; H = 2013; I = 2015; J = 2017 }
$components = @(
  "DEMO","DIQ","BMX","BPX","BPQ","SMQ","ALQ","PAQ","MCQ","KIQ_U","HIQ","HUQ","DPQ","SLQ",
  "WHQ","CDQ","OCQ","DBQ","HSQ","PFQ","DUQ","RHQ","TCHOL","HDL","TRIGLY","BIOPRO","CBC","ALB_CR"
)
$magic = "HEADER RECORD*******LIBRARY HEADER RECORD"

function Test-Xpt($path) {
  if (-not (Test-Path $path)) { return $false }
  if ((Get-Item $path).Length -lt 80) { return $false }
  $fs = [System.IO.File]::OpenRead($path)
  try { $buf = New-Object byte[] 41; [void]$fs.Read($buf, 0, 41) } finally { $fs.Close() }
  return ([System.Text.Encoding]::ASCII.GetString($buf) -eq $magic)
}

$ok = 0; $skipped = 0; $failed = @()
foreach ($s in $cycles.Keys) {
  $year = $cycles[$s]
  foreach ($c in $components) {
    $name = "${c}_${s}.xpt"
    $target = Join-Path $rawDir $name
    if (Test-Xpt $target) { $skipped++; continue }
    $url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/$year/DataFiles/$name"
    $done = $false
    for ($attempt = 1; $attempt -le 3 -and -not $done; $attempt++) {
      try {
        Invoke-WebRequest -Uri $url -OutFile $target -UserAgent "Mozilla/5.0 (educational research script)" -TimeoutSec 120
        if (Test-Xpt $target) { $done = $true } else { Remove-Item $target -ErrorAction SilentlyContinue; Start-Sleep -Seconds (2 * $attempt) }
      } catch {
        Remove-Item $target -ErrorAction SilentlyContinue
        Start-Sleep -Seconds (2 * $attempt)
      }
    }
    if ($done) { $ok++; Write-Host "OK   $name" } else { $failed += $name; Write-Host "FAIL $name" -ForegroundColor Red }
  }
}

Write-Host ""
Write-Host "Downloaded: $ok   Already present: $skipped   Failed: $($failed.Count)"
if ($failed.Count -gt 0) { Write-Host "Re-run the script to retry: $($failed -join ', ')" -ForegroundColor Yellow }
