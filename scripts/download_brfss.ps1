# Download BRFSS annual LLCP files (SAS transport, zipped) from CDC into data/raw/brfss/ and unzip them.
# Safe to re-run: years whose .XPT file already exists are skipped.
#
#   powershell -ExecutionPolicy Bypass -File scripts\download_brfss.ps1                  # 2011-2024
#   powershell -ExecutionPolicy Bypass -File scripts\download_brfss.ps1 -Years 2015,2016  # chosen years (or -Years 2017-2020)
#
# Each year is ~70-130 MB zipped and roughly 0.5-1 GB unzipped.

param([string]$Years = "2011-2024")

# Accept "2017,2018,2019,2020", "2017 2018", or a range like "2017-2020"
# (powershell -File passes the whole argument as one string).
$list = @()
foreach ($part in ($Years -split '[,\s]+' | Where-Object { $_ })) {
  if ($part -match '^(\d{4})-(\d{4})$') { $list += [int]$matches[1]..[int]$matches[2] }
  else { $list += [int]$part }
}

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$outDir = Join-Path $PSScriptRoot "..\data\raw\brfss"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# File names checked on the CDC annual-data pages (2011-2014 use upper-case .ZIP).
function Get-Url($year) {
  $ext = if ($year -le 2014) { "ZIP" } else { "zip" }
  return "https://www.cdc.gov/brfss/annual_data/$year/files/LLCP${year}XPT.$ext"
}

$failed = @()
foreach ($y in $list) {
  $existing = Get-ChildItem -Path $outDir -Filter "LLCP$y*.XPT*" -ErrorAction SilentlyContinue
  if ($existing) { Write-Host "SKIP $y (already have $($existing[0].Name))"; continue }
  $zip = Join-Path $outDir "LLCP${y}XPT.zip"
  $url = Get-Url $y
  $ok = $false
  for ($attempt = 1; $attempt -le 3 -and -not $ok; $attempt++) {
    try {
      Write-Host "GET  $y  $url"
      # cdc.gov rejects (403) requests that do not look like a browser, so send browser-like headers.
      $headers = @{
        "Accept" = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        "Accept-Language" = "en-US,en;q=0.9"
        "Referer" = "https://www.cdc.gov/brfss/annual_data/annual_$y.html"
      }
      $ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
      Invoke-WebRequest -Uri $url -OutFile $zip -UserAgent $ua -Headers $headers -TimeoutSec 900
      Expand-Archive -Path $zip -DestinationPath $outDir -Force
      Remove-Item $zip
      $ok = $true
    } catch {
      Write-Host "  attempt $attempt failed: $($_.Exception.Message)" -ForegroundColor Yellow
      Remove-Item $zip -ErrorAction SilentlyContinue
      Start-Sleep -Seconds (5 * $attempt)
    }
  }
  if ($ok) { Write-Host "OK   $y" } else { $failed += $y; Write-Host "FAIL $y" -ForegroundColor Red }
}
Write-Host ""
Get-ChildItem $outDir | Select-Object Name, @{n="MB";e={[math]::Round($_.Length/1MB)}} | Format-Table -AutoSize
if ($failed) { Write-Host "Failed years (re-run to retry): $($failed -join ', ')" -ForegroundColor Yellow }
