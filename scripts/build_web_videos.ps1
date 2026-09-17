param(
  [string]$FfmpegPath = 'D:\Anaconda\envs\delearning\Library\bin\ffmpeg.exe',
  [string]$FfprobePath = 'D:\Anaconda\envs\delearning\Library\bin\ffprobe.exe',
  [string]$OnlyOutputFolder = ''
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $FfmpegPath)) {
  throw "FFmpeg was not found at: $FfmpegPath"
}
if (-not (Test-Path -LiteralPath $FfprobePath)) {
  throw "FFprobe was not found at: $FfprobePath"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $repoRoot 'src\assets\videos'
$outputRoot = Join-Path $repoRoot 'src\assets\videos_web'

$jobs = @(
  @{ SourceFolder = 'realworld_demo_2x'; OutputFolder = 'realworld_demo_2x'; Mode = 'encode'; VideoFilter = 'scale=498:360:flags=lanczos' },
  @{ SourceFolder = 'simulation_demo'; OutputFolder = 'simulation_demo_compressed'; Mode = 'encode'; VideoFilter = '' },
  @{ SourceFolder = 'ood_real_videos_2x'; OutputFolder = 'ood_real_videos_2x'; Mode = 'encode'; VideoFilter = 'scale=498:360:flags=lanczos' },
  @{ SourceFolder = 'ood_sim_videos'; OutputFolder = 'ood_sim_videos_compressed'; Mode = 'encode'; VideoFilter = '' }
)
if ($OnlyOutputFolder) {
  $jobs = @($jobs | Where-Object OutputFolder -eq $OnlyOutputFolder)
  if ($jobs.Count -eq 0) {
    throw "Unknown output folder: $OnlyOutputFolder"
  }
}

$report = @()

foreach ($job in $jobs) {
  $sourceFolder = Join-Path $sourceRoot $job.SourceFolder
  $outputFolder = Join-Path $outputRoot $job.OutputFolder
  if (-not (Test-Path -LiteralPath $sourceFolder)) {
    $sourceFolder = Join-Path $outputRoot $job.SourceFolder
  }
  if (-not (Test-Path -LiteralPath $sourceFolder)) {
    throw "Source video folder was not found: $($job.SourceFolder)"
  }
  if ($sourceFolder -eq $outputFolder) {
    throw "Source and output folders must differ when creating encoded derivatives: $sourceFolder"
  }
  New-Item -ItemType Directory -Force -Path $outputFolder | Out-Null

  $inputs = Get-ChildItem -LiteralPath $sourceFolder -Filter '*.mp4' | Sort-Object Name
  foreach ($input in $inputs) {
    $output = Join-Path $outputFolder $input.Name
    $sourceHashBefore = (Get-FileHash -LiteralPath $input.FullName -Algorithm SHA256).Hash
    $reuseExisting = $false
    if (Test-Path -LiteralPath $output) {
      $existingHash = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
      if ($job.Mode -eq 'copy' -and $existingHash -eq $sourceHashBefore) {
        $reuseExisting = $true
      } else {
        throw "Refusing to overwrite existing derivative: $output"
      }
    }

    if (-not $reuseExisting) {
      $conversionExitCode = 0
      switch ($job.Mode) {
        'copy' {
          Copy-Item -LiteralPath $input.FullName -Destination $output
        }
        'faststart' {
          & $FfmpegPath -hide_banner -loglevel error -n `
            -i $input.FullName -map '0:v:0' -an -c:v copy `
            -movflags '+faststart' $output
          $conversionExitCode = $LASTEXITCODE
        }
        'encode' {
          $encodeArgs = @(
            '-hide_banner', '-loglevel', 'error', '-n',
            '-i', $input.FullName, '-map', '0:v:0', '-an'
          )
          if ($job.VideoFilter) {
            $encodeArgs += @('-vf', $job.VideoFilter)
          }
          $encodeArgs += @(
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '25',
            '-maxrate', '1000k', '-bufsize', '2000k',
            '-pix_fmt', 'yuv420p', '-profile:v', 'high',
            '-force_key_frames', 'expr:gte(t,n_forced*2)',
            '-movflags', '+faststart', $output
          )
          & $FfmpegPath @encodeArgs
          $conversionExitCode = $LASTEXITCODE
        }
        default {
          throw "Unknown conversion mode: $($job.Mode)"
        }
      }

      if ($conversionExitCode -ne 0 -or -not (Test-Path -LiteralPath $output)) {
        throw "Video conversion failed: $($input.FullName)"
      }
    }

    $sourceHashAfter = (Get-FileHash -LiteralPath $input.FullName -Algorithm SHA256).Hash
    if ($sourceHashBefore -ne $sourceHashAfter) {
      throw "Source video changed unexpectedly: $($input.FullName)"
    }

    $probeJson = & $FfprobePath -v error -select_streams 'v:0' `
      -show_entries 'stream=codec_name,width,height,avg_frame_rate,pix_fmt,nb_frames' `
      -show_entries 'format=duration,size' -of json $output
    if ($LASTEXITCODE -ne 0) {
      throw "FFprobe failed: $output"
    }
    $probe = $probeJson | ConvertFrom-Json
    $stream = $probe.streams[0]

    $report += [pscustomobject]@{
      source_folder = $job.SourceFolder
      folder = $job.OutputFolder
      file = $input.Name
      mode = $job.Mode
      source_sha256 = $sourceHashBefore
      output_sha256 = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash
      source_bytes = $input.Length
      output_bytes = (Get-Item -LiteralPath $output).Length
      codec = $stream.codec_name
      width = $stream.width
      height = $stream.height
      pixel_format = $stream.pix_fmt
      frame_rate = $stream.avg_frame_rate
      frames = $stream.nb_frames
      duration = [double]$probe.format.duration
    }
  }
}

$reportName = if ($OnlyOutputFolder) { "$($OnlyOutputFolder)_conversion_report.json" } else { 'conversion_report.json' }
$reportPath = Join-Path $outputRoot $reportName
$report | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $reportPath -Encoding utf8NoBOM
Write-Output "Created $($report.Count) web video derivatives."
Write-Output "Report: $reportPath"
