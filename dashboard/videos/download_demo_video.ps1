
$ErrorActionPreference = "Stop"

$videoFolder = Join-Path $PSScriptRoot "videos"
$videoPath = Join-Path $videoFolder "guidekaro_demo.mp4"
$videoUrl = "https://github.com/intel-iot-devkit/sample-videos/raw/refs/heads/master/person-bicycle-car-detection.mp4"

New-Item -ItemType Directory -Force -Path $videoFolder | Out-Null

Write-Host "Downloading the GuideKaro person/vehicle demonstration video..."
Invoke-WebRequest -Uri $videoUrl -OutFile $videoPath

Write-Host ""
Write-Host "Downloaded successfully:"
Write-Host $videoPath
Write-Host ""
Write-Host "Run it with:"
Write-Host "python presentation_video_runner.py --source videos/guidekaro_demo.mp4 --presentation-mode --loop"
