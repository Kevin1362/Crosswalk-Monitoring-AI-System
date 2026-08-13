param([string]$Video=".\videos\real_crosswalk.mp4")
Set-Location $PSScriptRoot\..
.\.venv\Scripts\Activate.ps1
python enhanced_video_runner.py --source $Video --display --loop
