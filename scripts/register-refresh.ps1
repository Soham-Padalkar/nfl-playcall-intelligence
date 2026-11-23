$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $project '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $python)) { throw 'Create the local virtual environment first.' }
$action = New-ScheduledTaskAction -Execute $python -Argument '-m playcall.refresh --train --freeze-updates' -WorkingDirectory $project
$trigger = New-ScheduledTaskTrigger -Daily -At '06:00'
$kickoff = (Get-Content -Raw -LiteralPath (Join-Path $project 'playcall/context.json') | ConvertFrom-Json).game.kickoff_utc
$trigger.EndBoundary = [DateTimeOffset]::Parse($kickoff).AddDays(2).LocalDateTime.ToString('yyyy-MM-ddTHH:mm:ss')
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'NFL Playcall Intelligence - refresh' -Action $action -Trigger $trigger -Settings $settings -Description 'Refresh local nflverse data and check the pregame training inputs.' -Force
