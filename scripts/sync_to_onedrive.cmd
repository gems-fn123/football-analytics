@echo off
rem Periodic backup of the geo-tracker working copy into corporate OneDrive.
rem
rem The working copy lives OUTSIDE OneDrive on purpose: live sync churns on
rem every build artefact and slows the repo. This mirror runs on a schedule
rem (Task Scheduler task "geo-tracker-onedrive-sync") instead.
rem
rem The destination is the ORIGINAL side-job folder, kept as the OneDrive
rem backup by request. It is a mirror: never edit it directly, /MIR will
rem overwrite it (deletions included). Excluded items (.env, caches) are
rem invisible to robocopy on both sides, so the .env already there survives.
rem
rem Exclusions:
rem   .env            credentials stay out of the copy traffic
rem   __pycache__ etc regenerated junk
rem   big weights     re-downloadable via scripts/download_weights.sh; too large
rem                   for corporate OneDrive (SV_kp.pth alone is 265 MB)

set SRC=C:\Users\FAAU\geo-tracker
set DST=D:\Users\FAAU\OneDrive - starenergy.co.id\side-job\football-analytics

robocopy "%SRC%" "%DST%" /MIR /COPY:DAT /R:2 /W:5 ^
  /XD __pycache__ .pytest_cache .ruff_cache ^
  /XF .env SV_kp.pth SV_lines.pth rfdetr_football.pt ^
  /NFL /NDL /NP /LOG+:"%TEMP%\geo-tracker-sync.log"

rem robocopy exit codes 0-7 mean success; 8+ are real failures.
if %ERRORLEVEL% GEQ 8 exit /b %ERRORLEVEL%
exit /b 0
