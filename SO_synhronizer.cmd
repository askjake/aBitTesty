@echo off
SETLOCAL ENABLEEXTENSIONS

REM ─── Configuration ─────────────────────────────────
set "REPO_URL=https://github.com/askjake/aBitTesty.git"
set "TARGET_DIR=C:\DPUnified\UserDefinedTasks\aBitTesty"

REM ─── Make sure Git is on the PATH ───────────────────
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
  echo ERROR: git not found in PATH. Please install Git or add it to your PATH.
  exit /b 1
)

REM ─── Clone if missing ────────────────────────────────
if not exist "%TARGET_DIR%\.git" (
  echo Cloning aBitTesty into "%TARGET_DIR%"…
  git clone "%REPO_URL%" "%TARGET_DIR%"
  if %ERRORLEVEL% neq 0 (
    echo ERROR: git clone failed.
    exit /b 1
  )
  echo Clone complete.
) else (
  REM ─── Otherwise pull latest on main ───────────────
  echo Updating existing aBitTesty in "%TARGET_DIR%"…
  pushd "%TARGET_DIR%" >nul || exit /b 1
    git fetch --all --prune || exit /b 1
    git reset --hard origin/main || exit /b 1
  popd >nul
  echo Update complete.
)

echo.
echo ✔ aBitTesty scripts are now up-to-date in "%TARGET_DIR%".
ENDLOCAL
exit /b 0
