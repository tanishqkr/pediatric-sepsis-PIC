@echo off
setlocal

REM =====================================================
REM Pediatric Sepsis PIC
REM Phase 7 Federated Learning + Phase 9 Federated SHAP
REM =====================================================

REM Always run from the directory where this BAT file is located.
REM This BAT file should be placed at:
REM D:\pediatric-sepsis-PIC\sepsis_ml\
cd /d "%~dp0"

set "LOG=run_phase7_phase9.txt"
set "ERRLOG=error_phase7_phase9.txt"

echo =============================================== > "%LOG%"
echo Phase 7 + Phase 9 Pipeline Started %DATE% %TIME% >> "%LOG%"
echo =============================================== >> "%LOG%"

echo =============================================== > "%ERRLOG%"
echo Failed Commands >> "%ERRLOG%"
echo =============================================== >> "%ERRLOG%"


REM =====================================================
REM Step 1: Phase 7 Federated Learning
REM =====================================================

call :Run "cd /d "%~dp0fl" && python phase7_federated_learning.py"

if errorlevel 1 goto :FAILED_PIPELINE


REM =====================================================
REM Step 2: Verify Required Phase 7 Checkpoints
REM =====================================================

echo.
echo ==================================================
echo Verifying Phase 7 checkpoints...
echo ==================================================

if not exist "%~dp0fl\models\fedavg_global_final.pt" (
    echo ERROR: fedavg_global_final.pt not found.
    echo [FAILED] Missing fedavg_global_final.pt >> "%ERRLOG%"
    goto :FAILED_PIPELINE
)

if not exist "%~dp0fl\models\fedprox_global_final.pt" (
    echo ERROR: fedprox_global_final.pt not found.
    echo [FAILED] Missing fedprox_global_final.pt >> "%ERRLOG%"
    goto :FAILED_PIPELINE
)

if not exist "%~dp0fl\models\scaler.pkl" (
    echo ERROR: scaler.pkl not found.
    echo [FAILED] Missing scaler.pkl >> "%ERRLOG%"
    goto :FAILED_PIPELINE
)

if not exist "%~dp0fl\models\model_metadata.json" (
    echo ERROR: model_metadata.json not found.
    echo [FAILED] Missing model_metadata.json >> "%ERRLOG%"
    goto :FAILED_PIPELINE
)

echo Checkpoints verified successfully.
echo [SUCCESS] All Phase 7 checkpoints verified. >> "%LOG%"


REM =====================================================
REM Step 3: Phase 9 Federated SHAP Pipeline
REM =====================================================

call :Run "cd /d "%~dp0phase9_federated_shap" && python run_pipeline.py --run-name uni_session_1"

if errorlevel 1 goto :FAILED_PIPELINE


REM =====================================================
REM Step 4: SUCCESS
REM =====================================================

echo.
echo ===============================================
echo ALL COMMANDS COMPLETED SUCCESSFULLY
echo ===============================================
echo.
echo Phase 7 Federated Learning completed.
echo Phase 9 Federated SHAP completed.
echo.
echo Results:
echo   %~dp0phase9_federated_shap\results\uni_session_1\
echo.
echo Log:
echo   %LOG%
echo.
echo Errors:
echo   %ERRLOG%
echo.

echo =============================================== >> "%LOG%"
echo Pipeline completed successfully %DATE% %TIME% >> "%LOG%"
echo =============================================== >> "%LOG%"

pause
exit /b 0


REM =====================================================
REM Command Runner
REM =====================================================

:Run
set "CMD=%~1"

echo.
echo ==================================================
echo Running:
echo %CMD%
echo ==================================================

echo. >> "%LOG%"
echo ================================================== >> "%LOG%"
echo [%DATE% %TIME%] RUNNING: %CMD% >> "%LOG%"
echo ================================================== >> "%LOG%"

cmd /c "%CMD%" >> "%LOG%" 2>&1

if errorlevel 1 (
    echo FAILED
    echo [FAILED] %CMD% >> "%LOG%"
    echo [FAILED] %CMD% >> "%ERRLOG%"
    exit /b 1
) else (
    echo SUCCESS
    echo [SUCCESS] %CMD% >> "%LOG%"
    exit /b 0
)


REM =====================================================
REM Pipeline Failure
REM =====================================================

:FAILED_PIPELINE

echo.
echo ===============================================
echo PIPELINE FAILED
echo ===============================================
echo.
echo Check:
echo   %LOG%
echo   %ERRLOG%
echo.

echo =============================================== >> "%LOG%"
echo PIPELINE FAILED %DATE% %TIME% >> "%LOG%"
echo =============================================== >> "%LOG%"

pause
exit /b 1