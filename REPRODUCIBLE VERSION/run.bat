@echo off
echo =================================
echo CPR-SAT Evaluation
echo =================================
echo.

REM Activate conda base properly
call "%USERPROFILE%\anaconda3\Scripts\activate.bat"

REM Activate environment
call conda activate cprsat-eval

echo.
echo Running PPD Evaluation...
python PPD_Eval.py

echo.
echo Running CRD Evaluation...
python CRD_Eval.py

echo.
echo Completed.
pause