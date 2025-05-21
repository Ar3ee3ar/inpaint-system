@echo off
cd client_node
:Start
REM Ask the user for the number of iterations
set /p iterations="Enter the number of client node to run the Python file: "

REM Trim any leading/trailing spaces from input
for /f "delims=" %%a in ("%iterations%") do set iterations=%%a
echo %iterations%

REM Check if the input is numeric
SET "var="&for /f "delims=0123456789" %%i in ("%iterations%") do set var=%%i
if defined var (
    echo Invalid input. Please enter a numeric value.
    goto Start
)
REM Run the Python file in a loop
for /L %%i in (1,1,%iterations%) do (
    start cmd /k ".\venv\scripts\activate & python client.py"
)

