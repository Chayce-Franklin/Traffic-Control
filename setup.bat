@echo off
echo ===================================================
echo Setting up Smart Traffic Prediction environment...
echo ===================================================
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo ---------------------------------------------------
echo Setup complete!
echo To run your program, double-click run.bat
echo ---------------------------------------------------
pause