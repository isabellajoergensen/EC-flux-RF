# EC Flux Random Forest Analysis

This project uses machine learning to analyze and predict energy flux data.

You do not need machine learning experience to run it. Follow the steps below exactly, and you should be able to install and run everything.

## What this project does

The code:
1. Trains a model using weather and flux data.
2. Measures which weather variables are most important.
3. Saves results as tables and figures.

## Before you start

You only need:
- Python 3.7 or newer (recommended: latest Python 3)
- VS Code (recommended, but not required)

Download Python here:
https://www.python.org/downloads/

Important during Python install on Windows:
- Check the box that says "Add Python to PATH".

## Why use a virtual environment?

A virtual environment is a small, isolated Python setup just for this project.

It helps you:
- Avoid conflicts with other Python projects.
- Install exactly the packages this repo needs.
- Keep your computer's global Python clean.

## Step-by-step setup (first time)

### 1. Download or clone this repository

Put the project folder somewhere easy to find.

### 2. Open the project folder in VS Code

In VS Code:
1. Click File -> Open Folder.
2. Select this project folder.

### 3. Open a terminal in VS Code

In VS Code:
1. Click Terminal -> New Terminal.
2. A terminal panel opens at the bottom.

### 4. Create a virtual environment

In the terminal, run:

```bash
python -m venv .venv
```

This creates a folder named `.venv` inside the project.

### 5. Activate the virtual environment

Run the command for your operating system:

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt (cmd):
```cmd
.venv\Scripts\activate.bat
```

macOS/Linux:
```bash
source .venv/bin/activate
```

After activation, you should usually see `(.venv)` at the start of the terminal line.

Important:
- You can continue in the same terminal right away. You do not need to open a new terminal.
- Selecting the VS Code interpreter is recommended, but not required to run commands in a terminal where `(.venv)` is active.

### 6. Install project dependencies

With the virtual environment active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs all required packages from `requirements.txt`.

### 7. Select the virtual environment in VS Code

This is recommended because it makes sure the Run button, debugging, and new terminals use the correct Python.

1. Press Ctrl+Shift+P in VS Code.
2. Type `Python: Select Interpreter`.
3. Choose the interpreter that includes `.venv` in its path.

## Run the scripts

Run either script from the terminal:

```bash
python NEE_shap_analysis_modified.py
```

or

```bash
python Plots.py
```

## Expected output

- The scripts may take a few minutes.
- Progress and messages appear in the terminal.
- Result files are saved by the scripts (for example plot images and analysis outputs).

## Running again later

When you come back another day:
1. Open project folder in VS Code.
2. Open terminal.
3. Activate virtual environment again.
4. Run the script.

You do not need to reinstall dependencies every time.

## Troubleshooting

### "python is not recognized" (Windows)

Python is not available in your PATH.
- Reinstall Python from python.org and check "Add Python to PATH" during install.
- Then close and reopen VS Code.

### PowerShell says script execution is disabled

In PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### "ModuleNotFoundError" when running a script

Usually means dependencies are not installed in the active environment.

Fix:
1. Make sure `(.venv)` appears in terminal.
2. Run:

```bash
python -m pip install -r requirements.txt
```

## Project files

- `NEE_shap_analysis_modified.py`: Main analysis script
- `Plots.py`: Plot/visualization script
- `requirements.txt`: Python dependencies
- `data/`: Input CSV data files
