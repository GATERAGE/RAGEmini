#!/usr/bin/env bash
# RAGE (c) 2025 Gregory L. Magnusson MIT
# script to automate the creation of the rage venv with min_python_version variable and installation of requirements
# chmod +x install.sh
# ./install.sh

# Minimum required Python version
min_python_version="3.9"

# Find an available Python 3 interpreter
python_interpreter=""
for version in $(seq 11 -1 9); do  # Check from 3.11 down to 3.9
  if command -v python3."$version" &>/dev/null; then
    python_interpreter="python3.$version"
    break
  fi
done

# If no suitable Python 3 interpreter is found, use 'python3' as a fallback
if [[ -z "$python_interpreter" ]]; then
  if command -v python3 &>/dev/null; then
    python_interpreter="python3"
  else
    echo "Error: No suitable Python 3 interpreter found."
    exit 1
  fi
fi

# Get the Python version
python_version=$("$python_interpreter" --version 2>&1 | awk '{print $2}')

# Check if the Python version meets the minimum requirement
if [[ "$(printf '%s\n%s' "$min_python_version" "$python_version" | sort -V | head -n1)" == "$min_python_version" ]]; then
    echo "Using Python interpreter: $python_interpreter ($python_version)"
else
    echo "Error: Python version $python_version is below the minimum requirement ($min_python_version)."
    exit 1
fi

# Create the virtual environment
"$python_interpreter" -m venv rage
if [[ $? -ne 0 ]]; then
  echo "Error: Failed to create virtual environment."
  exit 1
fi

# Activate the virtual environment
source rage/bin/activate
if [[ $? -ne 0 ]]; then
  echo "Error: Failed to activate virtual environment."
  exit 1
fi

# Install dependencies from requirements.txt
pip install -r requirements.txt
if [[ $? -ne 0 ]]; then
  echo "Error: Failed to install dependencies from requirements.txt."
  exit 1
fi

# Install the package in editable mode (if setup.py or pyproject.toml exists)
if [[ -f setup.py ]] || [[ -f pyproject.toml ]]; then
    pip install -e .
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to install package in editable mode."
        exit 1
    fi
    echo "Package installed in editable mode."
else
    echo "Warning: No setup.py or pyproject.toml found. Skipping editable install."
fi

echo "Virtual environment 'rage' created and activated. Dependencies installed."
