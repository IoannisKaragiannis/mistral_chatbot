#!/bin/bash

# Control variables (modify them accordingly to your OS and HW)
PYTHON="python3.10"
VENVS_PATH=$HOME/python_venv
NCORES=$(nproc)

sudo apt-get update

# essential native packages for virtual environment creation
sudo apt-get install python3-pip $PYTHON-venv git

# Check if exactly two arguments are provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <name_of_venv>"
  exit 1
fi

# Pass the input arguments in variables with proper names
VENV_NAME=$1


# Create directory where all pythonic virtual environments are stored
if [ ! -d "$VENVS_PATH" ]; then
    echo "***************************************************"
    echo "Creating directory to store virtual environments"
    echo "***************************************************"
    mkdir -p $VENVS_PATH
fi

REQUIREMENTS=requirements.txt

echo "**************************[$VENV_NAME-SETUP]*******************************"
echo "Installing essential packages for $VENV_NAME" virtual environment
echo "***************************************************************************"

$PYTHON -m venv $VENVS_PATH/$VENV_NAME

# Activate virtual environment
source $VENVS_PATH/$VENV_NAME/bin/activate

# Upgrade important packages such as pip to fetch most recent packages with pip install
$VENVS_PATH/$VENV_NAME/bin/$PYTHON -m pip install --upgrade pip setuptools wheel

# Onstall requirements on virtual environment
$PYTHON -m pip install -r requirements/$REQUIREMENTS

echo "*********************************************************************"
echo "Virtual environment '$VENV_NAME' was successfully setup and it takes" 
du -sh $VENVS_PATH/$VENV_NAME
echo "********************************************************************"

echo "To activate the venv: source $VENVS_PATH/$VENV_NAME/bin/activate"
