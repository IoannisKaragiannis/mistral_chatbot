#!/bin/bash

PYTHON="python3.10"
PURGE_CACHE=true

# Check if exactly one argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <abs_path_to_venv>"
  exit 1
fi

# Access the single argument
VENV_ABS_PATH=$1

echo "**************************[$VENV_ABS_PATH-REMOVAL]************************"
echo "Uninstalling '$VENV_ABS_PATH' packages and erasing environment"
echo "**********************************************************************"

if [ -d "$VENV_ABS_PATH" ]; then

    # activate environment
    source $VENV_ABS_PATH/bin/activate

    if [ "$PURGE_CACHE" = true ] ; then
        $VENV_ABS_PATH/bin/$PYTHON -m pip cache purge
    fi

    deactivate

    sudo rm -rf $VENV_ABS_PATH

    echo "*******************************************************"
    echo "$VENV_ABS_PATH WAS SUCCESSFULLY UNINSTALLED!!!"
    echo "*******************************************************"
else
    echo "[WARNING]:: $VENV_ABS_PATH directory does not exist! Exiting"
    exit 1
fi

rm -rf models
