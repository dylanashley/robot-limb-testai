#! /bin/bash

# Get the script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Setup the model path
export GZ_SIM_RESOURCE_PATH=$SCRIPT_DIR/models:$GZ_SIM_RESOURCE_PATH
echo "GZ_SIM_RESOURCE_PATH"
echo $GZ_SIM_RESOURCE_PATH

# Setup the plugin path
export GZ_SIM_SYSTEM_PLUGIN_PATH=${GZ_SIM_SYSTEM_PLUGIN_PATH}:$SCRIPT_DIR/plugins
echo "GZ_SIM_SYSTEM_PLUGIN_PATH"
echo $GZ_SIM_SYSTEM_PLUGIN_PATH
