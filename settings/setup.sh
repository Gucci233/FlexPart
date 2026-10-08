#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install -r settings/requirements.txt
if [ "${SKIP_SYSTEM_DEPS:-0}" != "1" ]; then
    sudo apt-get install libegl1 libgl1-mesa-dev -y # for rendering
fi
