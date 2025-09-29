#!/usr/bin/env bash

# Prepend repo root (current dir) to PYTHONPATH if not already present
if [[ ":${PYTHONPATH:-}:" != *":$PWD:"* ]]; then
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
fi

echo "PYTHONPATH=$PYTHONPATH"
pip install -e .