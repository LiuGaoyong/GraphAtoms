#!/usr/bin/env bash


set -euo pipefail
SOURCE="${BASH_SOURCE[0]}"
# Resolve symlinks to get the real script path
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P -- "$(dirname -- "$SOURCE")" && pwd)"
  SOURCE="$(readlink -- "$SOURCE")"
  [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P -- "$(dirname -- "$SOURCE")" && pwd)"
CURRENT_DIR="$(pwd -P)"


if [[ "$SCRIPT_DIR" == "$CURRENT_DIR" ]]; then
  source $SCRIPT_DIR/../../../.venv/bin/activate
  echo "Activated the virtual environment."
  echo "Python location: $(which python)"
  python --version
  rm -rf $SCRIPT_DIR/outputs
  graphatoms-run  -cp $SCRIPT_DIR -cn config.yaml
else
  echo "Please into the script's directory first."
  echo "cd $SCRIPT_DIR"
fi