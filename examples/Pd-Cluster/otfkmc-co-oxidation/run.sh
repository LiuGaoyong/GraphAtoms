#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --ntasks-per-node=1
#SBATCH --tasks-per-node=1
#SBATCH --job-name=OTFKMC-Pd-nogas-conf-10
#SBATCH --partition=hfacnormal01
#SBATCH --exclusive

set +u
set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do
        DIR="$(cd -P -- "$(dirname -- "$SOURCE")" && pwd)"
        SOURCE="$(readlink -- "$SOURCE")"
        [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
    done
    SCRIPT_DIR="$(cd -P -- "$(dirname -- "$SOURCE")" && pwd)"
fi; CURRENT_DIR="$(pwd -P)"
echo "SCRIPT_DIR : $SCRIPT_DIR"
echo "CURRENT_DIR: $CURRENT_DIR"


if [[ "$SCRIPT_DIR" == "$CURRENT_DIR" ]]; then
  rm -rf $SCRIPT_DIR/outputs $SCRIPT_DIR/config.yaml
  graphatoms-config parallel=ray          \
    restart=false outputs=./outputs       \
    atoms.filename=../structure.xyz       \
    +event.gas_sticking="{CO:1.0}"        \
    +event.gas_sticking="{O2:1.0}"        \
    exploration.maxtry="3"                \
    exploration.maxconfidence="3"         \
    ~calculator \
    +calculator="{_target_:ase.calculators.emt.EMT}"  |\
	sed 's/NequIP-OAM-S-0.1/PdAgCHO-S/g' | tee config.yaml
  rm -rf $SCRIPT_DIR/outputs
  echo "Configuration Generation Finished!"
  echo ========================================================
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "== Slurm Environment =="
    echo "Job ID       : ${SLURM_JOB_ID}"
    echo "Job Name     : ${SLURM_JOB_NAME:-}"
    echo "Node List    : ${SLURM_JOB_NODELIST:-}"
    echo "Submit Dir   : ${SLURM_SUBMIT_DIR:-}"
    echo "NTasks       : ${SLURM_NTASKS:-}"
    echo "Proc ID      : ${SLURM_PROCID:-}"
    echo "Local ID     : ${SLURM_LOCALID:-}"
    echo "Starting Ray ...................."
    # Getting the node names
    nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    port=6379
    ip_head=$head_node:$port
    export ip_head
    echo "IP Head: $ip_head"
    # Start Ray cluster using symmetric_run.py on all nodes.
    # Symmetric run will automatically start Ray on all nodes and run the script ONLY the head node.
    # Use the '--' separator to separate Ray arguments and the entrypoint command.
    # The --min-nodes argument ensures all nodes join before running the script.

    # All nodes (including head and workers) will execute this block.
    # The entrypoint (simple-trainer.py) will only run on the head node.
    srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
      graphatoms-ray symmetric-run \
      --address "$ip_head" \
      --min-nodes "$SLURM_JOB_NUM_NODES" \
      --num-cpus="${SLURM_CPUS_PER_TASK}" \
      -- graphatoms-run  -cp $SCRIPT_DIR -cn config.yaml
  else
    echo "== Local Environment (No Slurm) =="
    graphatoms-run  -cp $SCRIPT_DIR -cn config.yaml
  fi
else
  echo "Please into the script's directory first."
  echo "cd $SCRIPT_DIR"
fi
