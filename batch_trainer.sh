#!/bin/bash -l
#SBATCH -J image_rotation_free_embedding
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 7
#SBATCH -G 1
#SBATCH --time=20:00:00
#SBATCH -p gpu

print_error_and_exit() {
    echo "***ERROR*** $*"
    exit 1
}

module purge || print_error_and_exit "No 'module' command"
module load lang/Python/3.11.5-GCCcore-13.2.0
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Ensure we're in the submission directory
cd "$SLURM_SUBMIT_DIR" || print_error_and_exit "Cannot cd to SLURM_SUBMIT_DIR"

echo "Running on $(hostname)"
echo "Working directory: $(pwd)"

source .venv/bin/activate

echo "Running main.py"
mkdir -p checkpoints

# Enable nullglob so the loop is skipped (not literal) if no files match
shopt -s nullglob
configs=(train_jsons/config_*.json)
shopt -u nullglob

if [ ${#configs[@]} -eq 0 ]; then
    print_error_and_exit "No config files found in train_json/"
else
    echo "${configs[@]}"
fi

for config in "${configs[@]}"; do
    echo "Using config: $config"
    echo " ----------------------------------------"
    cat $config
    echo " ----------------------------------------"
    checkpoint_dir=$(basename "$config" .json | cut -d_ -f2-)
    srun --unbuffered python main.py \
        --json "${config}"
done

echo "Done"