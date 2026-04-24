#!/bin/bash
#SBATCH --job-name=omp4py_thesis_test
#SBATCH --output=thesis_test_%j.out
#SBATCH --error=thesis_test_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=gpu            # Change this if CESGA requires a specific A100 partition name
#SBATCH --gres=gpu:a100:1          # Requesting exactly 1 A100 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G

echo "🚀 Job started on node: $SLURM_NODELIST"
echo "🕒 Date: $(date)"
echo "----------------------------------------"

# 1. Load the exact environment from your MVP
module purge
module load cesga/2020
module load cuda/system
module load nvhpc/21.2

#CUSTOM PYTHON ENVIRONMENT
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate thesis_env

# 2. Navigate to the directory (SLURM usually starts where you submit, but just to be safe)
cd $SLURM_SUBMIT_DIR

# 3. Run the PyTest suite
echo "🧪 Running OMP4Py GPU Test Suite..."
# $HOME/miniconda3/envs/thesis_env/bin/pytest test/basic/test_target.py -v
python main.py
echo "----------------------------------------"
echo "🏁 Tests Finished Successfully!"
