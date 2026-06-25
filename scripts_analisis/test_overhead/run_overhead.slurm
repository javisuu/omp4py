#!/bin/bash
#SBATCH --job-name=omp4py_overhead
#SBATCH --time=00:10:00
#SBATCH --partition=gpu          
#SBATCH --gres=gpu:a100:1        
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=8G

module purge
module load cesga/2020
module load cuda/system
module load nvhpc/21.2

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate thesis_env

echo "==============================================="
echo "  MIDIENDO OVERHEAD (5 EJECUCIONES COLD START)"
echo "==============================================="

# Aquí está el bucle mágico
for i in {1..5}; do 
    echo "--- RUN $i ---"
    rm -rf ../gpu_build
    $HOME/miniconda3/envs/thesis_env/bin/python -u test_overhead.py | grep TELEMETRÍA
done

echo "==============================================="
echo "  MEDICIÓN FINALIZADA"
echo "==============================================="
