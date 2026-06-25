#!/bin/bash
#SBATCH --job-name=omp4py_ciclo
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G

set -e

# 1. Módulos base del CESGA
module purge
module load cesga/2020
module load cuda/system
module load nvhpc/21.2

# 2. Entorno Python (LA CLAVE DEL ÉXITO)
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate thesis_env

export OMP_TARGET_OFFLOAD=MANDATORY

echo "=========================================================="
echo "  LANZANDO BENCHMARK DE CICLO DE VIDA (OMP4PY vs NUMBA)"
echo "=========================================================="

# Raíz del repo = directorio desde el que se lanza el sbatch (reproducible)
cd $SLURM_SUBMIT_DIR

# Limpiamos cachés pero RESPETAMOS la carpeta contenedora si existe
rm -f gpu_build/*.so
rm -f gpu_build/*.c

# 3. Instalamos localmente y ejecutamos
pip install -e .
python -u scripts_analisis/benchmark_jit.py

echo "=========================================================="
echo "  BENCHMARK FINALIZADO"
echo "=========================================================="
