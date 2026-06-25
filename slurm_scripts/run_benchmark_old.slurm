#!/bin/bash
#SBATCH --job-name=omp4py_ciclo
#SBATCH --time=00:20:00
#SBATCH --partition=gpu          
#SBATCH --gres=gpu:a100:1        
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
set -e
module purge
module load cesga/2020
module load cuda/system
module load nvhpc/21.2

# Obligamos a ejecutar en la GPU
export OMP_TARGET_OFFLOAD=MANDATORY

echo "=========================================================="
echo "  LANZANDO BENCHMARK DE CICLO DE VIDA (OMP4PY vs NUMBA)"
echo "=========================================================="
# Raíz del repo = directorio desde el que se lanza el sbatch (reproducible)
cd $SLURM_SUBMIT_DIR

rm -rf .omp4py_cache/
rm -rf gpu_build/ 
rm -f *.so
# 2. FORZAMOS LA INSTALACIÓN EDITABLE CON EL PIP DE TU ENTORNO
$HOME/miniconda3/envs/thesis_env/bin/pip install -e .
# Forzamos la ejecución con el Python moderno de tu entorno
$HOME/miniconda3/envs/thesis_env/bin/python -u scripts_analisis/benchmark_jit.py

echo "=========================================================="
echo "  BENCHMARK FINALIZADO"
echo "=========================================================="
