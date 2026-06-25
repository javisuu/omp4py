#!/bin/bash
#SBATCH --job-name=omp4py_aislado
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G

module purge
module load cesga/2020
module load cuda/system
module load nvhpc/21.2

export OMP_TARGET_OFFLOAD=MANDATORY

# Raíz del repo = directorio desde el que se lanza el sbatch (reproducible)
cd $SLURM_SUBMIT_DIR

# Variable con tu ruta absoluta de Python para evitar fallos de Numba/Conda
PY_BIN="$HOME/miniconda3/envs/thesis_env/bin/python"

# [DEBUG-REVERT] Diagnóstico de entorno (PATH de cython/nvc)
echo "PYTHON: $($PY_BIN -c 'import sys;print(sys.executable)')"
echo "CYTHON: $(command -v cython)"
echo "NVC:    $(command -v nvc)"
echo "PWD:    $(pwd)"

# Crear el archivo de resultados y ponerle cabecera
ARCHIVO_OUT="resultados_benchmark.csv"
echo "Motor,N,Iteracion,Tiempo_s" > $ARCHIVO_OUT

# Los valores de N que vamos a probar
VALORES_N=(1000 1000000 10000000 100000000)

echo "Iniciando bateria de pruebas aisladas..."

for N in "${VALORES_N[@]}"; do
    for iteracion in {1..10}; do
        
        # 1. Medir Numba
        t_numba=$($PY_BIN -u scripts_analisis/aislado.py numba $N)
        echo "numba,$N,$iteracion,$t_numba" >> $ARCHIVO_OUT

        # 2. Medir omp4py
        t_omp=$($PY_BIN -u scripts_analisis/aislado.py omp4py $N)
        echo "omp4py,$N,$iteracion,$t_omp" >> $ARCHIVO_OUT
        
    done
done

echo "Bateria finalizada. Resultados guardados en $ARCHIVO_OUT"
