#!/bin/bash
#SBATCH -J omp4py_bench
#SBATCH -o benchmark_justo_%j.out
#SBATCH -e benchmark_justo_%j.err
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gres=gpu:a100:1
#SBATCH -t 00:20:00
#SBATCH --mem=16G

# 1. Cargar el stack de software del superordenador
module purge
module load cesga/2020 cuda/system nvhpc/21.2

# 2. Inicializar Conda usando la ruta absoluta y activar el entorno
# (Ajusta "miniconda3" a "anaconda3" si usaste la versión completa)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_env

# 3. Forzar ejecución en hardware real sin fallback
export OMP_TARGET_OFFLOAD=MANDATORY

# 4. Lanzar el script de Python
python benchmark_cpu_gpu.py
