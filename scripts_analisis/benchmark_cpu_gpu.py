import time
import csv
import os
# Ajusta este import según la ruta real de tu decorador en el CESGA
from omp4py import omp 

# ==========================================
# 1. DEFINICIÓN DE LAS FUNCIONES
# ==========================================

# Motor 1: CPython Puro (CPU Secuencial)
def pi_cpu_nativa(N, step):
    pi_value = 0.0
    for i in range(N):
        x = (i + 0.5) * step
        pi_value += 4.0 / (1.0 + x * x)
    return pi_value * step

# Motor 2: omp4py JIT (GPU Secuencial)
@omp(debug=True)
def pi_gpu_omp4py(N: int, step: float):
    pi_value = 0.0
    with omp("target map(tofrom: pi_value)"):
        for i in range(N):
            x = (i + 0.5) * step
            pi_value += 4.0 / (1.0 + x * x)
    return pi_value * step
# ==========================================
# 2. CONFIGURACIÓN DEL BENCHMARK
# ==========================================
tamaños_N = [1000, 1_000_000, 10_000_000, 100_000_000]
iteraciones_por_punto = 10
resultados = []

print("🚀 INICIANDO BENCHMARK JUSTO: CPU Nativa vs GPU OMP4PY 🚀")
print("-" * 60)

# ==========================================
# 3. FASE DE CALENTAMIENTO (WARM-UP)
# ==========================================
print("[Fase 1] Calentando motores y compilando JIT (Cold Start)...")
N_warmup = 1000
step_warmup = 1.0 / N_warmup

# Calentamiento CPU
pi_cpu_nativa(N_warmup, step_warmup)

# Calentamiento GPU (Aquí se comerá los 8.5 segundos del nvc)
t0 = time.perf_counter()
pi_gpu_omp4py(N_warmup, step_warmup)
t1 = time.perf_counter()
print(f"➜ Compilación JIT completada en {t1 - t0:.4f} segundos.\n")

# ==========================================
# 4. FASE DE MEDICIÓN ESTADÍSTICA
# ==========================================
print("[Fase 2] Lanzando batería de mediciones aisladas...")

for N in tamaños_N:
    step = 1.0 / N
    print(f"  Analizando N = {N}...")
    
    for i in range(iteraciones_por_punto):
        # Medición CPU Nativa
        t_inicio = time.perf_counter()
        res_cpu = pi_cpu_nativa(N, step)
        t_fin = time.perf_counter()
        resultados.append(["CPython", N, i+1, t_fin - t_inicio])
        
        # Medición GPU omp4py
        t_inicio = time.perf_counter()
        res_gpu = pi_gpu_omp4py(N, step)
        t_fin = time.perf_counter()
        resultados.append(["omp4py_GPU", N, i+1, t_fin - t_inicio])

# ==========================================
# 5. GUARDADO DE DATOS
# ==========================================
csv_filename = "resultados_justos_cpu_gpu.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Motor", "N", "Iteracion", "Tiempo_s"])
    writer.writerows(resultados)

print("-" * 60)
print(f"✅ Batería finalizada. Resultados guardados en '{csv_filename}'.")
