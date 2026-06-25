import time
from numba import njit

# ---------------------------------------------------------
# 1. Implementación con Numba (LLVM JIT)
# ---------------------------------------------------------
@njit
def pi_numba(N, step):
    pi_value = 0.0
    for i in range(N):
        x = (i + 0.5) * step
        pi_value += 4.0 / (1.0 + x * x)
    return pi_value * step

# ---------------------------------------------------------
# 2. Implementación con tu framework (Cython + nvc JIT)
# ---------------------------------------------------------
t0=time.perf_counter()
from omp4py import omp

@omp()
def pi_omp4py(N: int, step: float):
    pi_value = 0.0
    with omp("target map(tofrom: pi_value)"):
        for i in range(N):
            x = (i + 0.5) * step
            pi_value += 4.0 / (1.0 + x * x)
    return pi_value * step

t1=time.perf_counter()
print(f"Tiempo de decoración: {t1-t0:.4f} s")
# ---------------------------------------------------------
# Motor de pruebas
# ---------------------------------------------------------
def run_test():
    print("🚀 INICIANDO BENCHMARK: OMP4PY vs NUMBA (CICLO DE VIDA) 🚀")
    print("-" * 60)

    # 1. PRUEBA DE OVERHEAD JIT (COLD START)
    N_cold = 1000
    step_cold = 1.0 / N_cold

    print("\n[Fase 1] Midiendo Overhead del JIT (Cold Start)...")

    start = time.perf_counter()
    pi_numba(N_cold, step_cold)
    numba_cold_time = time.perf_counter() - start
    print(f"➜ Numba (LLVM) Overhead:      {numba_cold_time:.4f} segundos")

    start = time.perf_counter()
    pi_omp4py(N_cold, step_cold)
    omp_cold_time = time.perf_counter() - start
    print(f"➜ omp4py (Cython+NVC) Overhead: {omp_cold_time:.4f} segundos")

    # 2. PRUEBA DE ESCALABILIDAD (WARM START)
    print("\n[Fase 2] Midiendo Escalabilidad Computacional (Warm Start)...")
    print(f"{'N (Iteraciones)':<15} | {'Numba (s)':<15} | {'omp4py (s)':<15}")
    print("-" * 60)

    # Escalamos hasta 500 millones de iteraciones
    N_values = [10**6, 10**7, 10**8, 5 * 10**8]

    for N in N_values:
        step = 1.0 / N

        # Medición Numba
        start = time.perf_counter()
        pi_numba(N, step)
        t_numba = time.perf_counter() - start

        # Medición omp4py
        start = time.perf_counter()
        pi_omp4py(N, step)
        t_omp = time.perf_counter() - start

        print(f"{N:<15} | {t_numba:<15.4f} | {t_omp:<15.4f}")

if __name__ == "__main__":
    run_test()
