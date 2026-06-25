import sys
import time
from numba import njit
from omp4py import omp

# --- 1. DEFINICIÓN DE FUNCIONES ---
@njit
def pi_numba(N, step):
    pi_value = 0.0
    for i in range(N):
        x = (i + 0.5) * step
        pi_value += 4.0 / (1.0 + x * x)
    return pi_value * step

@omp()
def pi_omp4py(N: int, step: float):
    pi_value = 0.0
    with omp("target map(tofrom: pi_value)"):
        for i in range(N):
            x = (i + 0.5) * step
            pi_value += 4.0 / (1.0 + x * x)
    return pi_value * step

# --- 2. MOTOR DE EJECUCIÓN AISLADA ---
if __name__ == "__main__":
    # Verificamos que el bash nos pasa el motor y la N
    if len(sys.argv) != 3:
        print("Uso: python aislado.py [numba|omp4py] [N]")
        sys.exit(1)

    motor = sys.argv[1]
    N = int(sys.argv[2])
    step = 1.0 / N

    # FASE DE CALENTAMIENTO SILENCIOSO (Warm-up)
    # Obligamos a compilar y cargar en GPU con un N minúsculo
    if motor == "numba":
        _ = pi_numba(100, 1.0/100)
    elif motor == "omp4py":
        _ = pi_omp4py(100, 1.0/100)

    # FASE DE MEDICIÓN REAL PAGA (Warm Start)
    t0 = time.perf_counter()
    if motor == "numba":
        pi_numba(N, step)
    elif motor == "omp4py":
        pi_omp4py(N, step)
    t1 = time.perf_counter()

    # IMPRIMIMOS SOLO EL TIEMPO PARA EL CSV
    print(t1 - t0)
