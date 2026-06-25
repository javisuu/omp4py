import sys
import os
sys.path.append(os.path.abspath(".."))
from omp4py import omp

@omp()
def calculo_tonto(N: int):
    resultado = 0.0
    with omp("target map(tofrom: resultado)"):
        for i in range(N):
            resultado += i * 2.0
    return resultado

# Forzamos 1 sola ejecución para medir el Cold Start
calculo_tonto(100)
