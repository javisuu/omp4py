import sys
import os
sys.path.append(os.path.abspath(".."))
from omp4py import omp

@omp("target map(tofrom: resultado)")
def calculo_tonto(N):
    resultado = 0.0
    for i in range(N):
        resultado += i * 2.0
    return resultado

# Forzamos 1 sola ejecución para medir el Cold Start
calculo_tonto(100)
