from omp4py import omp 

@omp("target map(tofrom: resultado)", cache=False, debug=True)
def bucle_roto(N):
    resultado = 0.0
    for i in range(N):
        # Introducimos semántica puramente dinámica que el silicio no entiende
        diccionario_trampa = {"valor": 1.5}
        texto = "esto_rompe_el_compilador_C"
        resultado += diccionario_trampa["valor"]
    return resultado

print("Ejecutando prueba de estrés...")
res = bucle_roto(100)
print(f"Resultado final: {res}")
