#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Genera las figuras de rendimiento del TFG a partir de los CSV de data/.
# Uso (desde la raíz del repo):  Rscript scripts_analisis/graficas.R
# Dependencias:  install.packages(c("ggplot2","dplyr","readr","scales"))
# Salida: grafica_benchmark_gpu.pdf, rendimiento_cpu_gpu.pdf, jit_overhead.pdf
#         (cópialos luego a la carpeta figuras/ de la memoria)
# ---------------------------------------------------------------------------

library(ggplot2)
library(dplyr)
library(readr)
library(scales)

tema <- theme_bw(base_size = 12) +
  theme(legend.position = "bottom",
        legend.title = element_blank(),
        panel.grid.minor = element_blank())

# Media por motor y tamaño de problema (las columnas vienen como Motor,N,Iteracion,Tiempo_s)
agg <- function(path) {
  read_csv(path, show_col_types = FALSE) |>
    rename(Motor = 1, N = 2, Tiempo = 4) |>
    group_by(Motor, N) |>
    summarise(Tiempo = mean(Tiempo), .groups = "drop")
}

log_plot <- function(d, colores) {
  ggplot(d, aes(N, Tiempo, color = Motor, shape = Motor)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2.5) +
    scale_x_log10(labels = trans_format("log10", math_format(10^.x))) +
    scale_y_log10(labels = trans_format("log10", math_format(10^.x))) +
    annotation_logticks(sides = "bl") +
    scale_color_manual(values = colores) +
    labs(x = "N (iteraciones)", y = "Tiempo de ejecución (s)") +
    tema
}

# ---- 1. Aislado: Numba (CPU/LLVM) vs omp4py (GPU secuencial) -> Fig. 5.7.3 ----
d1 <- agg("data/resultados_benchmark.csv") |>
  mutate(Motor = recode(Motor,
                        numba  = "Numba (LLVM, CPU vectorizada)",
                        omp4py = "omp4py (GPU, 1 hilo)"))
ggsave("grafica_benchmark_gpu.pdf",
       log_plot(d1, c("#1b9e77", "#d95f02")), width = 7, height = 4.5)

# ---- 2. CPython puro vs omp4py (GPU) -> Fig. 5.7.4 ----
d2 <- agg("data/resultados_justos_cpu_gpu.csv") |>
  mutate(Motor = recode(Motor,
                        CPython    = "CPython (intérprete, CPU)",
                        omp4py_GPU = "omp4py (GPU, 1 hilo)"))
ggsave("rendimiento_cpu_gpu.pdf",
       log_plot(d2, c("#7570b3", "#d95f02")), width = 7, height = 4.5)

# ---- 3. Desglose del overhead JIT (Cold Start, RUN 1 del build verificado) ----
ovh <- data.frame(
  Fase = factor(c("Frontend\n(AST/Tipos)", "Generación\nCython/C", "Compilación\nNVC"),
                levels = c("Frontend\n(AST/Tipos)", "Generación\nCython/C", "Compilación\nNVC")),
  ms = c(0.106, 471.994, 2989.949))   # RUN 1 cold start; total 3462.9 ms
p3 <- ggplot(ovh, aes(Fase, ms, fill = Fase)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = sprintf("%.1f ms", ms)), vjust = -0.4, size = 3.5) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = NULL, y = "Tiempo (ms)",
       subtitle = "Cold Start total ≈ 3463 ms (NVC = 86%)") +
  tema
ggsave("jit_overhead.pdf", p3, width = 7, height = 4.5)

cat("OK -> grafica_benchmark_gpu.pdf, rendimiento_cpu_gpu.pdf, jit_overhead.pdf\n")
