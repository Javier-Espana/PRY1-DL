# Proyecto 1: Competencia de Modelacion de Deep Learning
## Deep Learning Housing Price Regression (PyTorch Tabular ResNet & SwiGLU Ensemble)

Este repositorio contiene la solucion completa para el **Proyecto 1: Competencia de Modelacion (CC3092 Deep Learning y sistemas inteligentes)**. El proyecto desarrolla una arquitectura de Redes Neuronales Multicapa (*Multi-Layer Perceptrons*) optimizada para datos tabulares mediante bloques residuales (**ResNet-MLP**), unidades lineales de compuerta (**SwiGLU-MLP**) y un ensamble multi-semilla de 60 modelos (**Multi-Seed 10-Fold Cross Validation Ensemble**).

---

## Estructura del Repositorio

```
PRY1-DL/
├── data/
│   ├── train.csv                      # Dataset de entrenamiento principal
│   ├── saved_models/                  # Checkpoints del modelo y preprocesador (Tracked en Git)
│   │   ├── pipeline.joblib            # Preprocesador ajustado
│   │   ├── model_*.pt                 # Pesos de los 60 modelos del ensamble
│   │   ├── model_config.json          # Manifiesto y configuracion de hiperparametros
│   │   └── experiment_summary.json    # Resumen comparativo de experimentos
│   └── pruebas/
│       ├── pipeline_test.csv          # Muestra del dataset de prueba
│       └── expected_output.csv        # Formato de referencia de predicciones
├── docs/
│   ├── Informe_Proyecto_1.tex         # Informe academico en formato LaTeX
│   ├── Informe_Proyecto_1.pdf         # PDF compilado del informe academico
│   ├── Proyecto #1. Competencia de modelacion.pdf # Especificaciones del proyecto
│   └── plots/                         # Graficas de EDA y residuales generadas
├── notebooks/
│   ├── 01_eda.ipynb                   # Analisis Exploratorio de Datos
│   ├── 02_model_experiments.ipynb     # Experimentos, HPO y benchmarking
│   └── 03_evaluation_and_submission.ipynb # Verificacion del pipeline de prueba
├── src/
│   ├── data_processing.py             # Preprocesador tabular robusto con ingenieria avanzada
│   ├── dataset.py                     # Wrapper PyTorch Dataset/DataLoader
│   ├── models.py                      # Arquitecturas MLP (ResNet, SwiGLU, Wide&Deep, Standard)
│   ├── train.py                       # Pipeline 10-Fold CV & Multi-Seed Ensemble
│   ├── evaluate.py                    # Calculo de metricas y residuales
│   └── utils.py                       # Semillas, metricas y graficado
├── main.py                            # CLI unico del pipeline (Entrenamiento e Inferencia)
├── .gitignore                         # Configuracion de rastreo Git (incluye modelos y CSVs)
└── README.md                          # Manual de uso y reproduccion
```

---

## Compilacion del Informe en LaTeX

El informe tecnico escrito en formato LaTeX se encuentra en `docs/Informe_Proyecto_1.tex`. Para compilarlo manualmente:

```bash
cd docs
pdflatex Informe_Proyecto_1.tex
pdflatex Informe_Proyecto_1.tex
```

---

## Uso de main.py (Pipeline de Entrenamiento e Inferencia)

La ejecucion del proyecto se realiza exclusivamente a traves de `main.py`.

### 1. Inferencia / Generacion de Predicciones (Dia de la Presentacion)

Para generar predicciones a partir de un archivo de prueba con formato identico a `pipeline_test.csv` escribiendo las predicciones finales con el formato exacto requerido (`Id,Prediction`), ejecute:

```bash
python main.py predict --test_path data/pruebas/pipeline_test.csv --output_path data/pruebas/expected_output.csv
```

Parametros de la inferencia:
* `--test_path`: Ruta al CSV de prueba (por defecto `data/pruebas/pipeline_test.csv`).
* `--output_path`: Ruta donde se guardaran las predicciones (por defecto `data/pruebas/expected_output.csv`).
* `--saved_models_dir`: Directorio que contiene los modelos guardados (por defecto `data/saved_models`).

### 2. Entrenamiento y Reproduccion de Experimentos

Para reproducir todo el pipeline (analisis EDA, benchmarking de 4 arquitecturas, entrenamiento del ensamble de 60 modelos y evaluacion de residuos):

```bash
python main.py train
```

---

## Control de Versiones con Git (`.gitignore`)

El archivo `.gitignore` esta configurado para incluir en el repositorio todos los archivos CSV de datos y checkpoints de entrenamiento de los modelos:
* Checkpoints entrenados (`data/saved_models/*.pt`, `data/saved_models/*.joblib`, `data/saved_models/*.json`).
* Datasets y archivos CSV (`data/*.csv`, `data/**/*.csv`).
* Excluye carpetas de entorno virtual (`.venv/`), cache de Python (`__pycache__/`) y archivos auxiliares de LaTeX (`*.aux`, `*.log`).

---

## Resumen de Resultados Finales

* **Arquitecturas del Ensamble**: Tabular ResNet-MLP + SwiGLU-MLP
* **Estrategia de Ensamble**: 60 modelos neuronales (2 arquitecturas x 3 semillas x 10 pliegues)
* **RMSE Out-Of-Fold ($)**: **$26,772.24**
* **Coeficiente de Determinacion ($R^2$)**: **0.8798**
* **Error Relativo Porcentual (MAPE)**: **7.96%**
* **Error Absoluto Medio (MAE)**: **$13,627.83**
