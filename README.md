# Proyecto 1: Competencia de Modelacion de Deep Learning
## Deep Learning Housing Price Regression (PyTorch Tabular ResNet-MLP)

Este repositorio contiene la solucion completa para el **Proyecto 1: Competencia de Modelacion (CC3092 Deep Learning y sistemas inteligentes)**. El proyecto desarrolla una arquitectura de Red Neuronal Multicapa (*Multi-Layer Perceptron*) optimizada para datos tabulares mediante conexiones residuales (**ResNet-MLP**) y validacion cruzada de 10 pliegues (**10-Fold Cross Validation**).

---

## Estructura del Repositorio

```
PRY1-DL/
├── data/
│   ├── train.csv                      # Dataset de entrenamiento principal
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
│   ├── 02_model_experiments.ipynb     # Experimentos, HPO y entrenamiento
│   └── 03_evaluation_and_submission.ipynb # Verificacion del pipeline de prueba
├── saved_models/                      # Checkpoints del modelo y preprocesador (Tracked en Git)
│   ├── pipeline.joblib                # Preprocesador ajustado
│   ├── model_fold_*.pt                # Pesos de los 10 pliegues del ensamble
│   └── model_config.json              # Configuracion de hiperparametros
├── src/
│   ├── data_processing.py             # Preprocesador tabular robusto
│   ├── dataset.py                     # Wrapper PyTorch Dataset/DataLoader
│   ├── models.py                      # Arquitecturas MLP (ResNet, Standard, Wide&Deep)
│   ├── train.py                       # Pipeline 10-Fold CV & Optuna HPO
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
* `--saved_models_dir`: Directorio que contiene los modelos guardados (por defecto `saved_models`).

### 2. Entrenamiento y Reproduccion de Experimentos

Para reproducir todo el pipeline (analisis EDA, benchmarking de arquitecturas, optimizacion de hiperparametros con Optuna y entrenamiento final del ensamble de 10 pliegues):

```bash
python main.py train
```

---

## Control de Versiones con Git (`.gitignore`)

El archivo `.gitignore` esta configurado para incluir en el repositorio todos los archivos CSV de datos y checkpoints de entrenamiento de los modelos:
* Checkpoints entrenados (`saved_models/*.pt`, `saved_models/*.joblib`, `saved_models/*.json`).
* Datasets y archivos CSV (`data/*.csv`, `data/**/*.csv`).
* Excluye carpetas de entorno virtual (`.venv/`), cache de Python (`__pycache__/`) y archivos auxiliares de LaTeX (`*.aux`, `*.log`).

---

## Resumen de Resultados Finales

* **Arquitectura Ganadora**: Tabular ResNet-MLP (3 Bloques Residuales, GELU, LayerNorm, Dropout 0.20)
* **Validacion Cruzada**: 10-Fold Cross-Validation Ensemble
* **RMSE Out-Of-Fold ($)**: **$24,847.81**
* **Coeficiente de Determinacion ($R^2$)**: **0.8965**
* **Error Relativo Porcentual (MAPE)**: **8.10%**
