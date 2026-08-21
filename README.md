# Proyecto 1: Competencia de Modelacion de Deep Learning
## Deep Learning Housing Price Regression (PyTorch Tabular ResNet-MLP)

Este repositorio contiene la solucion completa para el **Proyecto 1: Competencia de Modelacion (CC3092 Deep Learning y sistemas inteligentes)**. El proyecto implementa una arquitectura neuronal optimizada para datos tabulares mediante bloques residuales (**Tabular ResNet-MLP** con normalizacion por capa LayerNorm, activaciones GELU, regularizacion por Dropout y funcion de perdida Huber/Smooth L1).

---

## Estructura del Repositorio

```
PRY1-DL/
├── data/
│   ├── train.csv                      # Dataset de entrenamiento principal
│   ├── saved_models/                  # Checkpoints del modelo y preprocesador (Tracked en Git)
│   │   ├── pipeline.joblib            # Preprocesador ajustado (248 dimensiones)
│   │   ├── model.pt                   # Pesos del modelo final de produccion (4.6 MB)
│   │   ├── model_config.json          # Configuracion y metricas del modelo
│   │   └── experiment_summary.json    # Resumen comparativo de iteraciones
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
│   ├── 02_model_experiments.ipynb     # Metodologia y comparativa de modelos
│   └── 03_evaluation_and_submission.ipynb # Verificacion del pipeline de prueba
├── src/
│   ├── data_processing.py             # Preprocesador tabular robusto
│   ├── dataset.py                     # Wrapper PyTorch Dataset/DataLoader
│   ├── models.py                      # Arquitecturas neuronales (Tabular ResNet, SwiGLU, etc.)
│   ├── train.py                       # Pipeline de entrenamiento y validacion
│   ├── evaluate.py                    # Calculo de metricas y residuales
│   └── utils.py                       # Semillas, metricas y graficado
├── main.py                            # CLI unico del pipeline (Entrenamiento e Inferencia)
├── .gitignore                         # Configuracion de rastreo Git (incluye modelo y CSVs)
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

Para reproducir todo el pipeline (analisis EDA, entrenamiento del modelo Tabular ResNet y evaluacion de residuos):

```bash
python main.py train
```

---

## Control de Versiones con Git (`.gitignore`)

El archivo `.gitignore` esta configurado para incluir en el repositorio todos los archivos CSV de datos y el checkpoint final del modelo:
* Checkpoint entrenado (`data/saved_models/model.pt`, `data/saved_models/pipeline.joblib`, `data/saved_models/model_config.json`).
* Datasets y archivos CSV (`data/*.csv`, `data/**/*.csv`).
* Excluye carpetas de entorno virtual (`.venv/`), cache de Python (`__pycache__/`) y archivos auxiliares de LaTeX (`*.aux`, `*.log`).

---

## Resumen del Modelo Seleccionado (Tabular ResNet-MLP)

* **Arquitectura**: Tabular ResNet-MLP (2 bloques residuales, 512 unidades ocultas, LayerNorm, Dropout 0.20)
* **Validacion Fuera de Muestra (10-Fold OOF RMSE)**: **$24,953.61**
* **Coeficiente de Determinacion ($R^2$)**: **0.8956** (Casi el 90% de varianza explicada)
* **Error Absoluto Medio (MAE)**: **$14,267.36**
* **Error Relativo Porcentual (MAPE)**: **8.10%**
* **Justificacion de Seleccion**: Punto optimo de parsimonia estructural, eliminando el riesgo de sobreajuste (\textit{overfitting}) de ensambles complejos y garantizando un despliegue liviano (4.6 MB) y determinista.
