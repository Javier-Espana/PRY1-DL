# Proyecto 1: Competencia de Modelación de Deep Learning
## Deep Learning Housing Price Regression (PyTorch Tabular ResNet-MLP)

Este repositorio contiene la solución completa para el **Proyecto 1: Competencia de Modelación (CC3092 Deep Learning y sistemas inteligentes)**. El proyecto desarrolla una arquitectura de Red Neuronal Multicapa (*Multi-Layer Perceptron*) optimizada para datos tabulares mediante conexiones residuales (**ResNet-MLP**) y validación cruzada de 10 pliegues (**10-Fold Cross Validation**).

---

## 📁 Estructura del Repositorio

```
PRY1-DL/
├── data/
│   ├── train.csv                      # Dataset de entrenamiento principal
│   └── pruebas/
│       ├── pipeline_test.csv          # Muestra del dataset de prueba
│       └── expected_output.csv        # Formato de referencia de predicciones
├── docs/
│   ├── Informe_Proyecto_1.tex         # Informe académico en formato LaTeX
│   ├── Informe_Proyecto_1.pdf         # PDF compilado del informe académico
│   ├── Proyecto #1. Competencia de modelación.pdf # Especificaciones del proyecto
│   └── plots/                         # Gráficas de EDA y residuales generadas
├── notebooks/
│   ├── 01_eda.ipynb                   # Análisis Exploratorio de Datos
│   ├── 02_model_experiments.ipynb     # Experimentos, HPO y entrenamiento
│   └── 03_evaluation_and_submission.ipynb # Verificación del pipeline de prueba
├── saved_models/                      # Checkpoints del modelo y preprocesador (Tracked en Git)
│   ├── pipeline.joblib                # Preprocesador ajustado
│   ├── model_fold_*.pt                # Pesos de los 10 pliegues del ensamble
│   └── model_config.json              # Configuración de hiperparámetros
├── src/
│   ├── data_processing.py             # Preprocesador tabular robusto
│   ├── dataset.py                     # Wrapper PyTorch Dataset/DataLoader
│   ├── models.py                      # Arquitecturas MLP (ResNet, Standard, Wide&Deep)
│   ├── train.py                       # Pipeline 10-Fold CV & Optuna HPO
│   ├── evaluate.py                    # Cálculo de métricas y residuales
│   └── utils.py                       # Semillas, métricas y graficado
├── predict_competition.py             # CLI de inferencia automática para el día de prueba
├── run_experiments.py                 # Script ejecutor completo de experimentos
├── .gitignore                         # Configuración de rastreo Git (incluye modelos y CSVs)
└── README.md                          # Manual de uso y reproducción
```

---

## 📄 Compilación del Informe en LaTeX

El informe técnico escrito en formato LaTeX se encuentra en `docs/Informe_Proyecto_1.tex`. Para compilarlo manualmente:

```bash
cd docs
pdflatex Informe_Proyecto_1.tex
pdflatex Informe_Proyecto_1.tex
```

---

## ⚡ Generación de Predicciones (Día de la Presentación)

Para generar las predicciones sobre el nuevo conjunto de datos de prueba el día de la presentación, ejecute el script `predict_competition.py`:

```bash
python predict_competition.py --test_path data/pruebas/pipeline_test.csv --output_path data/pruebas/expected_output.csv
```

### Parámetros opcionales:
- `--test_path`: Ruta al CSV de prueba (por defecto `data/pruebas/pipeline_test.csv`).
- `--output_path`: Ruta donde se guardará el CSV de resultados (por defecto `data/pruebas/output.csv`).
- `--saved_models_dir`: Directorio con los modelos guardados (por defecto `saved_models`).

El archivo de salida tendrá el formato exacto requerido:
```csv
Id,Prediction
893,146938.98
1106,329644.60
...
```

---

## 🏋️ Entrenamiento y Reproducción de Experimentos

Para reproducir todo el pipeline desde cero (EDA, benchmarking de arquitecturas, optimización de hiperparámetros con Optuna y entrenamiento del ensamble final de 10 pliegues):

```bash
python run_experiments.py
```

---

## ⚙️ Control de Versiones con Git (`.gitignore`)

El archivo `.gitignore` está configurado para **incluir e ingresar en el repositorio todos los artefactos de modelos y datos**:
- Checkpoints entrenados (`saved_models/*.pt`, `saved_models/*.joblib`, `saved_models/*.json`).
- Datasets y archivos CSV (`data/*.csv`, `data/**/*.csv`).
- Excluye entornos virtuales (`.venv/`), cachés de python (`__pycache__/`) y archivos auxiliares de compilación TeX (`*.aux`, `*.log`, etc.).

---

## 📊 Resumen de Resultados Finales

- **Arquitectura Ganadora**: Tabular ResNet-MLP (3 Bloques Residuales, GELU, LayerNorm, Dropout 0.20)
- **Validación Cruzada**: 10-Fold Cross-Validation Ensemble
- **RMSE Out-Of-Fold ($)**: **$24,847.81**
- **Coeficiente de Determinación ($R^2$)**: **0.8965**
- **Error Relativo Porcentual (MAPE)**: **8.10%**
