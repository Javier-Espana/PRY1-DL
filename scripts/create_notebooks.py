import json
import os

def create_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

def make_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def make_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

# 1. EDA Notebook
eda_cells = [
    make_markdown_cell("# 1. Análisis Exploratorio de Datos (EDA)\n\n**Proyecto 1: Competencia de Modelación - Deep Learning (CC3092)**\n\nEste notebook presenta la fase completa de análisis exploratorio para el conjunto de datos de viviendas (Ames Housing Dataset). Se analizan distribuciones, variables numéricas y categóricas, valores nulos, atípicos, y se fundamentan las decisiones de preprocesamiento."),
    make_code_cell("import os\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='whitegrid')\n%matplotlib inline"),
    make_markdown_cell("## 1.1 Carga del Dataset y Dimensiones Iniciales"),
    make_code_cell("train_df = pd.read_csv('../data/train.csv')\nprint(f'Dimensiones del conjunto de entrenamiento: {train_df.shape[0]} filas x {train_df.shape[1]} columnas')\ntrain_df.head()"),
    make_markdown_cell("## 1.2 Inspección de Tipos de Variables"),
    make_code_cell("num_vars = train_df.select_dtypes(include=[np.number]).columns.tolist()\ncat_vars = train_df.select_dtypes(exclude=[np.number]).columns.tolist()\nprint(f'Número de variables numéricas: {len(num_vars)}')\nprint(f'Número de variables categóricas: {len(cat_vars)}')"),
    make_markdown_cell("## 1.3 Distribución de la Variable Objetivo (`SalePrice`)\n\nExaminamos la asimetría de la variable objetivo y la necesidad de aplicar una transformación logarítmica $\\log(1 + y)$."),
    make_code_cell("fig, axes = plt.subplots(1, 2, figsize=(14, 5))\nsns.histplot(train_df['SalePrice'], kde=True, ax=axes[0], color='#2563EB', bins=40)\naxes[0].set_title('Distribución de SalePrice (Sesgada hacia la derecha)', fontweight='bold')\naxes[0].set_xlabel('Precio de Venta ($)')\n\nsns.histplot(np.log1p(train_df['SalePrice']), kde=True, ax=axes[1], color='#10B981', bins=40)\naxes[1].set_title('Distribución de log1p(SalePrice) (Normalizada)', fontweight='bold')\naxes[1].set_xlabel('log1p(SalePrice)')\nplt.tight_layout()\nplt.show()"),
    make_markdown_cell("## 1.4 Análisis de Valores Nulos e Inconsistencias"),
    make_code_cell("null_counts = train_df.isnull().sum()\nnull_percent = (null_counts / len(train_df)) * 100\nnull_summary = pd.DataFrame({'Valores_Faltantes': null_counts, 'Porcentaje (%)': null_percent})\nnull_summary = null_summary[null_summary['Valores_Faltantes'] > 0].sort_values(by='Porcentaje (%)', ascending=False)\nprint('Resumen de columnas con valores faltantes:')\nnull_summary"),
    make_markdown_cell("## 1.5 Correlaciones con la Variable Objetivo"),
    make_code_cell("corr = train_df[num_vars].corr()\ntop_corr = corr['SalePrice'].abs().sort_values(ascending=False).head(12).index\n\nplt.figure(figsize=(10, 8))\nsns.heatmap(train_df[top_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)\nplt.title('Top 12 Variables Numéricas más Correlacionadas con SalePrice', fontweight='bold')\nplt.tight_layout()\nplt.show()"),
    make_markdown_cell("## 1.6 Conclusiones y Decisiones de Preprocesamiento Derivadas del EDA\n\n1. **Transformación de Variable Objetivo**: Aplicar `log1p(SalePrice)` para estabilizar la varianza del modelo MLP.\n2. **Tratamiento de Nulos Dominiales**: Imputar `NA` como categoría `'None'` en características donde representa la ausencia del elemento (Alley, Bsmt, Garage, Fireplace, Pool, Fence).\n3. **Imputación de LotFrontage**: Mediana segmentada por vecindario (`Neighborhood`).\n4. **Ingeniería de Características**: `TotalSF` (suma de sótano y pisos), `TotalBath`, edades de casa/garaje, términos de interacción de calidad.\n5. **Codificación Ordinal & Nominal**: Mapeo ordinal explícito para calificaciones (`Ex`, `Gd`, `TA`, `Fa`, `Po`) y One-Hot Encoding para nominales.")
]

create_notebook(eda_cells, "notebooks/01_eda.ipynb")

# 2. Experiments Notebook
experiments_cells = [
    make_markdown_cell("# 2. Metodología de Desarrollo y Experimentos de Modelación\n\n**Proyecto 1: Competencia de Modelación - Deep Learning (CC3092)**\n\nEste notebook presenta los resultados del benchmarking de arquitecturas MLP, la optimización de hiperparámetros y la evaluación del ensamble final de 10 pliegues (10-Fold CV)."),
    make_code_cell("import sys\nimport json\nsys.path.append('..')\nimport pandas as pd\nimport numpy as np\nfrom src.evaluate import evaluate_predictions"),
    make_markdown_cell("## 2.1 Carga de Resumen de Experimentos Guardados"),
    make_code_cell("with open('../saved_models/experiment_summary.json', 'r') as f:\n    summary = json.load(f)\n\nprint('=== RESULTADOS DE BENCHMARKING DE ARQUITECTURAS ===')\nfor arch, rmse in summary['benchmark_results'].items():\n    print(f'  {arch.upper():12s}: OOF RMSE = ${rmse:,.2f}')\nprint(f'\\nArquitectura Ganadora: {summary[\"best_architecture\"].upper()}')"),
    make_markdown_cell("## 2.2 Mejores Hiperparámetros Encontrados por Optuna"),
    make_code_cell("print('Mejores hiperparámetros (Optuna):')\nfor k, v in summary['best_hyperparameters'].items():\n    print(f'  {k:15s}: {v}')"),
    make_markdown_cell("## 2.3 Métricas de Evaluación Out-Of-Fold del Ensamble Final de 10-Fold CV"),
    make_code_cell("with open('../saved_models/model_config.json', 'r') as f:\n    config = json.load(f)\n\nprint('=== CONFIGURACIÓN Y DESEMPEÑO DEL MODELO FINAL ===')\nprint(f'  Arquitectura       : {config.get(\"architecture\").upper()}')\nprint(f'  Dimensiones Entrada: {config.get(\"input_dim\")}')\nprint(f'  OOF RMSE Final     : ${config.get(\"overall_oof_rmse\"):,.2f}')\nprint(f'  Promedio Pliegues  : ${config.get(\"mean_fold_rmse\"):,.2f}')")
]

create_notebook(experiments_cells, "notebooks/02_model_experiments.ipynb")

# 3. Submission Notebook
submission_cells = [
    make_markdown_cell("# 3. Evaluación de Predicciones y Pipeline de Prueba\n\n**Proyecto 1: Competencia de Modelación - Deep Learning (CC3092)**\n\nEste notebook prueba la inferencia automática sobre el dataset de prueba de muestra (`pipeline_test.csv`) para garantizar la generación correcta de predicciones en el formato exacto requerido (`expected_output.csv`)."),
    make_code_cell("import sys\nsys.path.append('..')\nimport pandas as pd\nfrom predict_competition import predict"),
    make_markdown_cell("## 3.1 Inferencia en Dataset de Muestra (`pipeline_test.csv`)"),
    make_code_cell("output_df = predict(\n    test_path='../data/pruebas/pipeline_test.csv',\n    output_path='../data/pruebas/output_test_notebook.csv',\n    saved_models_dir='../saved_models'\n)"),
    make_markdown_cell("## 3.2 Verificación del Formato contra `expected_output.csv`"),
    make_code_cell("expected_df = pd.read_csv('../data/pruebas/expected_output.csv')\nprint('Formato esperado:')\nprint(expected_df.head())\nprint('\\nFormato generado por el pipeline:')\nprint(output_df.head())\n\nassert list(output_df.columns) == list(expected_df.columns), 'Error: Las columnas no coinciden'\nassert len(output_df) == len(expected_df), 'Error: El número de filas no coincide'\nprint('\\n¡VERIFICACIÓN EXITOSA! El pipeline produce predicciones válidas en el formato exacto.')")
]

create_notebook(submission_cells, "notebooks/03_evaluation_and_submission.ipynb")

print("Notebooks successfully updated.")
