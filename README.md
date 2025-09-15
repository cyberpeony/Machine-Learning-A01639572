# Regresión Lineal usando "Wine Quality"

Este proyecto consiste en la implementación de un algoritmo sin frameworks de ML (solo `numpy` y `pandas`). El algoritmo que elegí es **regresión lineal**, el cual entrené "from scratch" con gradiente descendente.

Para los datos, utilicé el dataset **Wine Quality**, donde el objetivo predecir la calidad del vino de 3-8. La columna se llama `quality`.

## Instrucciones

### Requisitos
Instalar dependencias (recomiendo un entorno virtual). `requirements.txt` incluye:

- numpy==1.26.4  
- pandas>=2,<3  

Instala con:  
`pip install -r requirements.txt`

### Dataset
Coloca el archivo `winequality-red.csv` en la carpeta `data/`. El dataset está en [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009/data). Nota-> el dataset tiene la columna `quality` como variable objetivo.

### Training
Para entrenar el modelo ejecuta:  
`python src/regLinealWine.py train --csv data/winequality-red.csv --target quality --model-path model.json --learning_rate 0.05 --num_iters 6000 --l2_penalty 0.0`

Ese comando hace el training del modelo, despliega métricas en consola y guarda los parámetros en un json (model.json).

### Predicción
Ya entrenado, para hacer predicciones:  
`python src/regLinealWine.py predict --csv data/winequality-red.csv --model-path model.json --out-csv predicciones.csv`

Ese comando imprime en consola las primeras predicciones y guarda un archivo `predicciones.csv` con la columna `prediction`. 

## Novedad entrega 2: Ajuste de hiperparámetros (validación cruzada)
Para fortalecer mi entrega en esta ocasión, opté por implementar validación cruzada con k-fold y grid search de hiperparámetros (learning_rate y l2_penalty). Esto me permite comparar resultados más eficientemente vs. el framework.

**Ejemplo 1: cross-validation con 5 kfolds**  
El siguiente comando imprime en consola el promedio de métricas +- desviación estandar de R2, RMSE y MAE:
`python src/regLinealWine.py train --csv data/winequality-red.csv --target quality \  --model-path model.json --num_iters 6000 --cv 5`

**Ejemplo 2: grid search en hiperparámetros**  
Para probar varias combinaciones de learning rate y l2 penalty, eligiendo la mejor combinación y mostrando los resultados de la crossvalidation. El mejor modelo (mejores hiperparámetros se entrena en todos los datos y se guarda en model.json):
`python src/regLinealWine.py train --csv data/winequality-red.csv --target quality \ --model-path model.json --num_iters 6000 --cv 5 \  --grid_lr 0.01 0.05 0.1 --grid_l2 0.0 0.01 0.1`

## Notas a considerar
- El modelo se implementó desde 0, sin scikit-learn ni otros frameworks de ML 
- Solo usa `numpy` y `pandas` para manipulación numérica/de datos
- Se calcula R2,  MSE, RMSE y MAE
