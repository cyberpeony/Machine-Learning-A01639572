# PARTE 1 (From scratch) - Regresión Lineal usando "Wine Quality"

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

### Novedad entrega 2: Ajuste de hiperparámetros (validación cruzada)
Para fortalecer mi entrega en esta ocasión, opté por implementar validación cruzada con k-fold y grid search de hiperparámetros (learning_rate y l2_penalty). Esto me permite comparar resultados más eficientemente vs. el framework.

**Ejemplo 1: cross-validation con 5 kfolds**  
El siguiente comando imprime en consola el promedio de métricas +- desviación estandar de R2, RMSE y MAE:
`python src/regLinealWine.py train --csv data/winequality-red.csv --target quality  --model-path model.json --num_iters 6000 --cv 5`

**Ejemplo 2: grid search en hiperparámetros**  
Para probar varias combinaciones de learning rate y l2 penalty, eligiendo la mejor combinación y mostrando los resultados de la crossvalidation. El mejor modelo (mejores hiperparámetros se entrena en todos los datos y se guarda en model.json):
`python src/regLinealWine.py train --csv data/winequality-red.csv --target quality  --model-path model.json --num_iters 6000 --cv 5 \  --grid_lr 0.01 0.05 0.1 --grid_l2 0.0 0.01 0.1`

### Notas a considerar
- El modelo se implementó desde 0, sin scikit-learn ni otros frameworks de ML 
- Solo usa `numpy` y `pandas` para manipulación numérica/de datos
- Se calcula R2,  MSE, RMSE y MAE

# PARTE 2 (Usando un framework) - Regresión Lineal con scikit-learn usando "Wine Quality"
En esta segunda entrega, volví a implementar una regresión lineal, pero con ayuda de un framework (scikit-learn). En cuanto a los datos, utilicé el mismo dataset Wine Quality (winequality-red.csv) con la columna quality como variable objetivo.

### Requisitos adicionales
Además de numpy y pandas, esta parte requiere:

- scikit-learn>=1.3,<2
- joblib>=1.3

Actualiza tu entorno con:
`pip install -r requirements.txt`

### Training
Los siguientes comandos entrenan el modelo, muestran métricas en consola y guardan el modelo en model_sklearn.pkl:

**Ejemplo 1: Training simple con LinearRegression**
`python src/regLinealWine_sklearn.py train --csv data/winequality-red.csv --target quality --model-path model_sklearn.pkl --modelo linear --test-size 0.2 --seed 42`

**Ejemplo 2: Training simple con Ridge**
`python src/regLinealWine_sklearn.py train --csv data/winequality-red.csv --target quality --model-path model_sklearn.pkl --modelo ridge --alpha 0.1 --test-size 0.2 --seed 42`

### Crossvalidation y grid (ridge)
Para hacer crossvalidation de 5 kfolds y probando valores de alpha para ridge, mostrando la media +- la desv. estándar del R2 en c/caso + entrenar al modelo con el mejor valor encontrado usando todos los datos y guardandolo en el archivo de salida:

**Ejemplo 3: Crossvalidation (5 folds) + grid search de alpha**
`python src/regLinealWine_sklearn.py train --csv data/winequality-red.csv --target quality --model-path model_sklearn.pkl --modelo ridge --cv 5 --grid_alpha 0.0 0.01 0.1 1.0 --seed 42`

### Predicción
Ya entrenado, para hacer predicciones:  
`python src/regLinealWine_sklearn.py predict --csv data/winequality-red.csv --model-path model_sklearn.pkl --out-csv predicciones_sklearn.csv`

Ese comando imprime en consola las primeras predicciones y guarda un archivo `predicciones_sklearn.csv` con la columna `prediction`. 

### Resultados esperados

**En training/test simple:**

Linear y Ridge obtuvieron R2 aprox = 0.40 en test, mejor que el baseline (aprox = −0.006)

**En crossvalidation (5-fold ridge):**

Mejor alpha = 1.0, con R2 promedio = 0.34 +- 0.06

### Notas a considerar
- Se usó Pipeline para encadenar estandarización y modelo (StandardScaler y LinearRegression/Ridge)
- Se calcula R2, MSE, RMSE, MAE y un baseline con la media
- Esta implementación complementa la parte desde cero, permitiendo comparar resultados y confirmar el correcto funcionamiento del modelo