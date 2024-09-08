# Ejecución de Tests Funcionales

### Alumno:
Nombre: Juan Francisco Rosales Kam
Correo: juanfcorosaleskam@hotmail.es

### Contexto:

Este proyecto de Github ha sido basado en la adaptación de un notebook encontrado en un proyecto de Kaggle hacia una estructura de carpetas y scripts similar al revisado en las sesión 3.

Dataset de Kaggle: https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction/data?select=Submission.csv
Proyecto adaptado: https://www.kaggle.com/code/ananysharma/big-mart-sales-prediction

Observación: Se escogió este proyecto porque fue fácil de entender en relación a los pasos que llego para proponer sus soluciones desde el EDA, Preprocesamiento y entrenamiento de distintos modelos "básicos" sin tanto tuneo de hiperparámeteros; por lo que adaptación hacia la estructura de carpetas y scripts fuera manejable. 

Es por ello que no fue prioritario los scores finales que se llegaron a publicar, así sean altos o bajos para haber sido considerados en un pase a producción final.
Se PRIORIZO la adaptación entendiendo que es un proyecto de un Data Scientist que ya fue aprobado de pase a producción y como MLE es momento de disponibilizar y pasar a producción la solución.

### Paso 0: Ingrese al Escritorio remoto

### Paso 1: Fork del Repositorio Original

En el navegador, inicie sesión en Github. Luego, vaya al enlace del proyecto original (https://github.com/lcajachahua/model-credit) y dé click al botón "Fork". Esto copiará todo el proyecto en su usuario de Github.


### Paso 2: Levantar el contenedor de Python

```
docker run -it --rm -p 8888:8888 jupyter/pyspark-notebook
```


### Paso 3: Configurar git

Abra una Terminal en JupyterLab e ingrese los siguientes comandos

```
git config --global user.name "<USER>"
git config --global user.email <CORREO>
```


### Paso 4: Clonar el Proyecto desde su propio Github

```
git clone https://github.com/<USER>/model-credit.git
```


### Paso 5: Instalar los pre-requisitos

```
cd model-credit/

pip install -r requirements.txt
```


### Paso 6: Ejecutar las pruebas en el entorno

```
cd src

python make_dataset.py

python train.py

python evaluate.py

python predict.py

cd ..
```


### Paso 7: Guardar los cambios en el Repo

```
git add .

git commit -m "Pruebas Finalizadas"

git push

```

Ingrese su usuario y Personal Access Token de Github. Puede revisar que los cambios se hayan guardado en el repositorio. Luego, puede finalizar JupyterLab ("File" => "Shut Down").
