# Código de Evaluación
############################################################################

import pandas as pd
import pickle
from sklearn.metrics import *
import os
import math
import numpy as np


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    print(filename, ' cargado correctamente')

    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')

    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['Item_Outlet_Sales'],axis=1)
    y_test = df['Item_Outlet_Sales']
    y_pred_test=model.predict(X_test)

    # Generamos métricas de diagnóstico
    print('Test score      : {}'.format(model.score(X_test, y_test)))

    rand_forest_mse = mean_squared_error(y_test , y_pred_test)
    rand_forest_rmse = math.sqrt(rand_forest_mse)
    rand_forest_r2 = r2_score(y_test, y_pred_test)
    rand_forest_mae = mean_absolute_error(y_test, y_pred_test)
    rand_forest_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    rand_forest_median_ae = median_absolute_error(y_test, y_pred_test)
    
    print('RandomForest MAPE \t       ----> {:.2f}%'.format(rand_forest_mape))      
    print('RandomForest MAE \t       ----> {}'.format(rand_forest_mae))
    print('RandomForest RMSE  \t       ----> {}'.format(rand_forest_rmse))
    print('RandomForest R2 Score       ----> {}'.format(rand_forest_r2))
    print('RandomForest Median AE \t       ----> {}'.format(rand_forest_median_ae))


# Validación desde el inicio
def main():
    df = eval_model('df_test.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()