# Código de Entrenamiento
############################################################################

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))

    X_train = df.drop(['Item_Outlet_Sales'],axis=1)
    y_train = df['Item_Outlet_Sales']
    
    print(filename, ' cargado correctamente')
    
    # Entrenamos el modelo con toda la muestra
    rf_mod=RandomForestRegressor()
    rf_mod.fit(X_train, y_train)
    print('Modelo entrenado')

    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(rf_mod, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('df_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()