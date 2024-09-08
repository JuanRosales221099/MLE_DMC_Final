# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df

def detect_outliers(df, feature):
        Q1  = df[feature].quantile(0.25)
        Q3  = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        return upper_limit, lower_limit

# Realizamos la transformación de datos
def data_preparation(df, flag_test):

    # Missing Value Treatment
    df['Outlet_Size'] = df.Outlet_Size.fillna(df.Outlet_Size.dropna().mode()[0])
    df['Item_Weight'] = df.Item_Weight.fillna(df.Item_Weight.mean())

    # Outliers
    if(flag_test != True):

        upper, lower = detect_outliers(df, "Item_Visibility")
        df = df[(df['Item_Visibility'] > lower) & (df['Item_Visibility'] < upper)]

        upper, lower = detect_outliers(df, "Item_Outlet_Sales")
        df = df[(df['Item_Outlet_Sales'] > lower) & (df['Item_Outlet_Sales'] < upper)]

    # Let's correct the errors in the Item_Fat_Content column

    df['Item_Fat_Content'] = df['Item_Fat_Content'].map({'Low Fat' :'Low Fat',
                                                           'low fat' :"Low Fat",
                                                           'LF'      :"Low Fat",
                                                           'Regular' :'Regular',
                                                           'reg'     :"Regular"
                                                          })
    

    # getting the amount of established years in new column and delete old column
    df['Outlet_Age'] = 2023 - df['Outlet_Establishment_Year']
    del df['Outlet_Establishment_Year']


    # Encoding Categorical Variables

    # 1. Label Encoding

    df['Outlet_Size'] = df['Outlet_Size'].map({'Small'  : 1,
                                                 'Medium' : 2,
                                                 'High'   : 3
                                                 }).astype(int)
    

    df['Outlet_Location_Type'] = df['Outlet_Location_Type'].str[-1:].astype(int)

    df['Item_Identifier_Categories'] = df['Item_Identifier'].str[0:2]


    encoder = LabelEncoder()
    ordinal_features = ['Item_Fat_Content', 'Outlet_Type', 'Outlet_Location_Type']

    for feature in ordinal_features:
        df[feature] = encoder.fit_transform(df[feature])

    #2. One Hot Encoding

    df = pd.get_dummies(df, columns=['Item_Type', 'Item_Identifier_Categories', 'Outlet_Identifier'], drop_first=True)

    # Let's drop useless columns    
    df.drop(labels=['Item_Identifier'], axis=1, inplace=True)
    
    return df 

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación
def main():

    df_data = read_file_csv("train.csv")
    df_scoring = read_file_csv("test.csv")

    df_data_train_test = data_preparation(df_data, False)
    df_scoring = data_preparation(df_scoring, True)

    df_train, df_val = train_test_split(df_data_train_test, test_size=0.2, random_state=0)


    #Matriz de entrenamiento
    data_exporting(df_train, df_train.columns, "df_train.csv")
    #Matriz de Validación
    data_exporting(df_val, df_val.columns, "df_test.csv")
    #Matriz de entrenamiento
    data_exporting(df_scoring, df_scoring.columns, "df_scoring.csv")

if __name__ == "__main__":
    main()

