#%%
from file_handle import File_Handle
from utilities.utilities import Utilities 
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%%
utl = Utilities()

#%%
class SIR():
    def __init__(self):
        pass

    def censo(self, df, city):
        n_poblacion = df[df['MPIO'] == city][2020].values[0]#.iloc[:,19]
        return n_poblacion
        # return pd.Series(n_poblacion).astype(int)

    def sir_tasas_init(self, df_covid):
        concat_df = pd.DataFrame()
        cities = df_covid['Ciudad de ubicación'].unique()
        for city in cities:

            df = pd.DataFrame()
            df = df_covid[df_covid['Ciudad de ubicación'] == city]

            df['t'] = range(0, len(df))
            df['tasa_trans'] = 0.05
            df['tasa_recup'] = 0.02
            df['tasa_muerte'] = 0.005
            df['activo'] = df['Casos']
            df['confirmado'] = df['Casos']
            df['Ciudad de ubicación'] = city

            for row in range(1,len(df)):
                df['tasa_trans'].iloc[row] = df.iloc[row]['Casos'] /  df.iloc[row-1]['Casos_Acum']
                df['tasa_recup'].iloc[row] = df.iloc[row]['Recuperados'] /  df.iloc[row-1]['Casos_Acum']
                df['tasa_muerte'].iloc[row] = df.iloc[row]['Muertos'] /  df.iloc[row-1]['Casos_Acum']
                df['tasa_trans'].replace([np.inf, -np.inf], np.nan, inplace = True)    
                df['tasa_trans'].fillna(0, inplace=True)
                df['tasa_recup'].replace([np.inf, -np.inf], np.nan, inplace = True)
                df['tasa_recup'].fillna(0, inplace=True)
                df['tasa_muerte'].replace([np.inf, -np.inf], np.nan, inplace = True)
                df['tasa_muerte'].fillna(0, inplace=True)
            
            dataframe_list = [concat_df, df]
            concat_df = pd.concat(dataframe_list)

        return concat_df
    
    def sir_tasas(self,df_covid, df_censo):
        concat_df = pd.DataFrame()
        cities = df_covid['Ciudad de ubicación'].unique()
        for city in cities:
            df = pd.DataFrame()
            df = df_covid[df_covid['Ciudad de ubicación'] == city]

            n_poblacion =  (self.censo(df_censo,city)) #pd.Series(self.censo(df_censo,city)).astype(int)
            df['suceptible'] = n_poblacion#self.censo(df_censo,city)
            df['total_rec'] = 0
            df['muertos'] = 0
            df['contagio'] = 0
            df['sanos'] = 0 #Recuperados amarillo
            df['activo'] = df['Casos']
            df['confirmado'] = df['Casos']
            df['Ciudad de ubicación'] = city

            for index in range(1,len(df)):

                df['contagio'].iloc[index] = 0 if ( df.iloc[index]['suceptible'] > n_poblacion).any() else df.iloc[index]['tasa_trans'] * df.iloc[index-1]['activo'] * (df.iloc[index-1]['suceptible'] / n_poblacion)

                df['suceptible'].iloc[index] = df.iloc[index-1]['suceptible'] - df.iloc[index] ['contagio']
                
                df['confirmado'].iloc[index] = df.iloc[index-1]['confirmado'] + df.iloc[index] ['contagio']

                df['sanos'].iloc[index] = df.iloc[index-1]['activo'] * df.iloc[index]['tasa_recup']

                df['muertos'].iloc[index] = df.iloc[index-1]['activo'] * df.iloc[index]['tasa_muerte']

                df['activo'].iloc[index] = df.iloc[index-1]['activo'] + df.iloc[index-1]['contagio'] - df.iloc[index-1]['sanos'] - df.iloc[index-1]['muertos']

                df['total_rec'].iloc[index] = df.iloc[index-1]['total_rec']  + df.iloc[index]['sanos']  

            dataframe_list = [concat_df, df]
            concat_df = pd.concat(dataframe_list)

        return concat_df