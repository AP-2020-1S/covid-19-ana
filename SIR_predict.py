#%%
from file_handle import File_Handle
from utilities.utilities import Utilities 
from SIR_model import SIR
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%%
utl = Utilities()
sir = SIR()

#%%
class SirPredict():
    def __init__(self):
        pass

    def predict(self, df_tasas, df_censo):
        tasas_ajustada = df_tasas[['Ciudad de ubicación','Casos','Recuperados','Muertos','Casos_Acum', 'Recuperados_Acum', 't', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible', 'tasa_trans', 'tasa_recup', 'tasa_muerte']].tail(15)

        concat_df = pd.DataFrame()
        cities = df_tasas['Ciudad de ubicación'].unique()
        for city in cities:

            #Crear registros a predecir
            tamano = int(tasas_ajustada['t'].tail(1))
            l_n = list(range(tamano + 1, tamano + 31))
            new_t = pd.DataFrame(columns=['Ciudad de ubicación','Casos','Recuperados','Muertos','Casos_Acum', 'Recuperados_Acum', 't', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible', 'tasa_trans', 'tasa_recup', 'tasa_muerte'])
            new_t['t'] = l_n 
            new_t = pd.concat([tasas_ajustada, new_t]).reset_index(drop=True)
            new_t['tasa_trans_prom'] = 0
            new_t['tasa_rec_prom'] = 0
            new_t['tasa_muerte_prom'] = 0
            new_t['Ciudad de ubicación'] = city
            n = (sir.censo(df_censo,city))

            for row in range(14,len(new_t)):

                new_t['tasa_trans_prom'] = new_t['tasa_trans'].rolling(5).mean()
                new_t['tasa_rec_prom'] = new_t['tasa_recup'].rolling(5).mean()
                new_t['tasa_muerte_prom'] = new_t['tasa_muerte'].rolling(5).mean()

                new_t['contagio'].iloc[row] = 0 if ( new_t.iloc[row]['suceptible'] > sir.censo(df_censo,city)).any() \
                else new_t.iloc[row-1]['tasa_trans_prom'] * new_t.iloc[row-1]['activo'] * (new_t.iloc[row-1]['suceptible'] ) / n #sir.censo(df_censo,city)

                new_t['suceptible'].iloc[row] = new_t.iloc[row-1]['suceptible'] - new_t.iloc[row] ['contagio']
                
                new_t['confirmado'].iloc[row] = new_t.iloc[row-1]['confirmado'] + new_t.iloc[row] ['contagio']

                new_t['sanos'].iloc[row] = new_t.iloc[row-1]['activo'] * new_t.iloc[row-1]['tasa_rec_prom']

                new_t['muertos'].iloc[row] = new_t.iloc[row-1]['activo'] * new_t.iloc[row-1]['tasa_muerte_prom']

                new_t['activo'].iloc[row] = new_t.iloc[row-1]['activo'] + new_t.iloc[row-1]['contagio'] - new_t.iloc[row-1]['sanos'] - new_t.iloc[row-1]['muertos']

                new_t['total_rec'].iloc[row] = new_t.iloc[row-1]['total_rec']  + new_t.iloc[row]['sanos']  

                new_t['tasa_trans'].iloc[row] = new_t.iloc[row-1]['tasa_trans_prom']
                new_t['tasa_recup'].iloc[row] = new_t.iloc[row-1]['tasa_rec_prom']
                new_t['tasa_muerte'].iloc[row] = new_t.iloc[row-1]['tasa_muerte_prom']

            original_df = df_tasas[['Ciudad de ubicación','t', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible','tasa_trans', 'tasa_recup', 'tasa_muerte']]
            predict_df = new_t[['Ciudad de ubicación','t', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible','tasa_trans', 'tasa_recup', 'tasa_muerte']]
            predict_df = new_t[new_t['t'] > tamano]
            # predichos.drop(0,inplace=True)
            # prueba3 = pd.concat([original_df, predichos]).reset_index(drop=True)
        return original_df, predict_df
