#%%
from file_handle import File_Handle
from utilities.utilities import Utilities 
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%%
handle = File_Handle()
handle.download_censo_file()
handle.download_covid_file()

utl = Utilities()

#%%
censo_df = pd.read_excel('data/ProyeccionMunicipios2005_2020.xls', sheet_name = 'Mpios',header=8)

censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Bogotá, D.C.', 'Bogotá D.C.', censo_df['MPIO'])
censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Cartagena', 'Cartagena de Indias', censo_df['MPIO'])

data = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')


#%%
# cities = ["Medellín"]
# data = data[data["Ciudad de ubicación"].isin(cities)]
data = utl.dates_fix(data)
data = utl.build_counters(data)
data = utl.clean_dataset(data)
cities = utl.get_cities(data)
dates = utl.get_dates(data)
mv = utl.build_mineable_view(data, cities, dates)

#%%
class SIR():
    def __init__(self):
        pass

    def censo(self, df, city):
        n_poblacion = df[df['MPIO'] == city].iloc[:,19]
        return int(n_poblacion)
        # return pd.Series(n_poblacion).astype(int)

    def sir_tasas(self, df_covid, df_censo):
        concat_df = pd.DataFrame()
        for city in df_covid['Ciudad de ubicación'].unique():
            df = df_covid[df_covid['Ciudad de ubicación'] == city]
            df['t'] = range(0, len(df))
            df['tasa_trans'] = 0.05
            df['tasa_recup'] = 0.02
            df['tasa_muerte'] = 0.005
            df['total_rec'] = 0
            df['muertos'] = 0
            df['contagio'] = 0
            df['sanos'] = 0 #Recuperados amarillo
            df['activo'] = df['Casos']
            df['confirmado'] = df['Casos']
            df['suceptible'] = self.censo(df_censo,city)


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

            for row in range(1,len(df)):
                df['contagio'].iloc[row] = 0 if ( df.iloc[row]['suceptible'] > self.censo(df_censo,city)).any() \
                else df.iloc[row]['tasa_trans'] * df.iloc[row-1]['activo'] * (df.iloc[row-1]['suceptible'] / self.censo(df_censo,city))

                df['suceptible'].iloc[row] = df.iloc[row-1]['suceptible'] - df.iloc[row] ['contagio']
                
                df['confirmado'].iloc[row] = df.iloc[row-1]['confirmado'] + df.iloc[row] ['contagio']

                df['sanos'].iloc[row] = df.iloc[row-1]['activo'] * df.iloc[row]['tasa_recup']

                df['muertos'].iloc[row] = df.iloc[row-1]['activo'] * df.iloc[row]['tasa_muerte']

                df['activo'].iloc[row] = df.iloc[row-1]['activo'] + df.iloc[row-1]['contagio'] - df.iloc[row-1]['sanos'] - df.iloc[row-1]['muertos']

                df['total_rec'].iloc[row] = df.iloc[row-1]['total_rec']  + df.iloc[row]['sanos']  

            dataframe_list = [concat_df, df]
            concat_df = pd.concat(dataframe_list)
            
        return concat_df

# sir_model = SIR()
# tasas = sir_model.sir_tasas(mv, censo_df)
# %%
