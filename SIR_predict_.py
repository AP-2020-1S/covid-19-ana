#%%
from SIR_model import SIR
from utilities.utilities import Utilities 
import pandas as pd
import numpy as np

#%%
sir_model = SIR()
ult = Utilities()

#%%
censo_df = pd.read_excel('data/ProyeccionMunicipios2005_2020.xls', sheet_name = 'Mpios',header=8)

censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Bogotá, D.C.', 'Bogotá D.C.', censo_df['MPIO'])
censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Cartagena', 'Cartagena de Indias', censo_df['MPIO'])

data = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')
#%%
data = ult.dates_fix(data)
data = ult.build_counters(data)
data = ult.clean_dataset(data)
cities = ult.get_cities(data)
dates = ult.get_dates(data)
mv = ult.build_mineable_view(data, cities, dates)


#%%
sir_model_df = sir_model.sir_tasas(mv)

#%%
def sir(df_covid, df_censo, city):
    # concat_df = pd.DataFrame()
    # cities = df_covid['Ciudad de ubicación'].unique()

    # for city in cities:
        # df = pd.DataFrame()
    df = df_covid[df_covid['Ciudad de ubicación'] == city]

    df['suceptible'] = sir_model.censo(df_censo,city)
    df['total_rec'] = 0
    df['muertos'] = 0
    df['contagio'] = 0
    df['sanos'] = 0 #Recuperados amarillo
    df['activo'] = df['Casos']
    df['confirmado'] = df['Casos']

    for index in range(1,len(df)):

        df['contagio'].iloc[index] = df.iloc[index]['tasa_trans'] * df.iloc[index-1]['activo'] * df.iloc[index-1]['suceptible'] / sir_model.censo(df_censo,city)
        # 0 if ( df.iloc[index]['suceptible'] > self.censo(df_censo,city)).any() \
        # else

        df['suceptible'].iloc[index] = df.iloc[index-1]['suceptible'] - df.iloc[index] ['contagio']
        
        df['confirmado'].iloc[index] = df.iloc[index-1]['confirmado'] + df.iloc[index] ['contagio']

        df['sanos'].iloc[index] = df.iloc[index-1]['activo'] * df.iloc[index]['tasa_recup']

        df['muertos'].iloc[index] = df.iloc[index-1]['activo'] * df.iloc[index]['tasa_muerte']

        df['activo'].iloc[index] = df.iloc[index-1]['activo'] + df.iloc[index-1]['contagio'] - df.iloc[index-1]['sanos'] - df.iloc[index-1]['muertos']

        df['total_rec'].iloc[index] = df.iloc[index-1]['total_rec']  + df.iloc[index]['sanos']  

        # dataframe_list = [concat_df, df]
        # concat_df = pd.concat(dataframe_list)
        
    return df
#%%
data_sir = sir(sir_model_df, censo_df, 'Medellín')
# %%
