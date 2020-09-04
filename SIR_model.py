#%%
from file_handle import File_Handle
from utilities.utilities import Utilities 
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#%%
# handle = File_Handle()
# handle.download_censo_file()
# handle.download_covid_file()

#%%
utl = Utilities()

# #%%
# censo_df = pd.read_excel('data/ProyeccionMunicipios2005_2020.xls', sheet_name = 'Mpios',header=8)

# censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Bogotá, D.C.', 'Bogotá D.C.', censo_df['MPIO'])
# censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Cartagena', 'Cartagena de Indias', censo_df['MPIO'])

# data = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')


# #%%
# cities = ["Medellín"]
# data = data[data["Ciudad de ubicación"].isin(cities)]
# data = utl.dates_fix(data)
# data = utl.build_counters(data)
# data = utl.clean_dataset(data)
# cities = utl.get_cities(data)
# dates = utl.get_dates(data)
# mv = utl.build_mineable_view(data, cities, dates)

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

#%%
# from SIR_model import SIR
# sir_model = SIR()
# tasas = sir_model.sir_tasas_init(mv)
# sir = sir_model.sir_tasas(tasas, censo_df)
# %%
# ax = plt.gca()
# bog.plot('t', 'total_rec', label='Recuperados_Total', ax=ax)
# bog.plot('t', 'activo', label='Activos', ax=ax)
# bog.plot('t', 'confirmado', label='Confirmados', ax=ax)
# bog.plot('t', 'muertos', label='Muertos', ax=ax)

# plt.show()  


# #%%
# tasas = sir_model.sir_tasas(mv, censo_df)
# tasas_bog = sir_model.sir_tasas(mv[mv['Ciudad de ubicación'] == 'Bogotá D.C.'], censo_df)

# #%%
# # tasa_trans_prom = tasas_bog[tasas_bog['t']>(len(tasas_bog)-15)]['tasa_trans'].mean()#tasas_bog['tasa_trans'].mean()
# # tasa_rec_prom = tasas_bog[tasas_bog['t']>(len(tasas_bog)-15)]['tasa_recup'].mean()#tasas_bog['tasa_recup'].mean()
# # tasa_muerte_prom = tasas_bog[tasas_bog['t']>(len(tasas_bog)-5)]['tasa_muerte'].mean()#tasas_bog['tasa_muerte'].mean()

# tasas_bog_ajustada = tasas_bog[['Casos','Recuperados','Muertos','Casos_Acum', 'Recuperados_Acum', 't', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible', 'tasa_trans', 'tasa_recup', 'tasa_muerte']].tail(15)

    
# #%%
# #Crear registros a predecir
# tamano = int(tasas_bog_ajustada['t'].tail(1))
# l_n = list(range(tamano + 1, tamano + 30))
# new_t = pd.DataFrame(columns=['Casos','Recuperados','Muertos','Casos_Acum', 'Recuperados_Acum', 't', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible', 'tasa_trans', 'tasa_recup', 'tasa_muerte'])
# new_t['t'] = l_n 
# new_t = pd.concat([tasas_bog_ajustada, new_t]).reset_index(drop=True)
# new_t['tasa_trans_prom'] = 0
# new_t['tasa_rec_prom'] = 0
# new_t['tasa_muerte_prom'] = 0

# # %%
# #Calculo contagios muertos

# for row in range(14,len(new_t)):

#     new_t['tasa_trans_prom'] = new_t['tasa_trans'].rolling(5).mean()
#     new_t['tasa_rec_prom'] = new_t['tasa_recup'].rolling(5).mean()
#     new_t['tasa_muerte_prom'] = new_t['tasa_muerte'].rolling(5).mean()

    # new_t['contagio'].iloc[row] = 0 if ( new_t.iloc[row]['suceptible'] > sir_model.censo(censo_df,'Bogotá D.C.')).any() \
    # else new_t.iloc[row-1]['tasa_trans_prom'] * new_t.iloc[row-1]['activo'] * (new_t.iloc[row-1]['suceptible'] / sir_model.censo(censo_df,'Bogotá D.C.'))

    # new_t['suceptible'].iloc[row] = new_t.iloc[row-1]['suceptible'] - new_t.iloc[row] ['contagio']
    
    # new_t['confirmado'].iloc[row] = new_t.iloc[row-1]['confirmado'] + new_t.iloc[row] ['contagio']

    # new_t['sanos'].iloc[row] = new_t.iloc[row-1]['activo'] * new_t.iloc[row-1]['tasa_rec_prom']

    # new_t['muertos'].iloc[row] = new_t.iloc[row-1]['activo'] * new_t.iloc[row-1]['tasa_muerte_prom']

    # new_t['activo'].iloc[row] = new_t.iloc[row-1]['activo'] + new_t.iloc[row-1]['contagio'] - new_t.iloc[row-1]['sanos'] - new_t.iloc[row-1]['muertos']

    # new_t['total_rec'].iloc[row] = new_t.iloc[row-1]['total_rec']  + new_t.iloc[row]['sanos']  

#     new_t['tasa_trans'].iloc[row] = new_t.iloc[row-1]['tasa_trans_prom']
#     new_t['tasa_recup'].iloc[row] = new_t.iloc[row-1]['tasa_rec_prom']
#     new_t['tasa_muerte'].iloc[row] = new_t.iloc[row-1]['tasa_muerte_prom']
# # %%

# originales = tasas_bog[['t', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible','tasa_trans', 'tasa_recup', 'tasa_muerte']]
# predichos = new_t[['t', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible','tasa_trans', 'tasa_recup', 'tasa_muerte']]
# predichos = new_t[new_t['t'] > tamano]
# # predichos.drop(0,inplace=True)
# prueba3 = pd.concat([originales, predichos]).reset_index(drop=True)




# ax = plt.gca()
# originales.plot('t', 'total_rec', label='Recuperados_Total', color = 'g',ax=ax)
# predichos.plot('t', 'total_rec', label='Recuperados_Total_Pred', color = 'r', ax=ax)
# originales.plot('t', 'activo', label='Activos', color = 'b',ax=ax)
# predichos.plot('t', 'activo', label='Activos_Pred', color = 'r',ax=ax)
# originales.plot('t', 'confirmado', label='Confirmado', color = 'y',ax=ax)
# predichos.plot('t', 'confirmado', label='Confirmado_Pred', color = 'r',ax=ax)
# originales.plot('t', 'muertos', label='Muertos', color = 'k', ax=ax)
# predichos.plot('t', 'muertos', label='Muertos_Pred', color = 'r',ax=ax)
# plt.show() 
# # %%
