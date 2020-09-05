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

#%%
censo_df = pd.read_excel('data/ProyeccionMunicipios2005_2020.xls', sheet_name = 'Mpios',header=8)

censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Bogotá, D.C.', 'Bogotá D.C.', censo_df['MPIO'])
censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Cartagena', 'Cartagena de Indias', censo_df['MPIO'])

data = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')

#%%
cities = ["Cartagena de Indias"]
data = data[data["Ciudad de ubicación"].isin(cities)]
data = utl.dates_fix(data)
data = utl.build_counters(data)
data = utl.clean_dataset(data)
cities = utl.get_cities(data)
dates = utl.get_dates(data)
mv = utl.build_mineable_view(data, cities, dates)


# %%

#Población año 2020
N_Med = int(censo_df[censo_df['MPIO'] == 'Medellín'].iloc[:,19])
N_Cal = int(censo_df[censo_df['MPIO'] == 'Cali'].iloc[:,19])
N_Bog = int(censo_df[censo_df['MPIO'] == 'Bogotá D.C.'].iloc[:,19])
N_Car = int(censo_df[censo_df['MPIO'] == 'Cartagena de Indias'].iloc[:,19])
N_Bar = int(censo_df[censo_df['MPIO'] == 'Barranquilla'].iloc[:,19])


#%%
#Cálculo de tasas   
mv['tasa_trans'] = 0.05
mv['tasa_recup'] = 0.02
mv['tasa_muerte'] = 0.005
mv['activo'] = mv['Casos']

for row in range(1,len(mv)):
    mv['tasa_trans'].iloc[row] = mv.iloc[row]['Casos'] /  mv.iloc[row-1]['Casos_Acum']
    mv['tasa_recup'].iloc[row] = mv.iloc[row]['Recuperados'] /  mv.iloc[row-1]['Casos_Acum']
    mv['tasa_muerte'].iloc[row] = mv.iloc[row]['Muertos'] /  mv.iloc[row-1]['Casos_Acum']

mv['tasa_trans'].replace([np.inf, -np.inf], np.nan, inplace = True)    
mv['tasa_trans'].fillna(0, inplace=True)
mv['tasa_recup'].replace([np.inf, -np.inf], np.nan, inplace = True)
mv['tasa_recup'].fillna(0, inplace=True)
mv['tasa_muerte'].replace([np.inf, -np.inf], np.nan, inplace = True)
mv['tasa_muerte'].fillna(0, inplace=True)

#%%
#Nuevos campos
mv['contagio'] = 0
mv['sanos'] = 0 #Recuperados amarillo
mv['suceptible'] = N_Car
mv['confirmado'] = mv['Casos']
mv['total_rec'] = 0
mv['muertos'] = 0

for row in range(1,len(mv)):
    mv['contagio'].iloc[row] = 0 if (mv.iloc[row]['suceptible'] > N_Car) \
        else mv.iloc[row]['tasa_trans'] * mv.iloc[row-1]['activo'] * (mv.iloc[row-1]['suceptible'] / N_Car)
    
    mv['suceptible'].iloc[row] = mv.iloc[row-1] ['suceptible'] - mv.iloc[row] ['contagio']
    
    mv['confirmado'].iloc[row] = mv.iloc[row-1] ['confirmado'] + mv.iloc[row] ['contagio']

    mv['sanos'].iloc[row] = mv.iloc[row-1]['activo'] * mv.iloc[row]['tasa_recup']

    mv['muertos'].iloc[row] = mv.iloc[row-1]['activo'] * mv.iloc[row]['tasa_muerte']

    mv['activo'].iloc[row] = mv.iloc[row-1]['activo'] + mv.iloc[row-1]['contagio'] - mv.iloc[row-1]['sanos'] - mv.iloc[row-1]['muertos']

    mv['total_rec'].iloc[row] = mv.iloc[row-1]['total_rec']  + mv.iloc[row]['sanos']  

#%%
#Crear tiempo en numeros
mv['t'] = range(0, len(mv))

# %%
tasas_ajustadas = mv[['Casos','Recuperados','Muertos','Casos_Acum', 'Recuperados_Acum', 't', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible', 'tasa_trans', 'tasa_recup', 'tasa_muerte']].tail(15)
#%%
#Crear registros a predecir
tamano = int(tasas_ajustadas['t'].tail(1))
l_n = list(range(tamano + 1, tamano + 30))
new_t = pd.DataFrame(columns=['Casos','Recuperados','Muertos','Casos_Acum', 'Recuperados_Acum', 't', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible', 'tasa_trans', 'tasa_recup', 'tasa_muerte'])
new_t['t'] = l_n 
new_t = pd.concat([tasas_ajustadas, new_t]).reset_index(drop=True)
new_t['tasa_trans_prom'] = 0
new_t['tasa_rec_prom'] = 0
new_t['tasa_muerte_prom'] = 0

# %%
#Calculo contagios muertos

for row in range(14,len(new_t)):

    new_t['tasa_trans_prom'] = new_t['tasa_trans'].rolling(5).mean()
    new_t['tasa_rec_prom'] = new_t['tasa_recup'].rolling(5).mean()
    new_t['tasa_muerte_prom'] = new_t['tasa_muerte'].rolling(5).mean()

    new_t['contagio'].iloc[row] = 0 if ( new_t.iloc[row]['suceptible'] > N_Car).any() \
    else new_t.iloc[row-1]['tasa_trans_prom'] * new_t.iloc[row-1]['activo'] * (new_t.iloc[row-1]['suceptible'] / N_Car)

    new_t['suceptible'].iloc[row] = new_t.iloc[row-1]['suceptible'] - new_t.iloc[row] ['contagio']
    
    new_t['confirmado'].iloc[row] = new_t.iloc[row-1]['confirmado'] + new_t.iloc[row] ['contagio']

    new_t['sanos'].iloc[row] = new_t.iloc[row-1]['activo'] * new_t.iloc[row-1]['tasa_rec_prom']

    new_t['muertos'].iloc[row] = new_t.iloc[row-1]['activo'] * new_t.iloc[row-1]['tasa_muerte_prom']

    new_t['activo'].iloc[row] = new_t.iloc[row-1]['activo'] + new_t.iloc[row-1]['contagio'] - new_t.iloc[row-1]['sanos'] - new_t.iloc[row-1]['muertos']

    new_t['total_rec'].iloc[row] = new_t.iloc[row-1]['total_rec']  + new_t.iloc[row]['sanos']  

    new_t['tasa_trans'].iloc[row] = new_t.iloc[row-1]['tasa_trans_prom']
    new_t['tasa_recup'].iloc[row] = new_t.iloc[row-1]['tasa_rec_prom']
    new_t['tasa_muerte'].iloc[row] = new_t.iloc[row-1]['tasa_muerte_prom']
# %%

originales = mv[['t', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible','tasa_trans', 'tasa_recup', 'tasa_muerte']]
predichos = new_t[['t', 'total_rec','muertos','contagio','sanos','activo', 'confirmado', 'suceptible','tasa_trans', 'tasa_recup', 'tasa_muerte']]
predichos = new_t[new_t['t'] > tamano]
# predichos.drop(0,inplace=True)
prueba3 = pd.concat([originales, predichos]).reset_index(drop=True)



#%%
ax = plt.gca()
originales.plot('t', 'total_rec', label='Recuperados_Total', color = 'g',ax=ax)
predichos.plot('t', 'total_rec', label='Recuperados_Total_Pred', color = 'r', ax=ax)
originales.plot('t', 'activo', label='Activos', color = 'b',ax=ax)
predichos.plot('t', 'activo', label='Activos_Pred', color = 'r',ax=ax)
originales.plot('t', 'confirmado', label='Confirmado', color = 'y',ax=ax)
predichos.plot('t', 'confirmado', label='Confirmado_Pred', color = 'r',ax=ax)
originales.plot('t', 'muertos', label='Muertos', color = 'k', ax=ax)
predichos.plot('t', 'muertos', label='Muertos_Pred', color = 'r',ax=ax)
plt.show() 
# %%
