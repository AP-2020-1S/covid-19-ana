#%%
from file_handle import File_Handle
from utilities.utilities import Utilities 
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#%%
def dates_fix(df):
    # Convert dates to datetime
    df['Fecha de muerte'] = pd.to_datetime(df['Fecha de muerte'])
    df['Fecha diagnostico'] = pd.to_datetime(df['Fecha diagnostico'])
    df['FIS'] = pd.to_datetime(df['FIS'], errors="coerce") #This field contains a record with wrong values
    df['fecha reporte web'] = pd.to_datetime(df['fecha reporte web'])
    df['Fecha de notificación'] = pd.to_datetime(df['Fecha de notificación'])
    df['Fecha recuperado'] = pd.to_datetime(df['Fecha recuperado'])

    # Assign dates of asymptomatic
    df['FIS'].fillna(df['fecha reporte web'], inplace=True)

    return df

# Function to build counters
def build_counters(df):
    # Build counters fields
    df["Casos"] = 1
    df.loc[(df['Fecha de muerte'].notnull() == True), 'Muertos'] = 1 
    df['Muertos'].fillna(0, inplace = True)
    df.loc[(df['Fecha recuperado'].notnull() == True), 'Recuperados'] = 1
    df['Recuperados'].fillna(0, inplace = True)
    df.loc[(~df['Fecha de muerte'].notnull() == True) & (~df['Fecha recuperado'].notnull() == True), 'Activos'] = 1 
    df['Activos'].fillna(0, inplace = True)

    return df

# Clean dataset
def clean_dataset(df):
    # Delete records 
    df.drop(df[(~df['Fecha de muerte'].notnull() == False) & (~df['Fecha recuperado'].notnull() == False) & (df["atención"].notnull()==False)].index, inplace=True)

    # Clean death counter
    mask = (~df['Fecha de muerte'].notnull() == False) & (~df['Fecha recuperado'].notnull() == False) & (df["atención"]=="Recuperado")
    df.loc[mask, 'Muertos'] = 0
    
    # Clean death dates
    df.loc[mask, 'Fecha de muerte'] = np.NaN

    return df

# Get cities 
def get_cities(df):
    indixes = df.groupby(["Ciudad de ubicación"]).agg({'Casos': 'sum', 'Recuperados': 'sum', 'Activos': 'sum', 'Muertos': 'sum'}).sort_values(ascending=False, by="Casos").head().index
    cities = []
    for i in indixes:
        cities.append(i)
    
    return cities

def get_dates(df):
    fechas = []
    fecha_maxima_muerte = df['Fecha de muerte'].max()
    fecha_minima_muerte = df['Fecha de muerte'].min()
    fecha_maxima_recuperado = df['Fecha recuperado'].max()
    fecha_minima_recuperado = df['Fecha recuperado'].min()
    fecha_maxima_fis = df['FIS'].max()
    fecha_minima_fis = df['FIS'].min()
    #max_date 
    if ((fecha_maxima_fis > fecha_maxima_recuperado) | (fecha_maxima_fis > fecha_maxima_muerte)): max_date = fecha_maxima_fis
    elif ((fecha_maxima_recuperado > fecha_maxima_fis) | (fecha_maxima_recuperado > fecha_maxima_muerte)): max_date = fecha_maxima_recuperado
    else: max_date = fecha_maxima_muerte

    #min_date_med 
    if ((fecha_minima_fis  < fecha_minima_recuperado) | (fecha_minima_fis < fecha_minima_muerte)): min_date = fecha_minima_fis
    elif ((fecha_minima_recuperado < fecha_minima_fis) | (fecha_minima_recuperado < fecha_maxima_muerte)): min_date = fecha_minima_recuperado
    else: min_date = fecha_minima_muerte
    fechas.append(min_date)
    fechas.append(max_date)
    
    return fechas    

def build_mineable_view(df, cities, dates):
    mv_final = pd.DataFrame()
    for i in cities:
        min_date = dates[0]
        max_date = dates[1]
        mv_ = pd.date_range(start=min_date, end=max_date)
        muertos = pd.DataFrame()
        mv = pd.DataFrame()
        mv['fecha'] = mv_
        mv['fecha'] = mv['fecha'].dt.date

        # Casos
        df_casos = df[df["Ciudad de ubicación"] == i].groupby(["FIS"]).agg({'Casos': 'sum'}).sort_values(ascending=True, by="FIS").reset_index()
        df_casos['FIS'] = df_casos['FIS'].dt.date

        # Muertos
        muertos = df
        muertos['Fecha de muerte'] = muertos['Fecha de muerte'].dt.date
        muertos = muertos[muertos['Ciudad de ubicación'] == i].groupby(["Fecha de muerte"]).agg({'Muertos': 'sum'}).reset_index()
        
        # Recuperados
        recuperados = df
        recuperados['Fecha recuperado'] = recuperados['Fecha recuperado'].dt.date
        recuperados = recuperados[recuperados['Ciudad de ubicación'] == i].groupby(["Fecha recuperado"]).agg({'Recuperados': 'sum'}).reset_index()
        
        # Merge
        mv=mv.merge(df_casos, how='left', left_on = "fecha", right_on="FIS")
        mv = mv.groupby(["fecha"]).agg({'Casos': 'sum'}).reset_index()
        mv=mv.merge(recuperados, how='left', left_on = "fecha", right_on="Fecha recuperado")
        mv = mv.groupby(["fecha"]).agg({'Casos': 'sum', 'Recuperados': 'sum'}).reset_index()
        mv=mv.merge(muertos, how='left', left_on = "fecha", right_on="Fecha de muerte")
        mv = mv.groupby(["fecha"]).agg({'Casos': 'sum', 'Recuperados': 'sum', 'Muertos': 'sum'}).reset_index()
        #mv=mv.merge(df_acumulados, how='left', left_on = "fecha", right_on="FIS")

        mv['Casos_Acum'] = mv['Casos'].cumsum()
        mv['Recuperados_Acum'] = mv['Recuperados'].cumsum()
        mv['Muertos_Acum'] = mv['Muertos'].cumsum()

        # Filling couters missing values
        mv["Muertos"].fillna(0, inplace = True)
        mv["Recuperados"].fillna(0, inplace = True)
        mv["Casos"].fillna(0, inplace = True)
        mv["Casos_Acum"].fillna(0, inplace = True)
        mv["Recuperados_Acum"].fillna(0, inplace = True)
        #mv["Activos_Acum"].fillna(0, inplace = True)
        mv["Muertos_Acum"].fillna(0, inplace = True)
        
        # Filling city missing
        mv['Ciudad de ubicación'] = i
        
        dfs = [mv_final,mv]
        mv_final = pd.concat(dfs)
    
    return mv_final










#%%
# handle = File_Handle()
# handle.read_censo_file()
# handle.read_covid_file()

minable_view = Utilities()

#%%
data = pd.read_csv("data/Casos_positivos_de_COVID-19_en_Colombia.csv", sep=",")
#%%
cities = ["Medellín"]
data = data[data["Ciudad de ubicación"].isin(cities)]
data = dates_fix(data)
data = build_counters(data)
data = clean_dataset(data)
cities = get_cities(data)
dates = get_dates(data)
mv = build_mineable_view(data, cities, dates)

#%%
#Dataframes
censo_df = pd.read_excel('data/ProyeccionMunicipios2005_2020.xls', sheet_name = 'Mpios',header=8)
# data = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')


# %%

#Población año 2020
N_Med = int(censo_df[censo_df['MPIO'] == 'Medellín'].iloc[:,19])
N_Cal = int(censo_df[censo_df['MPIO'] == 'Cali'].iloc[:,19])
N_Bog = int(censo_df[censo_df['MPIO'] == 'Bogotá, D.C.'].iloc[:,19])
N_Car = int(censo_df[censo_df['MPIO'] == 'Cartagena'].iloc[:,19])
N_Bar = int(censo_df[censo_df['MPIO'] == 'Barranquilla'].iloc[:,19])

#%%
# #Estados iniciales
# data.loc[(data['Fecha de muerte'].notnull() == True) | (data['Fecha recuperado'].notnull() == True), 'recuperado'] = 1 
# data['recuperado'].fillna(0, inplace = True)

# data.loc[~(data['Fecha de muerte'].notnull() == True) & ~(data['Fecha recuperado'].notnull() == True), 'infectado'] = 1 
# data['infectado'].fillna(0, inplace = True)


#%%
#Para una ciudad
# covid_cases = data.groupby(['Ciudad de ubicación','fecha reporte web']).agg({'infectado':sum,'recuperado':sum}).reset_index()
# covid_cases['fecha reporte web'] = pd.to_datetime(covid_cases['fecha reporte web'],format="%Y-%m-%d")

# mv = covid_cases[covid_cases['Ciudad de ubicación'] == 'Medellín'].sort_values('fecha reporte web')

# #%%
# #Acumulados
# mv['infectado_acum'] = mv['infectado'].cumsum()
# mv['recuperado_acum'] = mv['recuperado'].cumsum()
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
mv['suceptible'] = N_Med
mv['confirmado'] = mv['Casos']
mv['total_rec'] = 0
mv['muertos'] = 0

for row in range(1,len(mv)):
    mv['contagio'].iloc[row] = 0 if (mv.iloc[row]['suceptible'] > N_Med) \
        else mv.iloc[row]['tasa_trans'] * mv.iloc[row-1]['activo'] * (mv.iloc[row-1]['suceptible'] / N_Med)
    
    mv['suceptible'].iloc[row] = mv.iloc[row-1] ['suceptible'] - mv.iloc[row] ['contagio']
    
    mv['confirmado'].iloc[row] = mv.iloc[row-1] ['confirmado'] + mv.iloc[row] ['contagio']

    mv['sanos'].iloc[row] = mv.iloc[row-1]['activo'] * mv.iloc[row]['tasa_recup']

    mv['muertos'].iloc[row] = mv.iloc[row-1]['activo'] * mv.iloc[row]['tasa_muerte']

    mv['activo'].iloc[row] = mv.iloc[row-1]['activo'] + mv.iloc[row-1]['contagio'] - mv.iloc[row-1]['sanos'] - mv.iloc[row-1]['muertos']

    mv['total_rec'].iloc[row] = mv.iloc[row-1]['total_rec']  + mv.iloc[row]['sanos']  

#%%
#Crear tiempo en numeros
mv['t'] = range(0, len(mv))


#%%



#---------

#%%
#Cambio de tasas
range_date = range(5, len(mv), 5)

for day in range_date:
    lista = list(range_date)
    trans_rate = float(mv[mv['t']==day]['tasa_trans'])
    recup_rate = float(mv[mv['t']==day]['tasa_recup'])
    muerto_rate = float(mv[mv['t']==day]['tasa_muerte'])
    if trans_rate == 0:
        trans_rate = float(mv[mv['t']==day-1]['tasa_trans'])
    if recup_rate == 0:
        recup_rate = float(mv[mv['t']==day-1]['tasa_recup'])
    if muerto_rate == 0:
        muerto_rate = float(mv[mv['t']==day-1]['tasa_muerte'])
    
    
    indice_day = range_date.index(day)
    if indice_day == 0:
        indice_day = 0 
    else:
        indice_day =lista[indice_day-1]
    if indice_day == 0:
        mv.loc[(mv['t'] >= indice_day) & (mv['t'] <= day), 'tasa_trans_new'] = trans_rate
        mv.loc[(mv['t'] >= indice_day) & (mv['t'] <= day), 'tasa_recup_new'] = recup_rate
        mv.loc[(mv['t'] >= indice_day) & (mv['t'] <= day), 'tasa_muerte_new'] = muerto_rate
    else:
        mv.loc[(mv['t'] > indice_day) & (mv['t'] <= day), 'tasa_trans_new'] = trans_rate
        mv.loc[(mv['t'] > indice_day) & (mv['t'] <= day), 'tasa_recup_new'] = recup_rate
        mv.loc[(mv['t'] > indice_day) & (mv['t'] <= day), 'tasa_muerte_new'] = muerto_rate

#%%
#Nuevos campos
mv['contagio'] = 0
mv['sanos'] = 0 #Recuperados amarillo
mv['suceptible'] = N_Med
mv['confirmado'] = mv['Casos']
mv['total_rec'] = 0
mv['muertos'] = 0

for row in range(1,len(mv)):
    mv['contagio'].iloc[row] = 0 if (mv.iloc[row]['suceptible'] > N_Med) \
        else mv.iloc[row]['tasa_trans_new'] * mv.iloc[row-1]['activo'] * (mv.iloc[row-1]['suceptible'] / N_Med)
    
    mv['suceptible'].iloc[row] = mv.iloc[row-1] ['suceptible'] - mv.iloc[row] ['contagio']
    
    mv['confirmado'].iloc[row] = mv.iloc[row-1] ['confirmado'] + mv.iloc[row] ['contagio']

    mv['sanos'].iloc[row] = mv.iloc[row-1]['activo'] * mv.iloc[row]['tasa_recup_new']

    mv['muertos'].iloc[row] = mv.iloc[row-1]['activo'] * mv.iloc[row]['tasa_muerte_new']

    mv['activo'].iloc[row] = mv.iloc[row-1]['activo'] + mv.iloc[row-1]['contagio'] - mv.iloc[row-1]['sanos'] - mv.iloc[row-1]['muertos'] 

    if mv.iloc[row]['activo'] <= 0:
        mv['activo'].iloc[row] = mv['activo'].iloc[row-1]

    mv['total_rec'].iloc[row] = mv.iloc[row-1]['total_rec']  + mv.iloc[row]['sanos']  

#%%
mv.plot('t', 'activo', label='Activos')
mv.plot('t', 'suceptible', label='Suceptibles')
mv.plot('t', 'confirmado', label='Confirmados')
mv.plot('t', 'muertos', label='Muertos')

#%%
mv.plot('t', 'suceptible', label='Suceptibles')

#%%

print(f"Inicio reporte del virus:\t {mv['Fecha de notificación'].min()}")
print(f"Último día reporado del virus':\t {mv['Fecha de notificación'].max()}")
print(f"Total de días con casos:\t {mv['Fecha de notificación'].max() - mv['Fecha de notificación'].min()}")

