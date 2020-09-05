#%%
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import dash_table
import base64
import pandas as pd
import numpy as np
from file_handle import File_Handle
from SIR_model import SIR
from SIR_predict import SirPredict
from utilities.utilities import Utilities 
from statsmodels.stats.outliers_influence import summary_table
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
# %%
#%%
#Instances
handle = File_Handle()
sirmodel = SIR()
sirpredict = SirPredict()
utl = Utilities()
fig = go.Figure()
# %%
#%%
#Loading data
censo_df = pd.read_excel('data/ProyeccionMunicipios2005_2020.xls', sheet_name = 'Mpios',header=8)

censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Bogotá, D.C.', 'Bogotá D.C.', censo_df['MPIO'])
censo_df['MPIO'] = np.where(censo_df['MPIO'] == 'Cartagena', 'Cartagena de Indias', censo_df['MPIO'])

data_org = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')

# %%
#%%
#Execution
#------------------------------------------------------------------
data = pd.DataFrame()
cities = ["Medellín"]
data = data_org[data_org["Ciudad de ubicación"].isin(cities)]
data = utl.dates_fix(data)
data = utl.build_counters(data)
data = utl.clean_dataset(data)
cities = utl.get_cities(data)
dates = utl.get_dates(data)
mv_med = utl.build_mineable_view(data, cities, dates)

tasas_med = sirmodel.sir_tasas_init(mv_med)
sir_formulas_med = sirmodel.sir_tasas(tasas_med, censo_df)
original_med, predict_med = sirpredict.predict(sir_formulas_med,censo_df)

# %%
#RMSE Medellín
# *********** Medición de RMSE ***********************************#
#Particion de la data 30 días de datos de test
top_day = sir_formulas_med.tail(1)['t'].values[0] - 30
sir_formulas_med_train = sir_formulas_med[:top_day]
# prediccion para calculo del error
original_med_test, predict_med_test = sirpredict.predict(sir_formulas_med_train,censo_df)
# 30 dias originales de test
original_med_err = original_med[top_day + 1:]
# Calculo de RMSE para muertos
y_muertes_original = original_med_err['muertos'].values
y_muertes_test = predict_med_test['muertos'].values
print("RMSE Medellín muertes -- > ", round(mean_squared_error(y_muertes_original, y_muertes_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para Muertes en la ciudad de Medellín
#consolidacion del DataFrame Orig y Pred
df_Pred_mes = pd.concat([original_med,predict_med])
data_muertes_mes = np.array(df_Pred_mes['muertos'].values)
stdev = np.sqrt(sum((y_muertes_test - y_muertes_original)**2) / (len(y_muertes_original)))
mu = data_muertes_mes.mean()
print("Intervalo de confianza para muestos en Medellin con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_activo_original = original_med_err['activo'].values
y_activo_test = predict_med_test['activo'].values
print("RMSE Medellín activos -- > ", round(mean_squared_error(y_activo_original, y_activo_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para Muertes en la ciudad de Medellín
#consolidacion del DataFrame Orig y Pred
data_activo_mes = np.array(df_Pred_mes['activo'].values)
stdev = np.sqrt(sum((y_activo_test - y_activo_original)**2) / (len(y_activo_original)))
mu = data_activo_mes.mean()
print("Intervalo de confianza para activos en Medellin con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_confirmado_original = original_med_err['confirmado'].values
y_confirmado_test = predict_med_test['confirmado'].values
print("RMSE Medellín confirmados -- > ", round(mean_squared_error(y_confirmado_original, y_confirmado_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para confirmados en la ciudad de Medellín
#consolidacion del DataFrame Orig y Pred
data_confirmado_mes = np.array(df_Pred_mes['confirmado'].values)
stdev = np.sqrt(sum((y_confirmado_test - y_confirmado_original)**2) / (len(y_confirmado_original) ))
mu = data_confirmado_mes.mean()
print("Intervalo de confianza para activos en Medellin con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_suceptible_original = original_med_err['suceptible'].values
y_suceptible_test = predict_med_test['suceptible'].values
print("RMSE Medellín suceptibles -- > ", round(mean_squared_error(y_suceptible_original, y_suceptible_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para suceptible en la ciudad de Medellín
#consolidacion del DataFrame Orig y Pred
data_suceptible_mes = np.array(df_Pred_mes['suceptible'].values)
stdev = np.sqrt(sum((y_suceptible_test - y_suceptible_original)**2) / (len(y_suceptible_original)))
mu = data_suceptible_mes.mean()
print("Intervalo de confianza para suceptibles en Medellin con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_contagio_original = original_med_err['contagio'].values
y_contagio_test = predict_med_test['contagio'].values
print("RMSE Medellín contagios -- > ", round(mean_squared_error(y_contagio_original, y_contagio_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para contagios en la ciudad de Medellín
#consolidacion del DataFrame Orig y Pred
data_suceptible_mes = np.array(df_Pred_mes['contagio'].values)
stdev = np.sqrt(sum((y_contagio_test - y_contagio_original)**2) / (len(y_contagio_original)))
mu = data_suceptible_mes.mean()
print("Intervalo de confianza para contagios en Medellin con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))

# ****************************************************************#

# %%
#RMSR Bogotá D.C.
#------------------------------------------------------------------
data = pd.DataFrame()
cities = ["Bogotá D.C."]
data = data_org[data_org["Ciudad de ubicación"].isin(cities)]
data = utl.dates_fix(data)
data = utl.build_counters(data)
data = utl.clean_dataset(data)
cities = utl.get_cities(data)
dates = utl.get_dates(data)
mv_bog = utl.build_mineable_view(data, cities, dates)

tasas_bog = sirmodel.sir_tasas_init(mv_bog)
sir_formulas_bog = sirmodel.sir_tasas(tasas_bog, censo_df)
original_bog, predict_bog = sirpredict.predict(sir_formulas_bog,censo_df)

# *********** Medición de RMSE ***********************************#
#Particion de la data 20 días de datos de test
top_day = sir_formulas_bog.tail(1)['t'].values[0] - 30
sir_formulas_bog_train = sir_formulas_bog[:top_day]
# prediccion para calculo del error
original_bog_test, predict_bog_test = sirpredict.predict(sir_formulas_bog_train,censo_df)
# 30 dias originales de test
original_bog_err = original_bog[top_day + 1:]
# Calculo de RMSE para muertos
y_muertes_original = original_bog_err['muertos'].values
y_muertes_test = predict_bog_test['muertos'].values
print("RMSE Bogotá D.C. muertes -- > ",round(mean_squared_error(y_muertes_original, y_muertes_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para Muertes en la ciudad de Bogotá D.C.
#consolidacion del DataFrame Orig y Pred
df_Pred_bog = pd.concat([original_bog,predict_bog])
stdev = np.sqrt(sum((y_muertes_test - y_muertes_original)**2) / (len(y_muertes_original)))
mu = np.array(df_Pred_bog['muertos'].values).mean()
print("Intervalo de confianza para muestos en Bogotá D.C. con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

y_activo_original = original_bog_err['activo'].values
y_activo_test = predict_bog_test['activo'].values
print("RMSE Bogotá D.C. activos -- > ", round(mean_squared_error(y_activo_original, y_activo_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para Activos en la ciudad de Bogotá D.C.
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_activo_test - y_activo_original)**2) / (len(y_activo_original)))
mu = np.array(df_Pred_bog['activo'].values).mean()
print("Intervalo de confianza para activos en Bogotá D.C. con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_confirmado_original = original_bog_err['confirmado'].values
y_confirmado_test = predict_bog_test['confirmado'].values
print( "RMSE Bogotá D.C. confirmado -- > ", round(mean_squared_error(y_confirmado_original, y_confirmado_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para confirmados en la ciudad de Bogotá D.C.
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_confirmado_test - y_confirmado_original)**2) / (len(y_confirmado_original)))
mu = np.array(df_Pred_bog['confirmado'].values).mean()
print("Intervalo de confianza para confirmados en Bogotá D.C. con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_suceptible_original = original_bog_err['suceptible'].values
y_suceptible_test = predict_bog_test['suceptible'].values
print( "RMSE Bogotá D.C. suceptible -- > ", round(mean_squared_error(y_suceptible_original, y_suceptible_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para suceptibles en la ciudad de Bogotá D.C.
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_suceptible_test - y_suceptible_original)**2) / (len(y_suceptible_original) ))
mu = np.array(df_Pred_bog['suceptible'].values).mean()
print("Intervalo de confianza para suceptibles en Bogotá D.C. con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_contagio_original = original_bog_err['contagio'].values
y_contagio_test = predict_bog_test['contagio'].values
print("RMSE Bogotá D.C. contagios -- > ", round(mean_squared_error(y_contagio_original, y_contagio_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para contagios en la ciudad de Bogota
#consolidacion del DataFrame Orig y Pred
data_suceptible_mes = np.array(df_Pred_mes['contagio'].values)
stdev = np.sqrt(sum((y_contagio_test - y_contagio_original)**2) / (len(y_contagio_original)))
mu = data_suceptible_mes.mean()
print("Intervalo de confianza para contagios en Bogotá D.C. con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))

# ****************************************************************#

# %%
#RMSE Cali
# ****************************************************************#

#------------------------------------------------------------------
data = pd.DataFrame()
cities = ["Cali"]
data = data_org[data_org["Ciudad de ubicación"].isin(cities)]
data = utl.dates_fix(data)
data = utl.build_counters(data)
data = utl.clean_dataset(data)
cities = utl.get_cities(data)
dates = utl.get_dates(data)
mv_cali = utl.build_mineable_view(data, cities, dates)

tasas_cali = sirmodel.sir_tasas_init(mv_cali)
sir_formulas_cali = sirmodel.sir_tasas(tasas_cali, censo_df)
original_cali, predict_cali = sirpredict.predict(sir_formulas_cali,censo_df)

# *********** Medición de RMSE ***********************************#
#Particion de la data 20 días de datos de test
top_day = sir_formulas_cali.tail(1)['t'].values[0] - 30
sir_formulas_cali_train = sir_formulas_cali[:top_day]
# prediccion para calculo del error
original_cali_test, predict_cali_test = sirpredict.predict(sir_formulas_cali_train,censo_df)
# 30 dias originales de test
original_cali_err = original_cali[top_day + 1:]
# Calculo de RMSE para muertos
y_muertes_original = original_cali_err['muertos'].values
y_muertes_test = predict_cali_test['muertos'].values
print( "RMSE Cali muertes -- > ",round(mean_squared_error(y_muertes_original, y_muertes_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para Muertes en la ciudad de Calí
#consolidacion del DataFrame Orig y Pred
df_Pred_cali = pd.concat([original_cali,predict_cali])
stdev = np.sqrt(sum((y_muertes_test - y_muertes_original)**2) / (len(y_muertes_original)))
mu = np.array(df_Pred_cali['muertos'].values).mean()
print("Intervalo de confianza para muestos en Calí con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-

y_activo_original = original_cali_err['activo'].values
y_activo_test = predict_cali_test['activo'].values
print("RMSE Cali activos -- > ",round(mean_squared_error(y_activo_original, y_activo_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para activos en la ciudad de Calí
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_activo_test - y_activo_original)**2) / (len(y_activo_original)))
mu = np.array(df_Pred_cali['activo'].values).mean()
print("Intervalo de confianza para activos en Calí con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_confirmado_original = original_cali_err['confirmado'].values
y_confirmado_test = predict_cali_test['confirmado'].values
print("RMSE Cali confirmados -- > ",round(mean_squared_error(y_confirmado_original, y_confirmado_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para confirmados en la ciudad de Calí
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_confirmado_test - y_confirmado_original)**2) / (len(y_confirmado_original)))
mu = np.array(df_Pred_cali['confirmado'].values).mean()
print("Intervalo de confianza para confirmados en Calí con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_suceptible_original = original_cali_err['suceptible'].values
y_suceptible_test = predict_cali_test['suceptible'].values
print("RMSE Cali suceptibles -- > ",round(mean_squared_error(y_suceptible_original, y_suceptible_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para suceptibles en la ciudad de Calí
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_suceptible_test - y_suceptible_original)**2) / (len(y_suceptible_original) ))
mu = np.array(df_Pred_cali['suceptible'].values).mean()
print("Intervalo de confianza para suceptibles en Calí con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_contagio_original = original_cali_err['contagio'].values
y_contagio_test = predict_cali_test['contagio'].values
print("RMSE Calí contagios -- > ", round(mean_squared_error(y_contagio_original, y_contagio_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para contagios en la ciudad de Calí
#consolidacion del DataFrame Orig y Pred
data_suceptible_mes = np.array(df_Pred_mes['contagio'].values)
stdev = np.sqrt(sum((y_contagio_test - y_contagio_original)**2) / (len(y_contagio_original)))
mu = data_suceptible_mes.mean()
print("Intervalo de confianza para contagios en Calí con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))

# ****************************************************************#

# %%
# RMSE Barranquilla
#------------------------------------------------------------------
data = pd.DataFrame()
cities = ["Barranquilla"]
data =data_org[data_org["Ciudad de ubicación"].isin(cities)]
data = utl.dates_fix(data)
data = utl.build_counters(data)
data = utl.clean_dataset(data)
cities = utl.get_cities(data)
dates = utl.get_dates(data)
mv_bar = utl.build_mineable_view(data, cities, dates)

tasas_bar = sirmodel.sir_tasas_init(mv_bar)
sir_formulas_bar = sirmodel.sir_tasas(tasas_bar, censo_df)
original_bar, predict_bar = sirpredict.predict(sir_formulas_bar,censo_df)


# *********** Medición de RMSE ***********************************#
#Particion de la data 20 días de datos de test
top_day = sir_formulas_bar.tail(1)['t'].values[0] - 30
sir_formulas_bar_train = sir_formulas_bar[:top_day]
# prediccion para calculo del error
original_bar_test, predict_bar_test = sirpredict.predict(sir_formulas_bar_train,censo_df)
# 30 dias originales de test
original_bar_err = original_bar[top_day + 1:]
# Calculo de RMSE para muertos
y_muertes_original = original_bar_err['muertos'].values
y_muertes_test = predict_bar_test['muertos'].values
print("RMSE Barranquilla muertes -- > ",round(mean_squared_error(y_muertes_original, y_muertes_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para Muertes en la ciudad de Barranquilla
#consolidacion del DataFrame Orig y Pred
df_Pred_bar = pd.concat([original_bar,predict_bar])
stdev = np.sqrt(sum((y_muertes_test - y_muertes_original)**2) / (len(y_muertes_original)))
mu = np.array(df_Pred_bar['muertos'].values).mean()
print("Intervalo de confianza para muestos en Barranquilla con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_activo_original = original_bar_err['activo'].values
y_activo_test = predict_bar_test['activo'].values
print("RMSE Barranquilla activos -- > ",round(mean_squared_error(y_activo_original, y_activo_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para activos en la ciudad de Barranquilla
#consolidacion del DataFrame Orig y Pred
df_Pred_bar = pd.concat([original_bar,predict_bar])
stdev = np.sqrt(sum((y_activo_test - y_activo_original)**2) / (len(y_activo_original)))
mu = np.array(df_Pred_bar['activo'].values).mean()
print("Intervalo de confianza para activos en Barranquilla con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_confirmado_original = original_bar_err['confirmado'].values
y_confirmado_test = predict_bar_test['confirmado'].values
print("RMSE Barranquilla conformados -- > ",round(mean_squared_error(y_confirmado_original, y_confirmado_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para confirmados en la ciudad de Barranquilla
#consolidacion del DataFrame Orig y Pred
df_Pred_bar = pd.concat([original_bar,predict_bar])
stdev = np.sqrt(sum((y_confirmado_test - y_confirmado_original)**2) / (len(y_confirmado_original) ))
mu = np.array(df_Pred_bar['confirmado'].values).mean()
print("Intervalo de confianza para confirmados en Barranquilla con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_suceptible_original = original_bar_err['suceptible'].values
y_suceptible_test = predict_bar_test['suceptible'].values
print("RMSE Barranquilla suceptibles -- > ",round(mean_squared_error(y_suceptible_original, y_suceptible_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para confirmados en la ciudad de Barranquilla
#consolidacion del DataFrame Orig y Pred
df_Pred_bar = pd.concat([original_bar,predict_bar])
stdev = np.sqrt(sum((y_suceptible_test - y_suceptible_original)**2) / (len(y_suceptible_original) ))
mu = np.array(df_Pred_bar['suceptible'].values).mean()
print("Intervalo de confianza para suceptibles en Barranquilla con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_contagio_original = original_bar_err['contagio'].values
y_contagio_test = predict_bar_test['contagio'].values
print("RMSE Barranquilla contagios -- > ", round(mean_squared_error(y_contagio_original, y_contagio_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para contagios en la ciudad de Barranquilla
#consolidacion del DataFrame Orig y Pred
data_suceptible_mes = np.array(df_Pred_mes['contagio'].values)
stdev = np.sqrt(sum((y_contagio_test - y_contagio_original)**2) / (len(y_contagio_original)))
mu = data_suceptible_mes.mean()
print("Intervalo de confianza para contagios en Barranquilla con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))

# ****************************************************************#

# %%
# RMSE Cartagena de Indias
# ****************************************************************#


#------------------------------------------------------------------
data = pd.DataFrame()
cities = ["Cartagena de Indias"]
data = data_org[data_org["Ciudad de ubicación"].isin(cities)]
data = utl.dates_fix(data)
data = utl.build_counters(data)
data = utl.clean_dataset(data)
cities = utl.get_cities(data)
dates = utl.get_dates(data)
mv_car = utl.build_mineable_view(data, cities, dates)

tasas_car = sirmodel.sir_tasas_init(mv_car)
sir_formulas_car = sirmodel.sir_tasas(tasas_car, censo_df)
original_car, predict_car = sirpredict.predict(sir_formulas_car,censo_df)

# ------------------------------------------------------------------

# *********** Medición de RMSE ***********************************#
#Particion de la data 20 días de datos de test
top_day = sir_formulas_car.tail(1)['t'].values[0] - 30
sir_formulas_car_train = sir_formulas_car[:top_day]
# prediccion para calculo del error
original_car_test, predict_car_test = sirpredict.predict(sir_formulas_car_train,censo_df)

# 30 dias originales de test
original_car_err = original_car[top_day + 1:]
# Calculo de RMSE para muertos
y_muertes_original = original_car_err['muertos'].values
y_muertes_test = predict_car_test['muertos'].values
print("RMSE Cartagena muertes -- > ",round(mean_squared_error(y_muertes_original, y_muertes_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para Muertes en la ciudad de Cartagena
#consolidacion del DataFrame Orig y Pred
df_Pred_car = pd.concat([original_car,predict_car])
stdev = np.sqrt(sum((y_muertes_test - y_muertes_original)**2) / (len(y_muertes_original)))
mu = np.array(df_Pred_car['muertos'].values).mean()
print("Intervalo de confianza para muestos en Cartagena con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_activo_original = original_car_err['activo'].values
y_activo_test = predict_car_test['activo'].values
print("RMSE Cartagena activos -- > ",round(mean_squared_error(y_activo_original, y_activo_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para activos en la ciudad de Cartagena
#consolidacion del DataFrame Orig y Pred
df_Pred_car = pd.concat([original_car,predict_car])
stdev = np.sqrt(sum((y_activo_test - y_activo_original)**2) / (len(y_activo_original)))
mu = np.array(df_Pred_car['activo'].values).mean()
print("Intervalo de confianza para activos en Cartagena con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_confirmado_original = original_car_err['confirmado'].values
y_confirmado_test = predict_car_test['confirmado'].values
print("RMSE Cartagena confirmados -- > ",round(mean_squared_error(y_confirmado_original, y_confirmado_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para confirmados en la ciudad de Cartagena
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_confirmado_test - y_confirmado_original)**2) / (len(y_confirmado_original)))
mu = np.array(df_Pred_car['confirmado'].values).mean()
print("Intervalo de confianza para confirmados en Cartagena con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_suceptible_original = original_car_err['suceptible'].values
y_suceptible_test = predict_car_test['suceptible'].values
print("RMSE Cartagena suceptibles -- > ",round(mean_squared_error(y_suceptible_original, y_suceptible_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para suceptibles en la ciudad de Cartagena
#consolidacion del DataFrame Orig y Pred
stdev = np.sqrt(sum((y_suceptible_test - y_suceptible_original)**2) / (len(y_suceptible_original)))
mu = np.array(df_Pred_car['suceptible'].values).mean()
print("Intervalo de confianza para suceptibles en Cartagena con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
y_contagio_original = original_car_err['contagio'].values
y_contagio_test = predict_car_test['contagio'].values
print("RMSE Cartagena contagios -- > ", round(mean_squared_error(y_contagio_original, y_contagio_test,squared=False),2))
#*-*-*-*-*-*-*-*-**-*-*-*-*-*-*-*-**-*-*-**--*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
# Calculo de los intervalos de confianza para contagios en la ciudad de Cartagena
#consolidacion del DataFrame Orig y Pred
data_suceptible_mes = np.array(df_Pred_mes['contagio'].values)
stdev = np.sqrt(sum((y_contagio_test - y_contagio_original)**2) / (len(y_contagio_original)))
mu = data_suceptible_mes.mean()
print("Intervalo de confianza para contagios en Cartagena con un 95% de conf --> ", ( round(mu - 1.96 * stdev,2),round(mu + 1.96 * stdev,2))," con mu --> ", round(mu,2), " y stdev --> ", round(stdev,2))

# ****************************************************************#

# ****************************************************************#


# %%
