import wget, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from datetime import date, timedelta
import datetime as dt


class Utilities:
    fileName = "params.dat"
   
    def __init__(self):
        pass
    
    def delete_directoty_content(self, data_path):
        for f in os.listdir(data_path):
            os.remove(data_path + f)

    def download(self, url, path, file):
        wget.download(url, path + file)

    def get_params(self, fileName):
        params = {}
        for line in open(fileName):
            key_value = line.strip().split("|")
            if not line.startswith("#"): params[key_value[0].strip()] = key_value[1].strip()
        return params
    
    # Function to fix dates
    def dates_fix(self,df):
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
    def build_counters(self,df):
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
    def clean_dataset(self, df):
        # Delete records 
        df.drop(df[(~df['Fecha de muerte'].notnull() == False) & (~df['Fecha recuperado'].notnull() == False) & (df["atención"].notnull()==False)].index, inplace=True)

        # Clean death counter
        mask = (~df['Fecha de muerte'].notnull() == False) & (~df['Fecha recuperado'].notnull() == False) & (df["atención"]=="Recuperado")
        df.loc[mask, 'Muertos'] = 0
        
        # Clean death dates
        df.loc[mask, 'Fecha de muerte'] = np.NaN

        return df

    # Get cities 
    def get_cities(self, df):
        indixes = df.groupby(["Ciudad de ubicación"]).agg({'Casos': 'sum', 'Recuperados': 'sum', 'Activos': 'sum', 'Muertos': 'sum'}).sort_values(ascending=False, by="Casos").head().index
        cities = []
        for i in indixes:
            cities.append(i)
        
        return cities

    def get_dates(self, df):
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

    def build_mineable_view(self, df, cities, dates):
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
            
            muertos = muertos[muertos['Ciudad de ubicación'] == i].groupby(["Fecha de muerte"]).agg({'Muertos': 'sum'}).reset_index()
            muertos['Fecha de muerte'] = muertos['Fecha de muerte'].dt.date
            
            # Recuperados
            recuperados = df
            
            recuperados = recuperados[recuperados['Ciudad de ubicación'] == i].groupby(["Fecha recuperado"]).agg({'Recuperados': 'sum'}).reset_index()
            recuperados['Fecha recuperado'] = recuperados['Fecha recuperado'].dt.date

            # Activos
            #activos = df
            #activos['FIS'] = activos['FIS'].dt.date
            #activos = activos[activos['Ciudad de ubicación'] == i].groupby(["FIS"]).agg({'Activos': 'sum'}).reset_index()
            
            # Merge
            mv=mv.merge(df_casos, how='left', left_on = "fecha", right_on="FIS")
            mv = mv.groupby(["fecha"]).agg({'Casos': 'sum'}).reset_index()
            mv=mv.merge(recuperados, how='left', left_on = "fecha", right_on="Fecha recuperado")
            mv = mv.groupby(["fecha"]).agg({'Casos': 'sum', 'Recuperados': 'sum'}).reset_index()
            mv=mv.merge(muertos, how='left', left_on = "fecha", right_on="Fecha de muerte")
            mv = mv.groupby(["fecha"]).agg({'Casos': 'sum', 'Recuperados': 'sum', 'Muertos': 'sum'}).reset_index()
            #mv=mv.merge(activos, how='left', left_on = "fecha", right_on="FIS")
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
