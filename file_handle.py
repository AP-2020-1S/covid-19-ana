import requests
import urllib3
import pandas as pd
urllib3.disable_warnings()

class File_Handle:

    def download_covid_file(self):
        URL = 'https://www.datos.gov.co/api/views/gt2j-8ykr/rows.csv?accessType=DOWNLOAD'
        FILE_NAME = 'data/Casos_positivos_de_COVID-19_en_Colombia.csv'

        request = requests.get(URL)

        with open(FILE_NAME, 'wb') as output:
            file = output.write(request.content)
        output.close()

        return 'sucess'

    def download_censo_file(self):
        URL = 'https://www.dane.gov.co/files/investigaciones/poblacion/proyepobla06_20/ProyeccionMunicipios2005_2020.xls'
        FILE_NAME = 'data/ProyeccionMunicipios2005_2020.xls'

        request = requests.get(URL, verify = False)

        with open(FILE_NAME, 'wb') as output:
            file = output.write(request.content)
        output.close()

        return 'sucess'

#     def df_covid_file(self, path):
#         return 
#         censo_df = pd.read_excel('data/ProyeccionMunicipios2005_2020.xls', sheet_name = 'Mpios',header=8)
# data = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')
