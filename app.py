import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output, State
import base64
from file_handle import File_Handle
import dash_table


handle = File_Handle()
file_status = handle.read_covid_file()
tabtitle = 'Covid-19'


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#---
data = pd.read_csv('data/Casos_positivos_de_COVID-19_en_Colombia.csv')
city_more_cases = ['Barranquilla', 'Bogotá D.C.', 'Cali', 'Cartagena de Indias', 'Medellín']
data = data[data['Ciudad de ubicación'].isin(city_more_cases)]


data.loc[(data['Fecha de muerte'].notnull() == True) | (data['Fecha recuperado'].notnull() == True), 'recuperado'] = 1 
data['recuperado'].fillna(0, inplace = True)

data.loc[~(data['Fecha de muerte'].notnull() == True) | ~(data['Fecha recuperado'].notnull() == True), 'infectado'] = 1 
data['infectado'].fillna(0, inplace = True) 

covid_cases = data.groupby(['Ciudad de ubicación','fecha reporte web']).agg({'infectado':sum,'recuperado':sum}).reset_index()
covid_cases['fecha reporte web'] = pd.to_datetime(covid_cases['fecha reporte web'],format="%Y-%m-%d")


covid_lastcases_med = covid_cases[(covid_cases['Ciudad de ubicación']=='Medellín') \
                    & (covid_cases['fecha reporte web'] == covid_cases['fecha reporte web'].max())] 

last_update = covid_cases['fecha reporte web'].max()

#----
app = dash.Dash(__name__,external_stylesheets=external_stylesheets) 
server = app.server
app.title=tabtitle

app.layout = html.Div(children=[
    
    html.Div([
        html.Div([
            html.Img(id='logo',
                    src='data:image/png;base64,{}'.format(base64.b64encode(open('img/logo.png', 'rb')\
                        .read()).decode()),
                    style={"height": '60%', "width": '60%', "margin-bottom": "25px"}
                    )],className="one-third column",),
        html.Div([
            html.Div([
            html.H1(children='Covid-19 en Colombia - Predicción',
            style = { "margin-bottom": "0px", 'font-size': '50px' }
            ,),
            html.Label(['Comportamiento del coronavirus en las principales ciudades de Colombia.\
            Información obtenida de la página web ', html.A('https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data', href='https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data')
            ,' recopilada por el Instituto Nacional de Salud'],
            style ={
                'font-size': '15px', 'text-align': 'justify'
            }),
            html.Label(['Fecha de última carga, ', last_update])
            ])],id='title')    
        ],id = 'header'),

    dash_table.DataTable(
            id = 'table',
            columns = [{'name':i, 'id':i} for i in covid_lastcases_med.columns if file_status == 'sucess'],
            data=covid_lastcases_med.to_dict('records')
        ),

    html.A(html.Button('Recarga de información'),href='/',style={
        'margin-left': '0px',
        }),

    html.Div([
        html.Div([
            html.P('Seleccione las ciudades a analizar: ',style={'font-weight': 'bold'}),
        dcc.Checklist(
            id='ciudades_checklist',
            options = [{'label': str(city), 'value': str(city)} for city in covid_cases['Ciudad de ubicación'].unique()],
            value = ['Medellín'])],
            style = {
            'width': '300px',#'calc(100%-40px)',
            'border-radius': '5px',
            'background-color': '#FFFFFF',
            'margin': '10px',
            'padding': '15px',
            'position': 'relative',
            'box-shadow': '2px 2px 2px lightgrey'
            })
            # labelStyle={'display': 'inline-block'}    
        ],id='filter'),

    html.Div([
        html.Div([dcc.Graph(id='g1')],
        style = {
        'border-radius': '5px',
        'background-color': '#FFFFFF',
        'margin': '10px',
        'padding': '15px',
        'position': 'relative',
        'box-shadow': '2px 2px 2px lightgrey'},        
        className="six columns"),

        html.Div([
            dcc.Graph(id='g2', figure={'data': [{'y': [1, 2, 3]}]})
        ], 
        style = {
        'border-radius': '5px',
        'background-color': '#FFFFFF',
        'margin': '10px',
        'padding': '15px',
        'position': 'relative',
        'box-shadow': '2px 2px 2px lightgrey'},
        className="six columns"),
    ], style = {
        'width': '100%',
        'border-radius': '5px',
        'padding': '15px',
        
        }, className="row"),

    dcc.Graph(
        id='scatter_infected',
        style = {
        'width': 'calc(100%-40px)',
        'border-radius': '5px',
        'background-color': '#FFFFFF',
        'margin': '10px',
        'padding': '15px',
        'position': 'relative',
        'box-shadow': '2px 2px 2px lightgrey'}
    ),

    html.Div([
        html.Div([
            html.P('Elaborado por:'),    
        
            html.Ul(children=[
            html.Li('Arboleda Santiago', style={'float':'left'}), 
            html.Li('Montoya Olga Lucía', style={'float':'left'}),
            html.Li('Ramirez Alberto'  , style={'float':'left'}),
            html.Li('Tangarife Juan David', style={'float':'left'})])
            ], style = {'fontSize': '12px'})
        ],style = {
            # 'width': 'calc(100%-40px)',
            'border-radius': '5px',
            'background-color': '#FFFFFF', #'#f9f9f9',
            'margin': '10px',
            'padding': '15px',
            'position': 'relative',
            'box-shadow': '2px 2px 2px lightgrey'
        })
],style = {
        'width': 'calc(100%-40px)',
        'border-radius': '5px',
        'background-color': '#f9f9f9',
        'margin': '10px',
        'padding': '15px',
        'position': 'relative',
        'box-shadow': '2px 2px 2px lightgrey'})

@app.callback(Output('scatter_infected', 'figure'), [Input('ciudades_checklist', 'value')])
def update_figure(selected_city):
    if file_status == 'sucess':
        filtered_df = covid_cases[covid_cases['Ciudad de ubicación'].isin(selected_city)]
        fig = px.scatter(filtered_df, x="fecha reporte web", y="infectado",
                        size="recuperado", color="Ciudad de ubicación",
                        size_max=35, title='Infectados vs Recuperados a lo largo del tiempo')
        fig.layout.plot_bgcolor = '#FFFFFF'##DCDCDC'
        # fig.layout.paper_bgcolor = '#FFFFFF'#'#fff'
    return fig

@app.callback(Output('g1', 'figure'), [Input('ciudades_checklist', 'value')])
def update_figure(selected_city):
    if file_status == 'sucess':
        filtered_df = covid_cases[covid_cases['Ciudad de ubicación'].isin(selected_city)]
        fig = px.scatter(filtered_df, x="fecha reporte web", y="infectado",
                        size="recuperado", color="Ciudad de ubicación",
                        size_max=35, title='Infectados vs Recuperados a lo largo del tiempo')
        fig.layout.plot_bgcolor = '#FFFFFF'
    return fig

@app.callback(Output('g2', 'figure'), [Input('ciudades_checklist', 'value')])
def update_figure(selected_city):
    if file_status == 'sucess':
        filtered_df = covid_cases[covid_cases['Ciudad de ubicación'].isin(selected_city)]
        fig = px.pie(filtered_df, names="Ciudad de ubicación", title= 'Participación por Ciudad')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
