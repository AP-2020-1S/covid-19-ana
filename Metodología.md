# Metodología usada en predicción de Covid-19 en Colombia

Para el desarrollo del proyecto de predicción de Covid-19 en las principales ciudades de Colombia se utilizó la metodología CRISP-DM (Cross-Industry Standard Process for Data Mining), la cual proporciona una descripción normalizada del ciclo de vida de un proyecto estándar de análisis de datos. Esta metodología contiene seis fases: 

1. Entendimiento del negocio
2. Entendimiento de los datos
3. Preparación de datos
4. Modelamiento
5. Evaluación
6. Desarrollo

La secuencia de las fases no es estricta. De hecho, la mayoría de los proyectos avanzan y retroceden entre fases en caso de ser necesario. 

<p align = "center" >
  <img src="https://miro.medium.com/max/494/1*VH9RlYPSjL6YOtBtHxunqQ.png" alt="CRISP-DM" width="300px" height="300px"/>
</p>

# Entendimiento del negocio
En esta fase se busca conocer la necesidad del negocio, evaluar la situación actual y determinar el objetivo a resolver con el proyecto de minería de datos.

Coronavirus es el nombre que recibe una amplia familia de virus que pueden infectar al ser humano causando una variedad de enfermedades severas. Un miembro de la familia de coronavirus es el SARS-CoV-2 o mejor conocido como Covid-19, el cual se transmite y circula ampliamente entre la comunidad.

La enfermedad del Covid-19 fue reportada el 31 de diciembre de 2019 en la ciudad de Wuhan en China, esta enfermedad se exparció rápidamente por el mundo al punto que la organización mundial de la salud (OMS), declaró el 4 de mayo de 2020, el Covid-19 como pandemia, luego de que se registraran 3.581.884 confirmados al rededor del mundo, dejando a esa fecha 248.558 muertes.

Colombia no ha sido ajena a la pandemia. El primer caso positivo se registró el 6 de marzo de 2020 y el gobierno nacional el 24 de marzo declara ailamiento estricto para evitar la rápida propagación y así poder mitigar el impacto que ocasionaría la enfermedad en la red hospitalaria. 

### Objetivo del negocio

El objetivo del negocio es conocer el comportamiento del Covid-19 en las 5 principales ciudades de Colombia, a partir de los estados por los que pasa una persona desde que se reporta como infectada.


### Objetivo de minería de datos

El objetivo de este proyecto es la predicción de corto y mediano plazo del total de casos confirmados, nuevos casos, casos activos, recuperados y muertes para las 5 principales ciudades de Colombia, utilizando técnicas estadísticas, de inteligencia artificial o modelos híbridos.

#### Requerimientos

  * El producto de datos propuesto debe descargar la última información dispobile de forma automática cada vez que se ejecute.

  * La limpieza de datos, recalibración de los modelos, etc, deben ejecutarse automaticamente sin intervención humana. Es decir, el usuario ejecuta unicamente la rutina principal y todo el sistema se actualiza automaticamente.

  * Debe presentar los resultados en un dashboard que use GitHub pages (recuerde que puede utilizar jinja2, HTML, etc)

  * Debe presentar intervalos de confianza para los pronósticos.

  * Debe presentar métricas de calibración para la información histórica.

# Entendimiento de los datos

En esta fase se explora las fuentes de datos a utilizar para el desarrollo del proyecto, se describen los datos, se exploran y se verifica la calidad de los mismos.

## Fuentes de datos

#### Covid-19

El comportamiento del Covid-19 en las principales ciudades de Colombia esta siendo recolectado y centralizado por el Instituto Nacional de Salud. Esta información esta siendo puesta a disposición del público general, a través de la página web [www.datos.gov.co](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data)

##### Diccionario de datos

|**Campo**                             |**Tipo de dato** |
|---------                             | --------------- |
|ID de caso.                           | Texto           | 
|Fecha de notificación	                | Texto           |
|Código DIVIPOLA                       | Texto           |
|Ciudad de ubicación                   | Texto           |
|Departamento o Distrito               | Texto           |
|atención                              | Texto           |
|Edad                                  | Texto           |
|Sexo                                  | Texto           |
|Tipo                                  | Texto           |
|Estado                                | Texto           |
|País de procedencia                   | Texto           |
|Fecha de inicio de síntomas (FIS)     | Texto           |
|Fecha de muerte                       | Texto           |
|Fecha diagnostico                     | Texto           |
|Fecha de confirmación por laboratorio | Texto           |
|Fecha recuperado                      | Texto           |
|Fecha reporte web                     | Texto           |
|Fecha de publicación en sitio web     | Texto           |
|Tipo recuperación                     | Texto           |
|Codigo departamento                   | Número          |
|Codigo pais                           | Número          |
|Pertenencia etnica                    | Texto           |
|Nombre grupo etnico                   | Texto           |

#### Datos Poblacionales - Dane

Para la información poblacional se utilizó la información del censo proyectado por el DANE para el año 2020, el cual se puede encontrar en la página web [Proyecciones y retroproyecciones de población](https://www.dane.gov.co/index.php/estadisticas-por-tema/demografia-y-poblacion/proyecciones-de-poblacion)

## Análisis exploratorio

Se realizó un análisis exploratorio de los datos y el detalle se puede encontrar en el siguiente vínculo:
[Análisis descriptivo](docs/Análisis_descriptivo_Covid.ipynb)

### Análisis gráfico a corte de 30 de Agosto de 2020

#### Comparación entre las 5 ciudades con mas casos
<p align = "center" >
  <img src="img/pieCiudades.JPG" alt="alt" height="300px"  width="300px"/>
</p>

#### Estados de la infección por ciudad
<p align = "center" >
  <img src="img/CiudadesEstados.JPG" alt="alt" height="300px"/>
</p>

#### Evolución en el tiempo
![Alt text](img/CiudadesTiempo.JPG)

#### Tabla comparativa
<p align = "center" >
  <img src="img/compara.JPG" alt="alt" height="300px"/>
</p>

Nota: La columna ERR cuenta los registros que originalmente tenian fecha de recuperado y fecha de muerte simultaneamente.

### Hallazgos

1. La columna de atención en la cual se registran diferentes estados se encontró que hay problemas de calidad como registros que tienen fecha de recuperado o de muerto y no su correspondiente estado, o valores nulos por lo cual se tomó la decisión de utilizar las fechas de recuperado y de muerte para realizar los respectivos conteos.
2. Se evidencia que el campo de fecha de inicio de síntomas se está utilizando para marcar los casos de personas asintomáticas por lo cual es necesario hacer una limpieza para realizar los conteos de infecciones con dicho campo. 
3. Se encontró que existen registros con el campo de fecha de recuperado y fecha de muerte simultaneamente lo cual es una inconsistencia por lo cual se debe realizar un tratamiento.
4. Al validar los conteos de infectados, muertos, recuperados y activos de los ultimos dias reportados, encontramos que los últimos días siempre se registran valores atípicos que en días posteriores se actualizan por lo cual determinamos que lo mejor es descartar los ultimos 3 días de la serie para evitar que dichos datos preliminares generen ruido en las tendencias.


# Preparación de datos

En esta fase se toma como insumo los hallazgos encontrados en la fase anterior para ser ajustados de acuerdo a lo requerido, aquí se seleccionan los datos a usar dentro del proyecto, se aumenta la calidad de los mismos, en caso de ser necesario se construyen nuevos atributos y se integra diferentes fuentes de información

### Actividades realizadas

Las actividades realizadas para la construcción de la vista minable, se relacionan a continuación:

#### Datos seleccionados

Se seleccionaron las cinco (5) ciudades de Colombia con mayor número de casos desde el inicio de la pandemia, estas ciudades son Bogota, Medellín, Cali, Barranquilla y Cartagena.

#### Limpieza

Dado que se tienen varios casos encontrados en los hallazgos, a continuación se menciona las actividades realizadas:

1. Para mejor manejo, se tomó la decisión de convertir todas las fechas a datetime.

2. Para los casos donde se encontró registros con Fecha de muerte y fecha de recuperados a la vez, se decidió dejar únicamente la fecha de acuerdo a su tipo de atención. Asi mismo, se asignó 0 para los contadores de los respectivos casos.

3. Dado que hay casos sin fechas de diagnostico, que hacen referencia a casos asintomaticos o que no se ha asentado su información en la base de datos, se decide imputar su "fecha de inicio de síntomas" con la "fecha de reporte web"

4. Dado que se encontró que para los ultimos días, por cada ciudad, días sin casos o con casos muy diferentes a su tendencia diria, se tomó la decisión de eliminar los ultimos 3 días por cada ciudad. 

##### Derivación de nuevos atributos

1. Para hacer conteos de contagios, recuperados, muertos, activos  y casos confirmados, se toma la decisión de generar variables de conteo teniendo como referencia las fechas.

2. Para generar la vista minable, se toma la decisión de generar un dataframe con una sola columna "fecha", que se calcula como todas las fechas a partir de fechas minimasy fechas maximsa que a su vez se calcularon entre fechas maximas y minimas encontradas de Fecha inicio sintomas, fecha de recuperado y fecha de muerte. 


# Modelamiento

En la fase de modelamiento se evaluan las diferentes técnicas que de acuerdo a la literatura y conocimiento previo tengan una aproximación significativa a la solución del objetivo de mineria de datos. Se selecciona la técnica que más se ajusta, para luego construir el modelo y ejecutarlo.

### Técnicas evaluadas

#### Modelo lógistico
Función mantemática usada en diferentes modelos de crecimiento poblacional, propagación de enfermedades epidémicas. Se caracteriza por un crecimiento incremental al inicio y luego decrecer en un estado del tiempo, dicho crecimiento es carecterizado por la fórmula:
![log_formula](https://miro.medium.com/max/536/1*ktniY6tA5bAZrhRTkyWEBg.png)


Se realizó una aproximación con este método para determinar los casos confirmados en el tiempo. A continuación se muestra el resultado obtenido:

##### Infectados acumulados por día
<p align = "center" >
  <img src="img/Infectados_acumulados_logistica.PNG" alt="Infectados Acumulados por día" width="400px" height="300px"/>
</p>

Como se puede observar, la aproximación usando la función logística supuso un ajuste significativo para este caso. Sin embargo, al repetir el ejercicio con otras variables como recuperados o fallecidos, la función no se ajustó, por lo cual se descartó su uso.

#### Librería Prophet

La libreria de codigo abierto [Prophet](https://facebook.github.io/prophet/#:~:text=Forecasting%20at%20scale.-,Prophet%20is%20a%20forecasting%20procedure%20implemented%20in%20R%20and%20Python,by%20data%20scientists%20and%20analysts.&text=Prophet%20is%20open%20source%20software,download%20on%20CRAN%20and%20PyPI.) desarrollada por facebook permite realizar pronosticos de series de tiempo basada en un [modelo aditivo](https://en.wikipedia.org/wiki/Additive_model#:~:text=In%20statistics%2C%20an%20additive%20model,class%20of%20nonparametric%20regression%20models.) donde las tendencias no lineales se ajustan con una estacionalidad anual, semanal o diaria. Es robsta para el manejo series con estacionalidades y data incompleta y por lo general maneja adecuadamente la ocurrencia de outliers. 
Al modelar los casos acumulados de infectados se encuentra que es capaz de seguir la tendencia como se tenía pero no es capaz de detectar la meseta por lo cual la predicción sigue con una tendencia fuerte al alza.
<p align = "center" >
  <img src="img/prophetInf.JPG" alt="alt" height="300px"/>
</p>
Con los muertos ocurre una situación similar, la tendencia crece indefinidamente
<p align = "center" >
  <img src="img/prophetMuert.JPG" alt="alt" height="300px"/>
</p>

#### SIRD (Suceptible - Infectado - Recuperado - Difuntos)

Los modelos basados en SIR asumen que la infección se propaga a través de varios grupos, por ejemplo, susceptibles, expuestos, infectados, recuperados, fallecidos e inmunes. Para los modelos estándar basados en SIR se tiene que una epidemia incluye susceptibles a la infección (clase S), infectados (clase I) y la población eliminada (clase R).

Los modelos epidemiológico suponen una serie de supuestos, los cuales se detallan:

  * La interacción entre los individuos es aleatoria.
  * Las personas recuperadas no se reinfectan.
  * No se tiene en cuenta que la población $\N$ es variable debido a fenómenos como nacimientos y muertes por otras causas y viajes al extranjero.
  * El modelo no tiene en cuenta que al saturarse el sistema de salud, la tasa de mortalidad $\mu$ tiende a crecer.
  * El modelo no tiene en cuenta las medidas tomadas por el gobierno Nacional.

Para la predicción de la tendencia de Covid-19 en Colombia, se incluye un estado adicional que son los difuntos (clase D). Estos modelos epidemiológicos se rigen por una serie de ecuaciones diferenciales  

<p align = "center" >
  <img src="img/SIRD.png" alt="Ecuaciones Diferenciales SIRD" width="400px" height="300px"/>
</p>

Donde $$\N$$ corresponde a la población general, $$\beta$$ corresponde a la tasa de transmisión de la enfermedad es decir, el número promedio de individuos susceptibles que una persona infectada contagia por día. Esta variable depende del grado de distanciamiento social y de las prácticas higiénicas que adopte la población. Poco distanciamiento social y malas prácticas higiénicas se reflejan en un $\beta$ alto. 

$\gamma$, corresponde a la tasa de recuperación, la cual representa la tasa de personas infectadas que se recuperan y dejan de ser contagiosas.
$\mu$ es la tasa de mortalidad, la cual representa la tasa de personas infectadas que mueren.


##### Predicción de Covid-19 en Colombia usando el modelo SIRD

Para la prediccion del Covid-19 en las principales ciudades de Colombia se llevaron a cabo las siguientes actividades, una vez ejecutada la limpieza de datos y la vista minable ajustada.

Para cada una de las ciudades se calcularon:
  * Tiempo como número de días, a partir de la fecha de inicio de síntomas 
  * Tasa de contagio ($\beta$), Tasa de recuperación ($\gamma$) y tasa de muerte ($\mu$) a nivel de día, partiendo de la información actual y acumulada. Para el tiempo 0 se tomo como refencia un ($\beta$) = 0.05, ($\gamma$)=0.02 y ($\mu$) = 0.005. Estos valores de acuerdo a lo encontrado en la literatura para el Covid-19 o similares.
  * Con las tasas anteriores y teniendo la información historica del comportamiento del virus, se procede a reemplazar los valores en las ecuaciones diferenciales que sigue el modelo. Con esto se halla los números de suceptibles,  muertos,  contagios,  recuperados e infectados diariamente.

Ejemplo del comportamiento del virus en Medellín, a partir de la información real

<p align = "center" >
  <img src="img/med.png" alt="Comportamiento Medellín"/>
</p>

La predicción de los números de suceptibles, muertos,  contagios, recuperados e infectados, se abordó bajo la siguiente metodología:
  * Se busca predecir el corto, mediano y largo plazo, definiendo como corto plazo 5 días, mediano plazo 10 días y largo plazo 15 días.
  * La predicción de las tasas de contagio ($\beta$), tasas de recuperación ($\gamma$) y tasas de muerte ($\mu$) se realizó a partir del cálculo de medias moviles de los últimos 5 días, se probaron diferentes días y se tomo 5 como el número óptimo para el cálculo de medias moviles.
  * Con estas tasas calculas se procede a reemplazar las ecuaciones diferenciales del modelo y con esto se determina el comportamiento del virus en cada una de las cinco ciudades.

  Ejemplo del comportamieto real para los casos activos más predicción a corto, mediano, largo plazo en Medellín

<p align = "center" >
  <img src="img/activos_med.png" alt="Predección Medellín" width="400px" height="200px"/>
</p>


# Evaluación 

En esta etapa se revisa los resultados obtenidos en la ejecución del modelo, con el fin de evaluar las métricas de desempeño que arroja el modelo y determinar si este cumple con los objetivos propuestos.

Despues de desarrollar el modelo SIRD, mediante el calculo de medias moviles en cada uno de los dias a pronosticar, se desarrollo la siguiente tecnica para realizar la evaluación del modelo. 

1. Partición de data.
Se tomo la data generada en la etapa de preparación de datos y se crearon dos dataset, realizando una division a partir de los ultimos 30 dias, contenidos en la información disponible. La primera porsión de datos se destino para realizar la predicción y la segunda la cual contiene los ultimos 30 días fue utilizada como data de contraste con la prediccion que arroja el modelo. Teniendo los 30 días seleccionados como data real y los 30 dias como resultado del modelo, se procedio a calcular el RMSE (Error Cuadratico Medio) para cada modelado realizado por Ciudad y Variable a estimar (muertos,activos,confirmados,suceptibles,contagiados). 

Se hallaron los siguientes resultados.

#### Calculos RMSE

![Alt text](img/RMSE_med.JPG)

![Alt text](img/RMSE_bog.JPG)

![Alt text](img/RMSE_cali.JPG)

![Alt text](img/RMSE_bar.JPG)

![Alt text](img/RMSE_car.JPG)




# Despliegue 

En esta fase se busca entregar los resulados de la solución a la organización.


Para el despliegue se ejecuta el modelo para cada una de las 5 ciudades y como producto de datos se creo un dashboard utilizando el framework de Python, Dash. 

La exposición en internet fue usando heroku, la cual es una plataforma como servicio de computación en la nube, con este despliegue se garantiza que cada vez que se visualice el dashboard se ejecuta el modelo con la información actualizada por el INS diariamente.

Para visualizar el dashboard se puede consultar [Predicción Covid-19 Colombia](https://covid-19-colombia.herokuapp.com)
