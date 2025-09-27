
En este proyecto vamos a realizar dos modelos de machine learning para un archivo dado que trata sobre el rendimiento académico de estudiantes. Para ello primero vamos a realizar un análisis exploratorio y un preprocesamiento de los datos. Luego realizaremos un modelo de regresión y otro de clasificación sobre las variables objetivo.

Sabemos que las columnas que contiene el archivo son:
		horas_estudio_semanal: Número de horas de estudio a la semana.
  
		nota_anterior: Nota que obtuvo el alumno en la convocatoria anterior.
  
		tasa_asistencia: Tasa de asistencia a clase en porcentaje.
  
		horas_sueno: Promedio de horas que duerme el alumno al día.
  
		edad: Edad del alumno.
  
		nivel_dificultad: Dificultad del alumno para el estudio.
  
		tiene_tutor: Indica si el alumno tiene tutor o no.
  
		horario_estudio_preferido: Horario de estudio preferido por el alumno.
  
		estilo_aprendizaje: Forma de estudio que emplea el alumno. 
  
Además también tiene otras dos columnas que son las variables objetivo (sobre las que se van a realizar los modelos):

		nota_final: Calificación final del alumno, variable numérica entre 0 y 100, sobre la que se debe hacer el modelo de regresión.
  
		aprobado: Binario, si el alumno ha aprobado o no (1 si la nota es ≥ 60, 0 en caso contrario), sobre la que se debe hacer el modelo de clasificación.
	

1. ANÁLISIS EXPLORATORIO

  En primer lugar cargamos las librerías de Python con las que vamos a trabajar. A continución leemos y guardamos el fichero dataset_estudiantes.csv como df y hacemos un df.head() que nos muestra las 5 primeras filas para ver qué pinta tiene el archivo.
	Seguimos analizando el archivo en líneas generales, mirando el número de filas y columnas (1000 y 11 respectivamente). Con df.info() obtenemos el nombre de las distintas columnas, el tipo que son y la cantidad de no nulos que tienen.
	Además de los nulos, también miramos las filas duplicadas que en este caso no hay ninguna. Los valores nulos son 150 en horas de sueño, 100 en horas de estudio preferido y 50 en estilo de aprendizaje. A priori parece que no representan un porcentaje muy alto teniendo en cuenta que hay 1000 filas, pero más adelante vamos a ver cómo los gestionamos.
	Lo siguiente que hacemos es identificar las variables númericas y las categóricas y separarlas en dos listas diferentes, num_cols y cat_cols, y obtener sus estadísticas descriptivas.
Comenzamos con las numéricas, sacamos las métricas generales con df.describe. 

También representamos histogramas de las variables.

Vemos que en líneas generales siguen una distribución normal (excepto aprobados que es una variable binaria y la edad que es discreta). En el caso de nota_final vemos que la mayoría de notas están entre 60 y 80, aunque hay algunos extremos cerca de 30 y 100, siendo el promedio 71. Notamos que la media de aprobados es 0.9, por tanto está bastante desbalanceado el número de aprobados y suspensos, algo a tener en cuenta de cara al modelo de clasificación. 
(Considero realizar comentarios sobre aprobados y nota_final por ser las variables objetivo y no extenderme desarrollando cada gráfica.)

También hacemos una matriz de correlación para ver cómo se relacionan entre sí los datos. Las correlaciones relevantes son:

			-nota_final se correlaciona bastante con nota_anterior (≈ 0.79) 
   
			-nota final también con tasa_asistencia (≈ 0.62).
   
   			-horas_estudio_semanal muestra correlación positiva moderada con nota_final.
	 
			-horas_sueno y edad casi no influyen en la nota.

Gracias a esta representación, vemos que debido a la poca influencia que tiene horas_sueno en la nota_final y aprobado, no será necesario tratar los valores nulos, teniendo en cuenta que además solo representan un 15% del total y podemos imputarlos por el valor promedio que es 7.

A continuación hacemos lo mismo para las variables categóricas con la función df.describe(). En este caso para representar en histogramas, tenemos que primero separar los valores únicos de cada variable (con df[col].unique() y luego sacar la frecuencia de cada uno con el df[col].value_counts()  df[col].unique() y df[col].value_counts(). Vemos que en nivel de dificultad predomina el medio, seguido del fácil y del difícil; en el horario de estudio preferido entre la noche y la tarde apenas se nota diferencia y son mayores que en la mañana; y en el estilo aprendizaje predomina el visual sobre los demás.

2- PREPROCESAMIENTO

2.1 GESTIÓN DE NULOS
Lo siguiente que vamos a hacer es gestionar los valores nulos que hay en las columnas. Primero identificamos cuántos valores nulos hay en cada columna. Vemos que solo hay en 3: horas_sueno (que es numérica), horario_estudio_preferido y estilo_aprendizaje. El número de valores nulos no representa un porcentaje muy significativos, pero vamos a imputarlos para que después no nos dé errores a la hora de hacer el modelo. En la variable horas_sueno vamos a sustituir los nulos por la mediana, y en las variables categóricas vamos a sustituir por la moda (el valor que más se repite).
2.2 GESTIÓN DE OUTLIERS
Vamos a gestionar los outliers para que no desajusten el modelo. En primer lugar mediente diagramas de caja hacemos una visualización gráfica. Construimos una subfigura con los diagramas. Escogemos los diagramas numéricos, excepto 'Aprobado' ya que es una variable binaria y no tiene sentido buscarle outliers.
Tras representarlo vemos que las variable que los tienen son horas_estudio_semanal, tasa_asistencia y nota_final.
Lo siguiente que hacemos es eliminarlos. Para ello hacemos una copia del dataframe y construimos un bucle que hace que solo nos quedemos con los valores que estén entre el pimer cuartil (Q1) y el tercero (Q3). 
Guardamos el nuevo dataframe

2.3 CODIFICACIÓN MODELO REGRESIÓN.

Lo primero que vamos a hacer es hacer una copia del data frame y seleccionar la variable objetivo, en este caso nota_final.

Vamos a codificar las variables categóricas que tenemos (ya que el modelo es numérico). 
Para ello seleccionamos las variables que nos interesan , y en este caso vamos a aplicar el método de codificación onehot, es decir que lo transforma en columnas binarias. Escogemos este ya que no tenemos muchas variables distintas dentro de cada categória (3/4) n i muchas categóricas (4).

Para escribir el código utlizmos la librería sklearn y la función OneHotEncoder. Aplicamos el fit_transform sobre las columnas categóricas (onehot_cols) del dataframe df_reg, que nos devuelve una matriz binaria. Luego con onehot_encoder.get_feature_names_out() generamos los nombres de las nuevas columnas y con pd.DataFrame convertimos la matriz en un dataframe y con el mismo índice que df_reg (index=df_reg.index). Por último concatenamos con el dataframe original y eliminamos las columnas originales. Así df_reg finalmente tiene solo variables numéricas.
Para completar todo el preproceso, a continuación hacemos un escalado de las variables con MinMaxScaler, esto sirve para evitar que valores grandes dominen sobre el resto. Escalamos todos los datos entre los valores [0,1]. Seleccionamos todas las columnas excepto el target y con scaler.fit_transform ajusta y transforma los datos en valores entre el rango [0,1]. 

Finalmente guardmos el dataframe preprocesado, preparado para realizar el modelo de regresión. 


2.4 MODELO CLASIFICACIÓN

En primer lugar hacemos copia del dataframe y seleccionamos la variable objetivo, aprobado.
Hacemos lo mismo que para el modelo de regresión (solo que ahora el dataframe es df_clas). Tenemos que codificar las variables categóricas, que son las mismas que antes. Las codificamos mediante onehot de nuevo por la misma razón. 

Hacmos el escalado de nuevo, en este caso nos da igual que escale la variable objetivo, aprobados ya que es una variable binaria y el escalado la va a dejar como estaba.

Guardamos el dataframe preprocesado. 

3. MODELO REGRESIÓN

Vamos a implementar el modelo de regresión sobre la variable nota_final. Para ello en primer lugar cargamos el dataframe después de haberle hecho el preprocesamiento.
Definimos las variables predisctoras y el objetivo. 
Después dividimos los datos en el entrenamiento y la prueba. Es decir vamos a tener dos conjuntos de datos. y_train, x_train son los datos reales con los que vamos a entrenar el modelo, y_test, x_test con los datos que separamos para luego comparar cómo se ajusta el modelo entrenado. Lo hacemos con un 90% entrenamiento y 10% prueba. (Inicialmente había probado hacerlo con un 80-20, pero había bastante diferencia entre los datos de entrenamiento y los de prueba.)

A continuación hacemos el ajuste del entrenamiento, que en este caso va a ser lineal, con modelo.fit. 
Una vez obtenido el modelo de estos datos de entrenamiento, aplicamos el modelo a los datos que separamos anteriormente, ese 10% que llamamos antes y_test, x_test y ahora obtenemos un nuevo conjunto de datos que son los predecidos por el modelo, y_pred, x_pred.

Una vez obtenidos los datos de esta forma, hacemos la comparación entre los valores reales (y_test,x_test) y los predichos. 

Hacemos una representación gráfica de las prediccionesy la línea ideal, vemos que se ajustan más o menos. También calculamos los residuos y apreciamos mejor que los datos están centrados en la línea del 0 pero los hay bastante dispersos. Finalmente calculamos las métricas (el R2) para compararlos cuantitativamente, y vemos que el modelo se ajusta un 53%. Se ajusta más de un 50% pero es un valor bastante bajo. Sin embargo vemos que los R2 del conjunto de entrenamiento y del de prueba son similares (53% y 54%), podemos decir que el modelo sí se ajusta. El R2 global no es muy elevado debido a que la estadística de datos es baja (1000 filas) por tanto al modelo le cuesta más ajustarse al haber pocos. 
Podemos concluir que seguramente este ajuste lineal no sea el mejor para este dataset.
Como se ha comentado antes, inicialmente con un entrenamiento de 80-20, obteníamos un R2 de entrenamiento de 48% y de la prueba un 53%, me pareció bastante diferencia además de valores muy bajitos y por eso opté por coger más datos para el entrenamiento.


También representamos el histograma de residuos, que debe tener forma gaussiana para un buen ajuste , en este caso se ve un poco deformada.

Por último se pide guardar este modelo en un fichero de tipo pkl para poder usarlo con otros conjuntos de datos. Para ello lo que hacemos es hacer el fit con todos los datos. Y por úlitmo con la biblioteca de joblib lo guardamos como modelo_regresion.plk.




