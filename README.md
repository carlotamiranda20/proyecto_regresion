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

A continuación hacemos lo mismo para las categóricas con la función df.describe(). En este caso para representar en histogramas, tenemos que primero separar los valores únicos de cada variable (con df[col].unique() y luego sacar la frecuencia de cada uno con el df[col].value_counts()  df[col].unique() y df[col].value_counts(). Vemos que en nivel de dificultad predomina el medio, seguido del fácil y del difícil; en el horario de estudio preferido entre la noche y la tarde apenas se nota diferencia y son mayores que en la mañana; y en el estilo aprendizaje predomina el visual sobre los demás.




