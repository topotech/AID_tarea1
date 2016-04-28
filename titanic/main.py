# *-* coding: utf-8 *-*

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Parte a)

#pd.options.display.mpl_style = 'default'
plt.style.use('ggplot')
data = pd.read_csv('data/titanic-train.csv', sep=';')


#Parte b)
print "\nForma matriz:" + str(data.shape)
print "--------------------\n"

print "Resumen de los datos:\n" + str(data.describe())
print "--------------------\n"

print "Nombres, tipos y cantidad de datos en cada variable:"
print data.info()

print " =============================================================== \n =============================================================== \n"


#Parte c)

print "Cola de la data:"
print "--------------------\n"
print data.tail()
print "--------------------\n"

print "Cabeza de la data:"
print "--------------------\n"
print data.head()
print "--------------------\n"

print "Datos 200-210:"
print "--------------------\n"
print data[200:210][:]
print "--------------------\n"

print "Cola de la data filtrada:"
print "--------------------\n"
print data[['Sex','Survived']].tail()
print "--------------------\n"

print "Cabeza de la data filtrada:"
print "--------------------\n"
print data[['Sex','Survived']].head()
print "--------------------\n"

print "Datos 200-210 filtrados:"
print "--------------------\n"
print data[['Sex','Survived']][200:210]
print "--------------------\n"


print " =============================================================== \n =============================================================== \n"

#Parte d)

print "Número de personas por sexo:"
print data['Sex'].value_counts()
print''
print "Número de sobrevivientes por sexo"
print data[data['Survived']==1].groupby('Sex').Survived.count()
print''
print "% personas sobrevivientes por sexo (usando la media)"
print data.groupby('Sex').Survived.mean()
print''
print "Data (Survived x Sex)"
print data.groupby('Survived')['Sex'].value_counts()
print''
print "Data (Sex x Survived)"
print data.groupby('Sex')['Survived'].value_counts()
print''
print "Renderizando plot..."
data.groupby('Sex')['Survived'].value_counts().unstack().plot(kind='bar')
plt.show()

print "Data porcentual (Survived x Sex)"
grouped_props = data.groupby('Survived')['Sex'].value_counts()/\
                data.groupby('Survived').size()

print grouped_props
print "Renderizando plot..."
grouped_props.unstack().plot(kind='bar')
plt.show()

print " =============================================================== \n =============================================================== \n"

#Parte e)

print "Edad promedio de sobrevivientes y muertos"
print data.groupby('Survived')['Age'].mean()
print ""

print "Cargando boxplots e histograma edad vs supervivencia..."
data.boxplot(column='Age',by='Survived')
data.hist(column='Age',by='Survived')
print "Renderizando plots..."
plt.show()

print "-----\nCantidad de muertos agrupados por conocimiento de edad:"
print "Desconocida:\t" + str(sum(data[data.Survived==0]['Age'].isnull()))
print "Conocida:\t\t" + str(sum(data[data.Survived==0]['Age'].notnull()))


print "\nDatos de persona más vieja:"
print data[data.Age==data['Age'].max()]

print " =============================================================== \n =============================================================== \n"

#Parte f)

# Respuesta a la pregunta final: Una instrucción del tipo 'data[CONDICION][LABEL]' genera una copia de los datos del DataFrame
# Por ello se recomienda usar una linea del tipo DataFrame.loc[row_indexer,col_indexer] para modificar los datos por referencia.

print "Se establece la media aritmética como el valor de reemplazo para las edades nulas. Ya que es una edad estimada " \
      "se utiliza la notación xx.5:\n"

M_average_age = float(int(data[data.Sex == 'male']['Age'].mean())) + .5
F_average_age = float(int(data[data.Sex == 'female']['Age'].mean())) + .5
data.loc[(data.Age.isnull()) & (data.Sex =='male'), 'Age']  = M_average_age
data.loc[(data.Age.isnull()) & (data.Sex =='female'), 'Age']  = F_average_age

print "Nueva edad masculina: " + str(M_average_age)
print "Nueva edad femenina: " + str(F_average_age)

print " =============================================================== \n =============================================================== \n"

#Parte g)

#Sabemos por la definición que existe 1º, 2º y 3º clase. Ratificamos:
print "Número de clases: " + str(data['Pclass'].unique().size) + "\n"

print "De los sobrevivientes: ¿cuántas partes constitutía cada clase?:\n"
print data.groupby(['Survived', 'Pclass']).size()/data.groupby(['Survived']).size()
print ""

print "¿Cuántos sobrevivieron de cada clase porcentualmente?:\n"
print data.groupby(['Pclass','Survived']).size()/data.groupby(['Pclass']).size()


females = data[data.Sex == 'female'].groupby(['Survived','Pclass']).size()/\
          data[data.Sex == 'female'].groupby(['Survived']).size()

males = data[data.Sex == 'male'].groupby(['Survived','Pclass']).size()/\
          data[data.Sex == 'male'].groupby(['Survived']).size()

print "¿Cuál es la proporción de sobreviviente por clase entre los hombres?:\n"
print males
print "¿Cuál es la proporción de sobreviviente por clase entre las mujeres?:\n"
print females

males.unstack().plot(kind='bar')
females.unstack().plot(kind='bar')
plt.show()

print " =============================================================== \n =============================================================== \n"

#Parte h)

print "Predicción con Regla 1 (sobreviven todos los de 1º clase y las mujeres de 2º):"

data['prediction'] = 0

#Se predice supervivencia de mujeres y primera clase. Si es mujer y de tercera, se rectifica a 0.
data.loc[(data.Sex == 'female') | (data.Pclass == 1), 'prediction'] = 1



print "Precisión:\n" + \
      str(data[data.prediction == 1][data.Survived == 1].size/float(data[data.prediction == 1].size))

print "Recall\n" + \
      str(data[data.prediction == 1][data.Survived == 1].size/float(data[data.Survived == 1].size))

data.to_csv('predicciones-titanic.csv')


print " =============================================================== \n =============================================================== \n"

#Parte i)

print "Nuevo análisis particionando en 5 clases según costo de pasaje:"

_, myBP = data.boxplot(column='Fare', return_type='both')
whiskers = [whiskers.get_ydata() for whiskers in myBP["whiskers"]]

print "Rango de datos relevantes: ["+str(whiskers[0][1])+","+str(whiskers[1][1])+"]" # Imprime  [bigote_inf,bigote_sup]
# De la impresión anterior se concluye que los outliers están sobre 65.0

dataFareTyp = data[ data.Fare <= whiskers[1][1] ].copy()
dataFareTyp.hist(column='Fare')
plt.show()


# Analisis de datos

dataFareTypSurv = data[(data.Fare <= whiskers[1][1]) & (data.Survived == 1)]
dataFareTypDied = data[(data.Fare <= whiskers[1][1]) & (data.Survived == 0)]
fig, ax = plt.subplots()

sns.distplot(dataFareTypSurv['Fare'], bins=[0, 10, 20, 30, 40, 65] ) #Curva pequeña
sns.distplot(dataFareTypDied['Fare'], bins=[0, 10, 20, 30, 40, 65] ) #Curva alta

#Muestra nuevo histograma con las nuevas "clases"
plt.show()

#Se usa dataFareTyp para definir nuevo criterio de clasificación económica


dataFareTyp.loc[(dataFareTyp.Fare >= 0) & (dataFareTyp.Fare < 10), 'Pclass'] = 5
dataFareTyp.loc[(dataFareTyp.Fare >= 10) & (dataFareTyp.Fare < 20), 'Pclass'] = 4
dataFareTyp.loc[(dataFareTyp.Fare >= 20) & (dataFareTyp.Fare < 30), 'Pclass'] = 3
dataFareTyp.loc[(dataFareTyp.Fare >= 30) & (dataFareTyp.Fare < 40), 'Pclass'] = 2
dataFareTyp.loc[(dataFareTyp.Fare >= 40) , 'Pclass'] = 1

# Se usa dataFareTyp para definir nuevo criterio de clasificación
# Tras observar el histograma, se define que los muertos serán todos de 5º clase...

print "Predicción con Regla 2 (Se salvan todos menos 5º clase):"

dataFareTyp['prediction'] = 1
data.loc[(data.Pclass == 5) , 'prediction'] = 0


print "Precisión:\n" + \
      str(dataFareTyp[dataFareTyp.prediction == 1][dataFareTyp.Survived == 1].size/float(dataFareTyp[dataFareTyp.prediction == 1].size))

print "Recall\n" + \
      str(dataFareTyp[dataFareTyp.prediction == 1][dataFareTyp.Survived == 1].size/float(dataFareTyp[dataFareTyp.Survived == 1].size))



print " =============================================================== \n =============================================================== \n"

#Parte j)

print "Se ponen a prueba las dos reglas de predicción sobre los datos de testing:\n"


data = pd.read_csv('data/titanic-test.csv', sep=',')
survival = pd.read_csv('data/titanic-gendermodel.csv', sep=',')

data = data.merge(survival,on='PassengerId').copy()

regla1 = data.copy()
regla2 = data.copy()



regla1['prediction'] = 0
regla1.loc[(regla1.Sex == 'female') | (regla1.Pclass == 1), 'prediction'] = 1


regla2['prediction'] = 1
regla2.loc[regla2.Fare < 10 , 'prediction'] = 0


print "\nRegla 1\n-------"

print "\tPrecisión:\n\t" + \
      str(regla1[regla1.prediction == 1][regla1.Survived == 1].size/float(regla1[regla1.prediction == 1].size))

print "\n\tRecall:\n\t" + \
      str(regla1[regla1.prediction == 1][regla1.Survived == 1].size/float(regla1[regla1.Survived == 1].size)) + "\n"


print "\nRegla 2\n-------"

print "\tPrecisión:\n\t" + \
      str(regla2[regla2.prediction == 1][regla2.Survived == 1].size/float(regla2[regla2.prediction == 1].size))

print "\n\tRecall:\n\t" + \
      str(regla2[regla2.prediction == 1][regla2.Survived == 1].size/float(regla2[regla2.Survived == 1].size))



regla1['prediction'] = 0
regla1.loc[(regla1.Sex == 'female') | (regla1.Pclass == 1), 'prediction'] = 1


regla2['prediction'] = 1
regla2.loc[(regla2.Pclass == 5) , 'prediction'] = 0