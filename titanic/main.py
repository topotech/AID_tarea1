# *-* coding: utf-8 *-*

import pandas as pd
import matplotlib.pyplot as plt


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

data['prediction'] = 0

#Se predice supervivencia de mujeres y primera clase. Si es mujer y de tercera, se rectifica a 0.
data.loc[(data.Sex == 'female') | (data.Pclass == 1), 'prediction'] = 1



print "Porcentaje de falsos positivos (se creyeron vivos pero murieron):\n" + \
      str(data[data.prediction == 1][data.Survived == 0].size/float(data[data.prediction == 1].size))

print "Porcentaje de falsos negativos (se creyeron muertos, pero sobrevieron):\n" + \
      str(data[data.prediction == 0][data.Survived == 1].size/float(data[data.Survived == 1].size))

data.to_csv('predicciones-titanic.csv')
