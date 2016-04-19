# *-* coding: utf-8 *-*

import pandas as pd
import matplotlib.pyplot as plt


#pd.options.display.mpl_style = 'default'
plt.style.use('ggplot')
data = pd.read_csv('data/titanic-train.csv', sep=';')


print "\nForma matriz:" + str(data.shape)
print "--------------------\n"

print "Resumen de los datos:\n" + str(data.describe())
print "--------------------\n"

print "Nombres, tipos y cantidad de datos en cada variable:"
print data.info()

print " =============================================================== \n =============================================================== \n"

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

