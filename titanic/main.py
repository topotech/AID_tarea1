# *-* coding: utf-8 *-*

import pandas as pd


pd.options.display.mpl_style = 'default'
data = pd.read_csv('data/titanic-train.csv', sep=';')


print "\nForma matriz:" + str(data.shape)

print data.describe() #Resumen de la data

print data.info() #Nombre y tipos de las columnas




