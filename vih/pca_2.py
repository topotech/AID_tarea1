import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib import pyplot as plt
import math

#Preugnta 1
data = pd.read_csv('data/TB.csv',sep=',',thousands=',', index_col = 0)
data.index.names = ['country']
data.columns.names = ['year']
X = data.ix[:,'1990':'2007'].values
X_std = StandardScaler().fit_transform(X)

fig, ax = plt.subplots(figsize=(8, 4))
data.loc[['Chile','Argentina','Bolivia','United States','Italy', 'United Kingdom', 'Sweden','Zimbabwe','Zambia','Cameroon', 'Congo, Rep,', 'Turkey', 'Singapore', 'Thailand', 'Australia', ''],'1990':].T.plot(ax=ax)
ax.legend(loc='upper left', bbox_to_anchor=(0.1, 1.1),prop={'size':'x-small'},ncol=6)
plt.tight_layout(pad=1.5)


#pregunta d
#Se calcula los vectores medios por columnas (son 22 columnas) y se calcula la matriz de la covarizana como el producto punto  X^T*X/n-1
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
#Se obtienen valores y vectores propios
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#Se genera una lista con los valores propios asociadios a cada vector
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()	
#Se obtiene un vector con los 4 valores propios mas grandes 
var_exp=[]
tot = sum(eig_vals)
for i in range(0,4):
	var_exp.append(sorted(eig_vals, reverse=True)[i]*100/tot)

cum_var_exp = np.cumsum(var_exp)

#Grafico de las varianzas explicadas
plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()

#Se genera una matriz en que cada columna tiene los vectores propios asociados a los dos valores propios mas grandes,
matrix_w = np.hstack((eig_pairs[0][1].reshape(18,1),
                      eig_pairs[1][1].reshape(18,1)))

#Se hace el producto punto entre la matriz X normalizada y la matriz con la cantidad de vectores propios seleccionados
Y_sklearn = X_std.dot(matrix_w)

#Se hace un nuevo dataframe a partir de la matriz con solo dos variables en las cuales se proyectaron las 22
data_2d = pd.DataFrame(Y_sklearn)
data_2d.index = data.index
data_2d.columns = ['PC1','PC2']

#Pregunta e
#Media de los row
row_means = data_2d.mean(axis=1)

#Variacion de los row
row_trends = data_2d.diff(axis=1).mean(axis=1)
#grafico de las medias
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), c=row_means,cmap='hot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')


#GRafico de la variacion
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), c=row_trends,cmap='PRGn')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

#Pregunta f
#El grafocp de burbujas
data_2d.plot(kind='scatter', x='PC1', y='PC2', figsize=(16,8), s=10*row_means, \
c=row_means,cmap='PRGn')

#Pregunta g
fig, ax = plt.subplots(figsize=(16,8))
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC1', y='PC2', ax=ax, s=10*row_means, c=row_means, \
cmap='PRGn')
Q3_HIV_world = data.mean(axis=1).quantile(q=0.85)
HIV_country = data.mean(axis=1)
names = data.index
for i, txt in enumerate(names):
	if(HIV_country[i]>Q3_HIV_world):
		ax.annotate(txt, (data_2d.iloc[i].PC1+0.2,data_2d.iloc[i].PC2))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
