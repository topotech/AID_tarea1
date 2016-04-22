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

#Pregunta 2

#Pregunta 3


#pregunta 4
#Se calcula los vectores de la media
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()	
var_exp=[]
tot = sum(eig_vals)
for i in range(0,4):
	var_exp.append(sorted(eig_vals, reverse=True)[i]*100/tot)


cum_var_exp = np.cumsum(var_exp)
plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center',
        label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid',
         label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()





matrix_w = np.hstack((eig_pairs[0][1].reshape(18,1),
                      eig_pairs[1][1].reshape(18,1)))
Y_sklearn = X_std.dot(matrix_w)

data_2d = pd.DataFrame(Y_sklearn)
data_2d.index = data.index
data_2d.columns = ['PC1','PC2']

row_means = data_2d.mean(axis=1)
row_trends = data_2d.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), c=row_means,cmap='hot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')



row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), c=row_trends,cmap='seismic')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

data_2d.plot(kind='scatter', x='PC2', y='PC1', figsize=(16,8), s=10*row_means, \
c=row_means,cmap='hot')

fig, ax = plt.subplots(figsize=(16,8))
row_means = data.mean(axis=1)
row_trends = data.diff(axis=1).mean(axis=1)
data_2d.plot(kind='scatter', x='PC2', y='PC1', ax=ax, s=10*row_means, c=row_means, \
cmap='RdBu')
Q3_TB_world = data.mean(axis=1).quantile(q=0.85)
TB_country = data.mean(axis=1)
names = data.index
for i, txt in enumerate(names):
	if(TB_country[i]>Q3_TB_world):
		ax.annotate(txt, (data_2d.iloc[i].PC2+0.2,data_2d.iloc[i].PC1))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
