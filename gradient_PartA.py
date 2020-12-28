import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (18.0, 9.0)
data = pd.read_csv('dataset.csv')
data.head()
longi = data.iloc[:,1]
lati = data.iloc[:,2]
alti = data.iloc[:,3]
v1 = 0
v2 = 0
v3 = 0
learn = 0.0005
ite = 10000
no_ele = float(len(longi))
errors = np.empty((int)((ite/20)) , dtype=float)
no_ite = np.empty((int)((ite/20)) , dtype=int)
for i in range(ite) :
    alti_pred = v1*longi + v2*lati + v3
    cost = (1 / (2 * no_ele)) * sum((alti - alti_pred) ** 2)
    if i==0 :
        errors[(int)(i / 20)] = cost
        no_ite[(int)(i / 20)] = (int)(i)

    Grad_v1 = (-1/no_ele) * sum (longi* (alti - alti_pred))
    Grad_v2 = (-1/no_ele) * sum(lati* (alti - alti_pred))
    Grad_v3 = (-1/no_ele) * sum(alti - alti_pred)
    v1 = v1 - learn*Grad_v1
    v2 = v2 - learn*Grad_v2
    v3 = v3 - learn*Grad_v3
    alti_pred = v1 * longi + v2 * lati + v3
    cost = (1/(2*no_ele)) * sum ((alti - alti_pred) **2)
    if i%20==0 and i!=0 :
        errors[(int)(i/20)]=cost
        no_ite[(int)(i/20)]=(int)(i)
    if cost<0.000001 :
        break
print(v1,v2,v3)
print(cost)
rmse = np.sqrt(sum((alti - alti_pred) ** 2) /no_ele)
print(rmse)
avg_alti = np.mean(alti)
ss_tot = sum((alti - avg_alti) ** 2)
ss_res = sum((alti - alti_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(r2)
plt.scatter(no_ite,errors)
plt.show()


