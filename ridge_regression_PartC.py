import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
data = pd.read_csv('dataset.csv')
longi_tr = data.iloc[1:300000,1]
lati_tr = data.iloc[1:300000,2]
alti_tr = data.iloc[1:300000,3]
longi_te = data.iloc[300001:434874,1]
lati_te = data.iloc[300001:434874,2]
alti_te = data.iloc[300001:434874,3]
val_err = np.empty((int)((15)) , dtype=float)
lam_val = np.empty((int)((15)) , dtype=float)
learn = 0.0005
it = 0
a = 0
err_prev = 0
err_pres = 0
v1_f = 0
v2_f = 0
v3_f = 0
err_f=0
e = 50
n = 300000
m = 134874
c =0
for i in range(-5,5) :
    lam=10**i
    v1 = 0
    v2 = 0
    v3 = 0
    for j in range(e):
        alti_pred = v1*longi_tr + v2*lati_tr + v3
        cost= (1 / (2 * n)) * sum((alti_tr - alti_pred) ** 2) + lam * (v1*v1+v2*v2+v3*v3)
        D_v1 = (-1 / n) * sum(longi_tr * (alti_tr - alti_pred)) + (1/n)*lam*v1
        D_v2 = (-1 / n) * sum(lati_tr * (alti_tr - alti_pred)) + (1/n)*lam*v2
        D_v3 = (-1 / n) * sum(alti_tr - alti_pred) +(1/n)*lam*v3
        v1 = v1 - learn * D_v1
        v2 = v2 - learn * D_v2
        v3 = v3 - learn * D_v3
        alti_pred = v1 * longi_tr + v2 * lati_tr + v3
        cost= (1 / (2 * n)) * sum((alti_tr - alti_pred) ** 2) + lam * (v1*v1+v2*v2+v3*v3)
        if cost < 0.0001:
            break
    alti_pred = v1*longi_te + v2*lati_te + v3
    err = (1 / (2 * m)) * sum((alti_te - alti_pred) ** 2) + lam *(v1*v1+v2*v2+v3*v3)
    if it==0 :
        err_prev = err
        err_pres = err
        val_err[it]=err
        lam_val[it]=lam
        it=it+1
    else :
        err_pre = err_pres
        err_pres = err
        val_err[it] = err
        lam_val[it] = lam
        it = it + 1
        if err_pres>=err_prev and c==0 :
            a = lam
            a_b = lam/10
            v1_f = v1
            v2_f = v2
            v3_f = v3
            err_f=err_prev
            c=1

print("error is",err_f)
print("v1 is",v1_f)
print("v2 is",v2_f)
print("v3 is",v3_f)
print("lambda is",a_b)
alti_pred = v1_f * longi_tr + v2_f * lati_tr + v3_f
rmse = np.sqrt(sum((alti_tr - alti_pred) ** 2) /n)
print(rmse)
avg_alti_tr = np.mean(alti_tr)
ss_tot = sum((alti_tr - avg_alti_tr) ** 2)
ss_res = sum((alti_tr - alti_pred) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(r2)
plt.scatter(lam_val,val_err)
plt.xlabel('values')
plt.ylabel('error')
plt.show()











