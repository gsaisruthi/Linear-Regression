import numpy as np
import pandas as pd

data = pd.read_excel('C:\\Users\\sruthi\\Desktop\\datacsv.csv')
data
x1= data.iloc[:,1]
x2=data.iloc[:,2]
y= data.iloc[:,3]

x1=np.array(x1)
x2=np.array(x2)

mu1=np.mean(x1)
sigma1=np.var(x1)
x1=(x1-mu1)/sigma1

x1

mu2 = np.mean(x2)
sigma2=np.var(x2)
x2=(x2-mu2)/sigma2

x2

w0=0
w1=0
w2=0

w1=[]
w2=[]
j=[]

n=len(x1)

for i in range(5000):
    y_pred=w1*x1+ x2+w0
    d0=(2/n)*sum(y-y_pred)
    d1=(2/n)*sum(x1*(y-y_pred))
    d2=(2/n)*sum(x2*(y-y_pred))
    
    
    w0=w0+0.0005*d0
    w1=w1+0.0005*d1
    w2=w2+0.0005*d2
    
    cost_function=0.5*sum((y-y_pred)*(y-y_pred))
    
    if(i%500==0 or i==0);
        w1.append(w1)
        w2.append(w2)
        j.append(cost_function)
    
    x_axis=np.linspace(0,4500,num=10,dtype=int)
    x-axis
    
    import matplotlib.pyplot as plt
    
    plt.plot(x_axis,j)
    
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax=plt.axes(projection='3d')
               
              # ax.plot3D(w1,w2,j,))
    
    #Part-D

s0=0
s1=0
s2=0

S1=[]
S2=[]
J1=[]
N1=[]

for t in range(500):
    for i in range(n):
        y_pred = s1*x1[i] + s2*x2[i] + s0
        d0 = (2/n)*(y[i] - y_pred)
        d1 = (2/n)*(x1[i]*(y[i] - y_pred))
        d2 = (2/n)*(x2[i]*(y[i] - y_pred))
    
s0= s0+0.0005*d0
s1= s1+0.0005*d1
s2= s2+0.0005*d2
cost_function= 0.5*(( y[i] - y_pred)*y[i])

N1.append(t)
S1.append(s1)
S2.append(s2)
J1.append(cost_function)


from mpl_toolkits import mplot3D
import matplotlib.pyplot as plt

plt.plot(J1,N1)
fig = plt.figure()
ax = plt.axis(projection = '')
ax.plot3D(S1,S2,J1,'gray')



