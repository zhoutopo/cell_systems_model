#!/usr/bin/env python
# coding: utf-8

# In[383]:


import numpy as np
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category10_10  
from bokeh.io import output_notebook
import pickle
from bokeh.models import Range1d,LinearAxis
import bokeh.io
from bokeh.layouts import *
bokeh.io.output_notebook()


# In[410]:


global T,noise,groups,dt,initial,mu1,sigma1,mu2,sigma2
T=120
noise=0.3
groups=500
dt=0.1
initial=[0.49,0.86]
mu1,sigma1,mu2,sigma2 = 0,0.05,0,0.05
def gly_ode(S,H,D,p):
    
    St,Ht = piece_wise(D,p)
        
    dSdt = 0.01*(St-S) + S**3/(0.52**3+S**3)*(St-S) - (0.1+H)*S
    dHdt = 0.01*(Ht-H) + H**3/(0.62**3+H**3)*(Ht-H) - (0.3+S)*H

    return np.array([dSdt, dHdt])

def piece_wise(D,p):
    k1,k2,k3,k4,d1,d2,d3,d4=p
    if D>=0.02 and D<=0.25:
        St = k1*D + d1
        Ht = k2*D + d2
    elif D>=0.25:
        St = k3*D + d3
        Ht = k4*D + d4
        
    return St,Ht

def solver(D,parameters):
    N = int(T/dt)
    s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
    h_init=np.random.normal(mu2,sigma2,groups)+initial[1]
    S = np.zeros((N,groups))
    H = np.zeros((N,groups))
    S[0,:]=s_init
    H[0,:]=h_init
    S_rand=np.random.randn(N, groups)
    H_rand=np.random.randn(N, groups)
    
    for i in range(N-1):    
        S[i,S[i,:]<0] =0
        H[i,H[i,:]<0] =0
        dS,dH = gly_ode(S[i,:],H[i,:],D,parameters)*dt 
        S[i+1,:] = S[i,:] + dS + noise*S_rand[i,:]*S[i,:]*dt
        H[i+1,:] = H[i,:] + dH + noise*H_rand[i,:]*H[i,:]*dt 
    return S,H

def mode1_ratio(D,S,H):
    t,groups=np.shape(H)
    if D==0.1: #delay factor
        if np.sum(S[200,:]<0.2)+np.sum(H[200,:]<0.2) < groups*0.9 and (
            np.sum(S[-1,:]<0.2)-np.sum(S[200,:]<0.2) > groups*0.04
        ): 
            #delayed commitment and more mode1 than mode2 late in life
            return np.sum(S[-1,:]<0.2)/(np.sum(S[-1,:]<0.2)+np.sum(H[-1,:]<0.2))
        else:
            return np.inf
    else: 
        if np.sum(S[200,:]<0.2)+np.sum(H[200,:]<0.2) > groups*0.9:
            return np.sum(S[-1,:]<0.2)/groups
        else:
            return np.inf


# In[78]:


#determine St and Ht at turning point (D=0.25%)
#method I smallest distance
# dist=np.inf
# conc=[0.25]
# for i in range(0,10000):
#     k1,k2,k3,k4=np.random.uniform(0,-0.9,4)
#     d1,d2,d3,d4=np.random.uniform(0,3,4)
#     parameters=[k1,k2,k3,k4,d1,d2,d3,d4]
#     ratio=[]
    
#     for D in conc:
#         St,Ht=piece_wise(D,parameters)
#         if St<=0 or Ht<=0:
#             ratio=np.array([np.inf]*len(conc))
#             break
            
#         S,H=solver(D,parameters)
#         ratio.append(mode1_ratio(D,S,H))
#     dist_new=sum(np.abs(np.array(ratio)-np.array([0.69])))
#     if dist_new<dist:
#         dist=dist_new
#         p=[k1,k2,k3,k4,d1,d2,d3,d4]


# In[239]:


# #mathod II use St&Ht to find mode ratio
# def gly_ode_vSH(S,H,St,Ht):
        
#     dSdt = 0.01*(St-S) + S**3/(0.52**3+S**3)*(St-S) - (0.1+H)*S
#     dHdt = 0.01*(Ht-H) + H**3/(0.62**3+H**3)*(Ht-H) - (0.3+S)*H

#     return np.array([dSdt, dHdt])

# def solver_v2(St,Ht):
#     N = int(T/dt)
#     s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
#     h_init=np.random.normal(mu2,sigma2,groups)+initial[1]
#     S = np.zeros((N,groups))
#     H = np.zeros((N,groups))
#     S[0,:]=s_init
#     H[0,:]=h_init
#     S_rand=np.random.randn(N, groups)
#     H_rand=np.random.randn(N, groups)
    
#     for i in range(N-1):    
#         S[i,S[i,:]<0] =0
#         H[i,H[i,:]<0] =0
#         dS,dH = gly_ode_vSH(S[i,:],H[i,:],St,Ht)*dt 
#         S[i+1,:] = S[i,:] + dS + noise*S_rand[i,:]*S[i,:]*dt
#         H[i+1,:] = H[i,:] + dH + noise*H_rand[i,:]*H[i,:]*dt 
#     return S,H


# In[122]:


# #determine St and Ht at turning point (D=0.25%)
# hit=[]
# no_hit=[]
# D=0.25
# for i in range(0,10000):
#     St=np.random.uniform(1.5,2.2)
#     Ht=np.random.uniform(1.5,2.2)
#     S,H = solver_v2(St,Ht)
#     if mode1_ratio(D,S,H)>0.685 and mode1_ratio(D,S,H)<0.694:
#         hit.append((St,Ht))
#     else:
#         no_hit.append((St,Ht))


# In[395]:


# # determine St and Ht at each point
# D_Ratio=[(0.02,0.99),(0.1,0.82),(0.25,0.68),(2,0.48),(4,0.44),(5,0.25)]
# # D_Ratio=[(0.1,0.78)]
# # D_Ratio=[(0.02,0.99)]
# data={'D':[],'St':[],'Ht':[]}
# for pair in D_Ratio:

#     D=pair[0]
#     for i in range(0,10000):
#         if i%5000==0:
#             print(i)
#         St=np.random.uniform(1.4,1.85)
#         Ht=np.random.uniform(1.4,2.45)
#         S,H = solver_v2(St,Ht)
#         if mode1_ratio(D,S,H)>pair[1]-0.01 and mode1_ratio(D,S,H)<pair[1]+0.01:
#             data['D'].append(D)
#             data['St'].append(St)
#             data['Ht'].append(Ht)
            
# pickle.dump(data, open(".\ fitting_dots.p", "wb+"))


# In[428]:


# x=np.array(data['D'])
# y=np.array(data['St'])
# z=np.array(data['Ht'])
# ind=np.where(x==2)
# SH=np.array(list(zip(y[ind],z[ind])))
# # max_v=np.max(Ht)
# # St[np.where(Ht==max_v)[0]],max_v
# SH[np.argsort(SH[:,1])[::-1]]


# In[413]:


S,H=solver_v2(1.519, 1.83)
t=np.linspace(0,T,int(T/dt))
ts=np.vstack([t]*groups)
ts=ts.tolist()
xs=S.transpose()
ys=H.transpose()
Sm=xs.tolist()
Hm=ys.tolist()
f1=figure(width=200,height=150)
f1.multi_line(xs=ts,ys=Hm)
show(f1)


# In[396]:


Ss,Hs,Su,Hu=pickle.load(open(".\model_stable_points.p", "rb"))
data=pickle.load(open(".\ fitting_dots.p","rb"))
# Ss,Hs,Su,Hu=pickle.load(open(".\model_stable_points_shift.p", "rb"))
D=np.linspace(0.02,5,100)

f14 = figure(width=400,height=400,x_axis_label="St",y_axis_label="Ht")
f14.grid.visible=False
f14.square(Ss,Hs,size=4,fill_color="#2980B9",line_color="#2980B9")
f14.square(Su,Hu,size=4,fill_color="#F4D03F",line_color="#F4D03F")
f14.x_range = Range1d(1.34, 2.4)
f14.y_range = Range1d(1.34, 2.4)

tags = data.get('D')
x_values = data.get('St')
y_values = data.get('Ht')
tag_colors = {tag: color for tag, color in zip(set(tags), Category10_10[::-1])}
source_data = {
    'x': x_values,
    'y': y_values,
    'tags': tags,
    'color': [tag_colors[tag] for tag in tags]
}
source = ColumnDataSource(source_data)
f14.scatter(x='x', y='y', size=2, color='color', legend_field='tags', source=source)
show(f14)


# In[384]:


f1=figure(width=400,height=400,x_axis_label="D",y_axis_label="Ht")
f1.scatter(x='tags', y='y', size=2, color='color', legend_field='tags', source=source)
f2=figure(width=400,height=400,x_axis_label="D",y_axis_label="St")
f2.scatter(x='tags', y='x', size=2, color='color', legend_field='tags', source=source)
show(row(f1,f2))


# In[389]:


#determine parameters for fast-change line
# St025=1.54873468
# Ht025=1.88831094
# conc=[0.02,0.1]
# dist=np.inf

# for i in range(0,5000):
#     if i%1000==0:
#         print(i)
#     k1,k2=np.random.uniform(0,-1.6,2)
#     d1=St025-0.25*k1
#     d2=Ht025-0.25*k2
#     k3,k4,d3,d4=0,0,0,0
#     parameters=[k1,k2,k3,k4,d1,d2,d3,d4]
#     ratio=[]
    
#     for D in conc:
#         St,Ht=piece_wise(D,parameters)
#         if St<=0 or Ht<=0:
#             ratio=np.array([np.inf]*len(conc))
#             break
            
#         S,H=solver(D,parameters)
#         ratio.append(mode1_ratio(D,S,H))
#     dist_new=sum(np.abs(np.array(ratio)-np.array([0.97,0.75])))
    
#     if dist_new<dist:
#         dist=dist_new
#         p=[k1,k2,k3,k4,d1,d2,d3,d4]  
    


# In[339]:


# #determine parameters for slow-change line
# St=1.631810119329212
# Ht=2.003823690222278
# conc=[1,2,3,4,5]
# dist=np.inf

# for i in range(0,10000):
#     if i%100==0:
#         print(i)
#     k3,k4=np.random.uniform(0,-0.9,2)
#     d3=St025-0.25*k1
#     d4=Ht025-0.25*k2
#     k1,k2,d1,d2=0,0,0,0
#     parameters=[k1,k2,k3,k4,d1,d2,d3,d4]
#     ratio=[]
    
#     for D in conc:
#         St,Ht=piece_wise(D,parameters)
#         if St<=0 or Ht<=0:
#             ratio=np.array([np.inf]*len(conc))
#             break
            
#         S,H=solver(D,parameters)
#         ratio.append(mode1_ratio(D,S,H))
#     dist_new=sum(np.abs(np.array(ratio)-np.array([0.65,0.48,0.46,0.44,0.25])))
    
#     if dist_new<dist:
#         dist=dist_new
#         p=[k1,k2,k3,k4,d1,d2,d3,d4]


# In[ ]:


#0.02: 1.6578211,2.164136903366279


# In[432]:


#fit with non-linear curve
conc=[0.02, 0.1, 2]
ratio=[0.99, 0.82, 0.48]
p=[]
for i in range(100000):
    if i%1000==0:
        print(i)
    fit_ratio=[]
    a_s,a_h = np.random.uniform(0,0.3,2)
    Ks,Kh = np.random.uniform(0,0.3,2)
    bs,bh = np.random.uniform(0,0.3,2)
    cs,ch = np.random.uniform(0,2,2)
    for D in conc:
        St = a_s*Ks**3/(Ks**3+D**3)-bs*D+cs
        Ht = a_h*Kh**3/(Kh**3+D**3)-bh*D+ch
        if St<=0 or Ht<=0:
            fit_ratio=np.array([np.inf]*len(conc))
            break
        S,H = solver_v2(St,Ht)
        fit_ratio.append(mode1_ratio(D,S,H))
    dist = sum(np.abs(np.array(fit_ratio)-np.array(ratio))*np.array([1,1,0.5,1]))
    if dist<0.2:
        p.extend([a_s,Ks,bs,cs,a_h,Kh,bh,ch])


# In[ ]:




