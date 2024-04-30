#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from sklearn.linear_model import LinearRegression
from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from yellowbrick.cluster import KElbowVisualizer
from numba import jit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from bokeh.io import output_notebook, show, export_png, export_svgs
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Whisker, HoverTool,Label,Arrow, NormalHead, OpenHead, VeeHead
from bokeh.palettes import magma
from bokeh.layouts import *
import pickle
import os.path  
import scipy.io as io
from scipy.stats import ttest_ind
import scipy.stats as stats 
from sklearn.decomposition import PCA
import bokeh.io
# import bokeh.layouts
import bokeh.models
from bokeh.models import Range1d,LinearAxis
import bokeh.plotting
notebook_url = "localhost:8891"
bokeh.io.output_notebook()


# In[6]:


global T,groups,initial,mu1,sigma1,mu2,sigma2
T=100
dt=0.1
groups=100
noise=0.28
switch_fre=0
parameters=2
initial=[0.49,0.86]
mu1,sigma1,mu2,sigma2 = 0,0.05,0,0.05

def gly_ode(S,H,glucose):
    D = glucose
    St,Ht = piecewise_glucose(D)
#     St,Ht = glucose_arbitrary(D)
    #2x sir2
#     St=1.2*St
    gamma2 = 1 #gamma2=0.6 when 2x Sir2; gamma2=1 when 1x Sir2
    
    dSdt = 0.01*(St-S) + S**3/(0.52**3+S**3)*(St-S) - (0.1+H)*S
    dHdt = 0.01*(Ht-H) + H**3/(0.62**3+H**3)*(Ht-H) - (0.3+gamma2*S)*H
#     St1,St2,Ht1,Ht2=1.5,1.5,2,1.75
#     dSdt = (0.01*(St1-S) + S**3/(0.52**3+S**3)*(St1-S) - (0.1+H)*S + 0.01*(St2-S) + S**3/(0.52**3+S**3)*(St2-S) - (0.1+H)*S)/2
#     dHdt = (0.01*(Ht1-H) + H**3/(0.62**3+H**3)*(Ht1-H) - (0.3+S)*H+0.01*(Ht2-H) + H**3/(0.62**3+H**3)*(Ht2-H) - (0.3+S)*H)/2
    return np.array([dSdt, dHdt])

def solver(S0,H0,T,glucose,switch_fre,noise):
    noise=noise
    dt = 0.1
    N = int(T/dt)
    ts = np.linspace(0,T,N)
    samples = np.size(np.array(S0))
    S = np.zeros((N,samples))
    H = np.zeros((N,samples))
    p = glucose
    S[0,:] = S0 #each row: samples; column: timeline
    H[0,:] = H0  
    S_rand=np.random.randn(N, samples)
    H_rand=np.random.randn(N, samples)
    if switch_fre==0: #no glucose oscillation
        D=np.ones(N)*p
    else:
        D_pulse=np.sin(switch_fre*ts*2*3.14) #switch_fre: times/h
        #sine wave to square signal
        D_pulse[D_pulse>=0]=0.02
        D_pulse[D_pulse<0]=2
        D=D_pulse
    
    for i in range(N-1):    
        S[i,S[i,:]<0] =0
        H[i,H[i,:]<0] =0
        dS,dH = gly_ode(S[i,:],H[i,:],D[i])*dt
#         S[i+1,:] = S[i,:] + dS + noise*(S_rand[i,:]-0.5)*S[i,:]*dt
#         H[i+1,:] = H[i,:] + dH + noise*(H_rand[i,:]-0.5)*H[i,:]*dt    
        S[i+1,:] = S[i,:] + dS + noise*S_rand[i,:]*S[i,:]*dt
        H[i+1,:] = H[i,:] + dH + noise*H_rand[i,:]*H[i,:]*dt 

    return S,H

def nullcline(Slim,Hlim,parameters):
    p=parameters
    points=250
    Smin=Slim[0]
    Smax=Slim[1]
    Hmin=Hlim[0]
    Hmax=Hlim[1]
    s=np.linspace(Smin,Smax,points)
    h=np.linspace(Hmin,Hmax,points)
    Sv,Hv = np.meshgrid(s,h)
    Sn_row,Hn_row = np.meshgrid(s,h)
    Sn_col,Hn_col = np.meshgrid(s,h)
    dSdt,dHdt = gly_ode(Sv,Hv,p)
    for i in range(points):
        Sn_row_sign = np.sign(dSdt[i,:])
        Sn_row_signchange = ((np.roll(Sn_row_sign, 1) - Sn_row_sign) != 0).astype(int)
        Sn_row_signchange[0] = 0
        Sn_row[i,:] = Sn_row_signchange
        
        Hn_row_sign = np.sign(dHdt[i,:])
        Hn_row_signchange = ((np.roll(Hn_row_sign, 1) - Hn_row_sign) != 0).astype(int)
        Hn_row_signchange[0] = 0
        Hn_row[i,:] = Hn_row_signchange
        
        Sn_col_sign = np.sign(dSdt[:,i])
        Sn_col_signchange = ((np.roll(Sn_col_sign, 1) - Sn_col_sign) != 0).astype(int)
        Sn_col_signchange[0] = 0
        Sn_col[:,i] = Sn_col_signchange
        
        Hn_col_sign = np.sign(dHdt[:,i])
        Hn_col_signchange = ((np.roll(Hn_col_sign, 1) - Hn_col_sign) != 0).astype(int)
        Hn_col_signchange[0] = 0
        Hn_col[:,i] = Hn_col_signchange
    
        Sn = np.logical_or(Sn_row,Sn_col)
        Hn = np.logical_or(Hn_row,Hn_col)
    
    S_p = np.where(Sn==1)
    H_p = np.where(Hn==1)
    S_null_y = S_p[0]/points*Hmax
    S_null_x = S_p[1]/points*Smax
    S_null = np.array([S_null_x,S_null_y])
    H_null_y = H_p[0]/points*Hmax
    H_null_x = H_p[1]/points*Smax
    H_null = np.array([H_null_x,H_null_y])
    return S_null, H_null

def mode1_ratio(H,group):
    return (group-np.sum(H[:,-1]<0.2))/group

def piecewise_glucose(glucose_conc):
    D=glucose_conc    
    St = 0.03**3/(0.03**3+D**3)*0.0515-0.02036*D+1.6207
    Ht = 0.04**3/(0.04**3+D**3)*0.126-0.06*D+1.999
#     Ht = 0.05**3/(0.05**3+D**3)*0.13-0.055*D+1.99
    return St,Ht


@jit(nopython=True)
def find_first(item, vec, N):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if vec[i]>=item:
            return i
    return N

def survival(LS,points):
    samples=len(LS)
    LS=np.array(LS) 
    survival_ratio=[]
    for i in range(points):
        survival_ratio.append(sum(LS>i*0.1)/samples)
    return survival_ratio

"""Bifurcation analysis"""
def test_ode(x,func,arg):
    St,Ht = func
    a1,K1,d1,a2,K2,d2 = arg
    gamma2 = 0.6 #gamma2=0.6 when 2x Sir2; gamma2=1 when 1x Sir2
    S = x[0]
    H = x[1]
    dSdt = a1*(St-S) + S**3/(K1**3+S**3)*(St-S) - (d1+H)*S
    dHdt = a2*(Ht-H) + H**3/(K2**3+H**3)*(Ht-H) - (d2+gamma2*S)*H
    
    return dSdt, dHdt

def test_CR_func(K1,K2,glucose):
    St = 0.6*K1/(K1+glucose)+1.2
    Ht = 0.7*K2/(K2+glucose)+1.6
    return St,Ht 

def test_const_func(St,Ht):
    return St,Ht

def piecewise_glucose_s(D):
    St,Ht=[],[]
    kappa = 1 #kappa=1.2 when 2x Sir2
    for i,g in enumerate(D):
        S,H = piecewise_glucose(g)
        St.append(kappa*S)
        Ht.append(H)
    return St,Ht

def piecewise_glucose_s2(D):
    St,Ht=[],[]
    for i,g in enumerate(D):
        S = 0.2016*g**2-0.1648*g+1.4432
        H = 0.2583*g**2-0.8349*g+2.4166
        St.append(S)
        Ht.append(H)
    return St,Ht

def piecewise_glucose_OE(D):
    St,Ht=[],[]
    r_S=1.2
    r_H=1
    for i,g in enumerate(D):
        S,H = piecewise_glucose(g)
        St.append(r_S*S)
        Ht.append(r_H*H)
    return St,Ht

def find_fixed_points(St,Ht,arg,local):
    fixed_points=[];
    results=[];
    S_guess_lim = [0,St]
    H_guess_lim = [0,Ht]
    for S in np.linspace(S_guess_lim[0],S_guess_lim[1],5):
        for H in np.linspace(H_guess_lim[0],H_guess_lim[1],5):
            guess = [S,H] #S,H
            y=fsolve(test_ode,guess,(test_const_func(St,Ht),arg),full_output=True)
            #test if within guess limit
            if local==1:
                if St*0.15<=y[0][0]<=St*0.7 and Ht*0.15<=y[0][1]<=Ht*0.7 and y[3]=='The solution converged.':
                    m=[round(y[0][0],2),round(y[0][1],2)]
                    results.append(m)
            elif local==0:
                if y[0][0]>=0 and y[0][1]>=0 and y[3]=='The solution converged.':
                    m=[round(y[0][0],2),round(y[0][1],2)]
                    results.append(m)
               
    for i in results:
        if i not in fixed_points: #delete same outcomes   
            fixed_points.append(i)
    return fixed_points

def gly_eigen_value(S_fix,H_fix,St,Ht,arg):
    S,H,St,Ht = S_fix,H_fix,St,Ht
    a1,K1,d1,a2,K2,d2 = arg
    S_dmt = K1**3+S**3
    H_dmt = K2**3+H**3
    E11 = -a1+3*K1**3*S**2/S_dmt**2*(St-S)-S**3/S_dmt-d1-H
    E12 = -S
    E21 = -H
    E22 = -a2+3*K2**3*H**2/H_dmt**2*(Ht-H)-H**3/H_dmt-d2-S
    trace = E11+E22
    det = E11*E22-E12*E21
    L1 = (trace+np.sqrt(trace**2-4*det))/2
    L2 = (trace-np.sqrt(trace**2-4*det))/2
    return L1,L2

def stability(fixed_points,St,Ht,arg):
    stable_points=[]
    untable_points=[]
    if fixed_points:
        for i in range(len(fixed_points)):
            S_fix,H_fix = fixed_points[i]
            L1,L2 = gly_eigen_value(S_fix,H_fix,St,Ht,arg)
            if L1.real<0 and L2.real<0:
                stable_points.append(fixed_points[i])
            else:
                untable_points.append(fixed_points[i])
    return stable_points,untable_points


"""calculate damage"""
def damage_ode(F,S,H,t):
    dFdt = 0.005*t-(S**2/(S**2+0.41**2)*H**2/(H**2+0.72**2))*F    
    return dFdt

def damage_solver(T,glucose,switch_fre):
    dt = 0.1
    N = int(T/dt)
    ts = np.linspace(0,T,N)
    groups=200
    switch_fre=switch_fre
    D=glucose
    s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
    h_init=np.random.normal(mu2,sigma2,groups)+initial[1]
    traj=solver(s_init,h_init,T,D,switch_fre,noise)
    S=traj[0]
    H=traj[1]
    F = np.zeros((N,groups))
    F_rand=np.random.randn(N, groups)
    for i in range(N-1):
        F[i,F[i,:]<0] =0
        dF = damage_ode(F[i,:],S[i,:],H[i,:],ts[i])*dt + noise*F_rand[i,:]*F[i,:]*dt
        F[i+1,:] = F[i,:]+dF  
        
    #calculate living_time
    threshold = 5
    living_time=[]
    for j in range(groups):
        living_time.append(find_first(threshold,F[:,j],N)*dt)
    
    return F,S,H,living_time

def lifespan(x,y,T): #sir2 trace;hap trace;total time
    lifetime=[]
    num,time=np.shape(x)
    for i in range(num):
        for j in range(time):
            if x[i,j]>1.4 and y[i,j]<0.15: #mode2
                lifetime.append(j)
                break
            elif x[i,j]<0.15 and y[i,j]>1.4: #mode1
                lifetime.append(j)
                break
            elif j==time-1:
                lifetime.append(T*10) #cant to go to death point
    return lifetime

"""mode2 statistics"""
def find_mode2_time(H_vec): #hap single trace
    for i in range(len(H_vec)):
        if H_vec[i]<=0.1:
            return i
    return 0

def find_mode1_time(S_vec): #sir2 single trace
    for i in range(len(S_vec)):
        if S_vec[i]<=0.05:
            return i
    return 0

def sort_mode2_occur_time(H):
    ID = H[-1,:]<0.1
    if not any(ID): #no mode2 cells
        return []
    else:
        mode2_vec = H[:,ID]
        trace,mode2_num = np.shape(mode2_vec)
        sort_mode2_occur_time = []
        for i in range(0,mode2_num):
            vec = mode2_vec[:,i]
            mode2_time = find_mode2_time(vec)
            sort_mode2_occur_time.append(mode2_time)  
        sort_mode2_occur_time.sort()
        return sort_mode2_occur_time
    
def sort_mode1_occur_time(S):
    ID = S[-1,:]<0.05
    if not any(ID): #no mode2 cells
        return []
    else:
        mode1_vec = S[:,ID]
        trace,mode1_num = np.shape(mode1_vec)
        sort_mode1_occur_time = []
        for i in range(0,mode1_num):
            vec = mode1_vec[:,i]
            mode1_time = find_mode1_time(vec)
            sort_mode1_occur_time.append(mode1_time)  
        sort_mode1_occur_time.sort()
        return sort_mode1_occur_time

def mode_seperator(s,h):
    mode1_ID=np.where(s[-1,:]<0.05)[0]
    mode2_ID=np.where(h[-1,:]<0.1)[0]
    modeS_ID=[]
    time_points,groups = np.shape(h)
    for i in mode1_ID:
        if find_mode1_time(s[:,i])>=188:
            modeS_ID.append(i)
    for j in mode2_ID:
        if find_mode2_time(h[:,j])>=200:
            modeS_ID.append(j)
    for k in range(groups):
        if s[-1,k]>0.1 and h[-1,k]>0.2:
            modeS_ID.append(k)
    return mode1_ID,mode2_ID,modeS_ID
    

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def generate_parameters(param_value,param):
    parameters=[]
    for i in range(len(param)):
        parameters.append(np.random.uniform(param_value[param[i]]["min"],param_value[param[i]]["max"]))
    return np.array(parameters)


# In[8]:


#build the plot

s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
h_init=np.random.normal(mu2,sigma2,groups)+initial[1]


s,h=nullcline([0,2],[0,2],parameters)
s_x=s[0] 
s_y=s[1]
h_x=h[0]
h_y=h[1]

traj=solver(s_init,h_init,T,parameters,switch_fre,noise)
xs=traj[0].transpose()
ys=traj[1].transpose()
ratio=mode1_ratio(ys,groups)
Stt=xs.tolist()
Htt=ys.tolist()
c=np.shape(Stt)
ts=np.linspace(0,T,c[1])
xs=np.vstack([ts]*c[0])
x = xs.tolist()

cds1 = bokeh.models.ColumnDataSource(dict(s_x=s_x, s_y=s_y))
cds2 = bokeh.models.ColumnDataSource(dict(h_x=h_x, h_y=h_y))
cds3 = bokeh.models.ColumnDataSource(dict(xs=x,Stt=Stt, Htt=Htt))
cds4 = bokeh.models.ColumnDataSource(dict(y=[1,2],right=[ratio,1]))
# cds5 = bokeh.models.ColumnDataSource(dict(s_init=s_init[0],h_init=h_init[0]))
cds5 = bokeh.models.ColumnDataSource(dict(s_init=s_init,h_init=h_init))

p = bokeh.plotting.figure(
    frame_width=250,
    frame_height=250,
    x_axis_label="S",
    y_axis_label="H",
)

p2 = bokeh.plotting.figure(
    frame_width=250,
    frame_height=250,
    x_axis_label="t",
    y_axis_label="H",
)

p3 = bokeh.plotting.figure(
    frame_width=250,
    frame_height=250,
    x_axis_label="t",
    y_axis_label="S",
)

p4 = bokeh.plotting.figure(
    frame_width=250,
    frame_height=30,
    y_axis_label="r"
)

p.circle(source=cds1,x="s_x",y="s_y",size=1.5,fill_color="#2E86C1",line_color="#2E86C1",legend_label="S")
p.circle(source=cds2,x="h_x",y="h_y",size=1.5,fill_color="#B03A2E",line_color="#B03A2E",legend_label="H")
# p.multi_line(source=cds3,xs="St",ys="Ht",line_color="grey",line_alpha=0.6)
p.circle(source=cds5,x="s_init",y="h_init",size=1,fill_color="grey")
p2.multi_line(source=cds3,xs="xs",ys="Htt",line_color="red",legend_label="H")
p3.multi_line(source=cds3,xs="xs",ys="Stt",line_color="green",legend_label="S")
p4.hbar(source=cds4,y="y",right="right")
p4.x_range = Range1d(0, 1)

#Build the widgets
width=180

D_slider = bokeh.models.Slider(
    title="D", start=0.02, end=5, step=0.01, value=parameters, width=width
)

switch_fre_slider = bokeh.models.Slider(
    title="switch_fre", start=0, end=2, step=0.01, value=0, width=width
)


#callback

def H_S_glucose_callback(attr,old,new):
    T=100
    groups=60
    parameters = D_slider.value
    noise=0.3

    s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
    h_init=np.random.normal(mu2,sigma2,groups)+initial[1]
    
    s,h=nullcline([0,2],[0,2],parameters)
    s_x=s[0] 
    s_y=s[1]
    h_x=h[0]
    h_y=h[1]

    traj=solver(s_init,h_init,T,parameters,switch_fre_slider.value,noise)
    xs=traj[0].transpose()
    ys=traj[1].transpose()
    ratio=mode1_ratio(ys,groups)
    Stt=xs.tolist()
    Htt=ys.tolist()
    c=np.shape(Stt)
    ts=np.linspace(0,T,c[1])
    xs=np.vstack([ts]*c[0])
    x=xs.tolist()
    
    cds1.data = dict(s_x=s_x, s_y=s_y)
    cds2.data = dict(h_x=h_x, h_y=h_y)
    cds3.data = dict(xs=x,Stt=Stt, Htt=Htt)
    cds4.data = dict(y=[1,2],right=[ratio,1])
    cds5.data = dict(s_init=s_init,h_init=h_init)
    
for slider in [D_slider,
               switch_fre_slider,]:
    slider.on_change("value", H_S_glucose_callback)
    
# p.x_range.on_change("end", H_S_glucose_callback)

# layout
layout = bokeh.layouts.column(
    bokeh.layouts.row(
        bokeh.layouts.column(p,),
        bokeh.layouts.column(bokeh.layouts.row(p4,),
                             bokeh.layouts.row(p2,)),
        bokeh.layouts.column(p3,),
    ),
    
    bokeh.layouts.column(
        bokeh.layouts.row(
            D_slider,switch_fre_slider,
        ),
    ),
    
)

def app(doc):
    doc.add_root(layout)
bokeh.io.show(app, notebook_url=notebook_url)


# In[436]:


D=np.linspace(0.02,5,50)
switch_fre=0
glu_ratio=[]
# tunable parameters
mu,sigma = 0,0.05
for glu in D:
    for j in range(30): #repeat experiment for 30 times 
        parameters=glu
        s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
        h_init=np.random.normal(mu2,sigma2,groups)+initial[1]
        traj=solver(s_init,h_init,T,parameters,0,noise)
        ys=traj[1].transpose()
        ratio=mode1_ratio(ys,groups)
        glu_ratio.append([glu,ratio])
# pickle.dump(glu_ratio, open(".\glu_ratio.p", "wb+"))


# In[437]:


#plot ratio
# glu_ratio=pickle.load(open(".\glu_ratio.p", "rb"))
D=np.linspace(0.02,5,50)
a=np.array(glu_ratio)
r_mean=[]
r_std=[]
pp=figure(width=450, height=400,x_axis_label="glucose concentration",y_axis_label="mode1%")
for i in D:
    r=a[a[:,0]==i,1]
    r_mean.append(np.mean(r))
    r_std.append(np.std(r))
pp.varea(x=D,
        y1=np.array(r_mean)+np.array(r_std),
        y2=np.array(r_mean)-np.array(r_std),
        color="#CAB2D6",
        alpha=0.5)
pp.line(x=D,y=r_mean,color="red",line_width=2)  

#data from experiment
glu=[  0.02,0.1, 0.25, 0.5, 1,  2,   3,   4,   5]
ratio=[0.96,0.82,0.68,0.67,0.65,0.48,0.46,0.43,0.25]
pp.line(x=glu,y=ratio,color="black",line_width=2)
pp.circle(x=glu,y=ratio,fill_color="black",line_color="white",size=8)

show(pp)
#save as svg for AI
# pp.output_backend = "svg"
# export_svgs(pp, filename="ratio.svg")


# In[29]:


"""plot trace and damage and lifespan"""
T=100
f,s,h,LS=damage_solver(T,0.02,0) #damage,sir2,hap4,lifetime
N = int(T/dt)
ts = np.linspace(0,T,N)
points,samples=np.shape(f)
F=f.transpose()
S=s.transpose()
H=h.transpose()
F=F.tolist()
S=S.tolist()
H=H.tolist()
t=np.vstack([ts]*samples)
t=t.tolist()
colors=[]
for i in range(samples):
    colors.append([255-LS[i]/T*255,255-LS[i]/T*255,255-LS[i]/T*255])
    
survival_ratio = survival(LS,N)


# In[30]:


"""plot traj and damage"""
f4=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Level(A.U.)",x_range=(0,T),y_range=(0,2))
f4.multi_line(xs=t,ys=S)
f4.grid.visible = False

f5=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Level(A.U.)",x_range=(0,T),y_range=(0,2.2))
f5.multi_line(xs=t,ys=H,line_color="red")
f5.grid.visible = False

f6=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Damage(A.U.)",
          y_range=(0,10),
          x_range=(0,T))    
f6.multi_line(t,F,color=colors)
f6.line(x=[0,T],y=[5,5],line_width=1,color=[0,0,0])
citation = Label(x=2, y=5.1, 
                 text='Threshold', text_font_size='10pt',
                 background_fill_color='white', background_fill_alpha=1.0)
f6.add_layout(citation)
f6.grid.visible = False

# f7=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Fraction Viable",x_range=(0,T))
# f7.line(x=ts,y=survival_ratio,line_width=2,color="black")
# f7.grid.visible = False

layout=bokeh.layouts.row(
    bokeh.layouts.column(
        bokeh.layouts.row(f4),
        bokeh.layouts.row(f5),
    ),
    bokeh.layouts.column(
        bokeh.layouts.row(f6),
#         bokeh.layouts.row(f7),
    ),
)
show(layout)


# In[31]:


f4.output_backend = "svg"
export_svgs(f4, filename="./V2/traj/D002_sir2.svg")
f5.output_backend = "svg"
export_svgs(f5, filename="./V2/traj/D002_hap4.svg")
f6.output_backend = "svg"
export_svgs(f6, filename="./V2/traj/D002_damage.svg")
# f7.output_backend = "svg"
# export_svgs(f7, filename="D2_survival.svg")


# In[375]:


glucose=[0.02,0.1,2]
survival_ratio=np.zeros([3,N])
for i,D in enumerate(glucose):
    f,s,h,LS=damage_solver(110,D,0)
    SR = survival(LS,N)
    survival_ratio[i,:]=np.array(SR)


# In[376]:


#plot survival curve for glucose
f8=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Fraction Viable",x_range=(20,90))
f8.line(x=ts,y=survival_ratio[0,:],color="#CB4335",line_width=2,legend_label="0.02% glucose")
f8.line(x=ts,y=survival_ratio[1,:],color="#3498DB",line_width=2,legend_label="0.1% glucose")
f8.line(x=ts,y=survival_ratio[2,:],color="black",line_width=2,legend_label="2% glucose")
f8.grid.visible = False
show(f8)


# In[9]:


f8.output_backend = "svg"
export_svgs(f8, filename="curvival_curve_glucose_2.svg")


# In[377]:


glucose=[0.02,2]
survival_ratio=np.zeros([3,N])
for i,D in enumerate(glucose):
    f,s,h,LS=damage_solver(T,D,0)
    SR = survival(LS,N)
    survival_ratio[i,:]=np.array(SR)
f,s,h,LS=damage_solver(T,2,0.6)
SR = survival(LS,N)
survival_ratio[2,:]=np.array(SR)


# In[378]:


#plot survival curve for switch
f9=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Fraction Viable",x_range=(20,90))
f9.line(x=ts,y=survival_ratio[0,:],color="#CB4335",line_width=2,legend_label="0.02% glucose")
f9.line(x=ts,y=survival_ratio[2,:],color="#27AE60",line_width=2,legend_label="glucose switch")
f9.line(x=ts,y=survival_ratio[1,:],color="black",line_width=2,legend_label="2% glucose")
f9.grid.visible = False
show(f9)


# In[191]:


f9.output_backend = "svg"
export_svgs(f9, filename="curvival_curve_switch.svg")


# In[6]:


glucose=[0.02,0.1,0.5,1,2,3,4,5]
mean, lower, upper,mLS = [], [], [],[]
for D in glucose:
    for i in range(10): #experiment for 10 times
        f,s,h,LS=damage_solver(110,D,0)
        mLS.append(np.mean(LS))
    mean.append(np.mean(mLS))
    lower.append(np.mean(mLS)-np.std(mLS))
    upper.append(np.mean(mLS)+np.std(mLS))
    mLS=[]


# In[12]:


#plot average lifespan for glucose
f1=figure(width=400, height=300,x_axis_label="glucose concentration",
          y_axis_label="Average lifespan(A.U.)",
          title="lifespan vs glucose conc",
          y_range=(28.5,35.5),
#           y_range=(108.5,109.5),
         )
    
source = ColumnDataSource(data=dict(x=glucose, lower=lower, upper=upper))


error=Whisker(source=source, base="x", upper="upper", lower="lower",line_width=2)
f1.add_layout(error)
error.upper_head.size=8
error.lower_head.size=8
f1.circle(x=glucose,y=mean,color="red",size=4)
f1.line(x=glucose,y=mean,line_width=2)
show(f1)


# In[13]:


f1.output_backend = "svg"
export_svgs(f1, filename="average_lifespan_TSH2.svg")


# In[427]:


frequency=np.linspace(0,1,200)
mean, lower, upper,mLS = [], [], [],[]
for fre in frequency:
    for i in range(3): #experiment for 3 times
        f,s,h,LS=damage_solver(110,2,fre)
        mLS.append(np.mean(LS))
    mean.append(np.mean(mLS))
    lower.append(np.mean(mLS)-np.std(mLS))
    upper.append(np.mean(mLS)+np.std(mLS))
    mLS=[]


# In[428]:


#plot average lifespan for switch
f2=figure(width=400, height=300,
          y_axis_label="Average lifespan(A.U.)",
          title="lifespan vs frequency",
#           y_range=(28,39),
         )
f2.xaxis.axis_label="Frequency (A.U.)"
f2.varea(x=frequency,
        y1=upper,
        y2=lower,
        color="#E74C3C",
        alpha=0.5)

f2.line(x=frequency,y=mean,line_width=2)
show(f2)


# In[23]:


f2.output_backend = "svg"
export_svgs(f2, filename="average_lifespan_frequency_2.svg")


# In[9]:


#plot average lifespan for continuous glucose
mean, lower, upper,mLS = [], [], [],[]
for D in glucose:
    for i in range(5): #experiment for 5 times
        f,s,h,LS=damage_solver(110,D,0)
        mLS.append(np.mean(LS))
    mean.append(np.mean(mLS))
    lower.append(np.mean(mLS)-np.std(mLS))
    upper.append(np.mean(mLS)+np.std(mLS))
    mLS=[]

f3=figure(width=400, height=300,x_axis_label="glucose concentration",
          y_axis_label="Average lifespan(A.U.)",
          title="lifespan vs glucose conc",
#           y_range=(29.5,34.5),
         )
f3.varea(x=glucose,
        y1=upper,
        y2=lower,
        color="#EC7063",
        alpha=0.5)   

f3.line(x=glucose,y=mean,line_width=2,color="#3498DB")

show(f3)


# In[31]:


f3.output_backend = "svg"
export_svgs(f3, filename="average_lifespan_glu_cont.svg")


# In[516]:


"""plot phase plane and traj"""
D=2
T=65
ts=np.linspace(0,T,T*10)
switch_fre=0
groups=10
x_max,y_max = 1.8,1.8
s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
h_init=np.random.normal(mu2,sigma2,groups)+initial[1]
traj=solver(s_init,h_init,T,D,switch_fre,noise)
x=traj[0].transpose()
y=traj[1].transpose()
x=x.tolist()
y=y.tolist()
t=np.vstack([ts]*groups)
s,h=nullcline([0,x_max],[0,y_max],D)
s_x=s[0] 
s_y=s[1]
h_x=h[0]
h_y=h[1]

f10=figure(width=400,height=400,x_axis_label="Sir2",y_axis_label="HAP",x_range=(0,x_max),y_range=(0,y_max))
f10.circle(x=s_x,y=s_y,size=1.5,fill_color="#2E86C1",line_color="#2E86C1",legend_label="Sir2_null")
f10.circle(x=h_x,y=h_y,size=1.5,fill_color="#B03A2E",line_color="#B03A2E",legend_label="HAP_null")
f10.multi_line(xs=x,ys=y,color=magma(groups),alpha=0.3,line_width=1)
f10.circle(x=s_init,y=h_init,fill_color="#8E44AD",line_color="#8E44AD",size=2.5)
f10.grid.visible=False
f10_1=figure(width=400,height=300)
f10_1.multi_line(xs=t.tolist(),ys=y,line_width=2)
show(row(f10,f10_1))


# In[50]:


# glucose_switch_xy=[x,y]
# pickle.dump(glucose_switch_xy, open(".\glucose_switch_xy.p", "wb+"))


# In[517]:


f10.output_backend = "svg"
export_svgs(f10, filename="2glu_phase_traj.svg")


# In[31]:


"""make movie from phase plane for glucose switch"""
x,y=pickle.load(open(".\glucose_switch_xy.p", "rb"))
T=50 #total time
n=5
ts=[]
ts=np.linspace(0,T,T*n) #image for every 60/n min
switch_fre=1/6 #glucose period for 6hrs
groups=10
initial=[0.49,0.86]
x_max,y_max = 1.8,1.8
X,Y = np.linspace(0,x_max,10),np.linspace(0,y_max,10)
arrow_scale=0.04
m=120
# s_init=(np.random.rand(1,groups)-0.5)*0.15+initial[0]
# h_init=(np.random.rand(1,groups)-0.5)*0.15+initial[1]
D=np.sin(switch_fre*ts*2*3.14) #switch_fre: times/h
D[D>=0]=2
D[D<0]=0.02
traj_color=magma(groups)
scale_factor=round(10/n)
for i,t in enumerate(ts):
    x,y=np.array(x),np.array(y)
    xx,yy=x[:,max([0,scale_factor*i-8]):scale_factor*i],y[:,max([0,scale_factor*i-8]):scale_factor*i]
    xx=xx.tolist()
    yy=yy.tolist()
    s,h=nullcline([0,x_max],[0,y_max],D[i])
    s_x=s[0] 
    s_y=s[1]
    h_x=h[0]
    h_y=h[1]
    f17=figure(width=400,height=400,x_axis_label="Sir2",y_axis_label="HAP",x_range=(0,x_max),y_range=(0,y_max))

    for x0 in X:
        for y0 in Y:
            delta=gly_ode(x0,y0,D[i])
            dist=np.sqrt(delta[0]**2+delta[1]**2)
            x1,y1=arrow_scale*delta[0]/dist+x0,arrow_scale*delta[1]/dist+y0
            colors=(255-m*dist**(1/3),255-m*dist**(1/3),255-m*dist**(1/3))
            f17.add_layout(Arrow(end=VeeHead(size=8,fill_color=colors,line_color=colors), 
                             line_color=colors,
                       x_start=x0, y_start=y0, x_end=x1, y_end=y1))
    
    f17.circle(x=s_x,y=s_y,size=1.5,fill_color="#2E86C1",line_color="#2E86C1")
    f17.circle(x=h_x,y=h_y,size=1.5,fill_color="#B03A2E",line_color="#B03A2E")
    f17.line([3],[3],line_color="#2E86C1",legend_label="Sir2 nullcline")
    f17.line([3],[3],line_color="#B03A2E",legend_label="HAP nullcline")
    f17.multi_line(xs=xx,ys=yy,color=traj_color,alpha=0.5,line_width=3)
    f17.circle(x=x[:,scale_factor*i],y=y[:,scale_factor*i],fill_color="#8E44AD",line_color="#8E44AD",size=6)
    f17.grid.visible=False
    

    export_png(f17, filename=".\movie\{}.png".format(i))
#     f17.output_backend = "svg"
#     export_svgs(f17, filename=".\movie\svg{}.png".format(i))


# In[69]:


ts_real=np.linspace(0,T,T*10)
for i,t in enumerate(ts):
    f18=figure(width=400,height=400,x_axis_label="Time (h)",y_axis_label="Sir2",x_range=(0,T),y_range=(0,1.8))
    f19=figure(width=400,height=400,x_axis_label="Time (h)",y_axis_label="HAP",x_range=(0,T),y_range=(0,1.8))
    s,h = x[:,0:max(1,scale_factor*i)],y[:,0:max(1,scale_factor*i)]
    s=s.tolist()
    h=h.tolist()
    tt=np.vstack([ts_real[0:max(1,scale_factor*i)]]*groups)
    tt=tt.tolist()
    f18.multi_line(xs=tt,ys=s,line_width=2,legend_label="Sir2")
    f19.multi_line(xs=tt,ys=h,color="red",line_width=2,legend_label="HAP")
    export_png(f18, filename=".\movie_sir2\{}.png".format(i))
#     export_png(f19, filename=".\movie_hap\{}.png".format(i))


# In[30]:


f11.output_backend = "svg"
export_svgs(f11, filename="D002glu_phase.svg")


# In[520]:


"""calculate points for bifurcation plot"""
S_stability=[]
H_stability=[]
arg=[0.01,0.52,0.1,0.01,0.62,0.3] #parameters for ode
D=np.hstack((np.linspace(0.02,0.5,300),np.linspace(0.5,5,200)))
St,Ht = piecewise_glucose_s(D)
# St,Ht = piecewise_glucose_OE(D)
for i,(s,h) in enumerate(zip(St,Ht)):
    x=find_fixed_points(s,h,arg,0)
    sta,unsta=stability(x,s,h,arg)
    sta=np.array(sta)
    unsta=np.array(unsta)
    S_stability.append([D[i], sta[:,0].tolist(), unsta[:,0].tolist()])
    H_stability.append([D[i], sta[:,1].tolist(), unsta[:,1].tolist()])


# In[17]:


f15=figure(width=300,height=250,x_axis_label="glucose%",y_axis_label="Sir2 level")
f16=figure(width=300,height=250,x_axis_label="glucose%",y_axis_label="HAP level")
f15.grid.visible=False
f16.grid.visible=False
f15.y_range = Range1d(0, 2)
f15.x_range = Range1d(0.0, 5)
f16.y_range = Range1d(0, 1.6)
f16.x_range = Range1d(0.0, 5)
for i,glu in enumerate(D):
    stb1=S_stability[i][1]
    unstb1=S_stability[i][2]
    glu_for_stb1 = [glu]*len(stb1)
    glu_for_unstb1 = [glu]*len(unstb1)
    stb2=H_stability[i][1]
    unstb2=H_stability[i][2]
    glu_for_stb2 = [glu]*len(stb2)
    glu_for_unstb2 = [glu]*len(unstb2)
    f15.circle(glu_for_stb1,stb1,size=1.5,fill_color="black",line_color="black")
    f15.circle(glu_for_unstb1,unstb1,size=1.5,fill_color="grey",line_color="grey")
    f16.circle(glu_for_stb2,stb2,size=1.5,fill_color="black",line_color="black")
    f16.circle(glu_for_unstb2,unstb2,size=1.5,fill_color="grey",line_color="grey")

show(row(f15,f16))


# In[18]:


f15.output_backend = "svg"
export_svgs(f15, filename="2xSir2_bifur_Sir2.svg")
f16.output_backend = "svg"
export_svgs(f16, filename="2xSir2_bifur_HAP.svg")


# In[15]:


"""calculate points for bifurcation plot for 2x Sir2"""
S_stability=[]
H_stability=[]
arg=[0.01,0.52,0.1,0.01,0.62,0.3] #parameters for ode
D=np.hstack((np.linspace(0.02,0.5,300),np.linspace(0.5,5,200)))
St,Ht = piecewise_glucose_OE(D)
for i,(s,h) in enumerate(zip(St,Ht)):
    x=find_fixed_points(s,h,arg,0)
    sta,unsta=stability(x,s,h,arg)
    sta=np.array(sta)
    unsta=np.array(unsta)
    S_stability.append([D[i], sta[:,0].tolist(), unsta[:,0].tolist()])
    H_stability.append([D[i], sta[:,1].tolist(), unsta[:,1].tolist()])
    
f15=figure(width=300,height=250,x_axis_label="glucose%",y_axis_label="Sir2 level")
f16=figure(width=300,height=250,x_axis_label="glucose%",y_axis_label="HAP level")
f15.grid.visible=False
f16.grid.visible=False
f15.y_range = Range1d(0, 2)
f15.x_range = Range1d(0.0, 5)
f16.y_range = Range1d(0, 1.6)
f16.x_range = Range1d(0.0, 5)
for i,glu in enumerate(D):
    stb1=S_stability[i][1]
    unstb1=S_stability[i][2]
    g_for_stb1 = [glu]*len(stb1)
    g_for_unstb1 = [glu]*len(unstb1)
    stb2=H_stability[i][1]
    unstb2=H_stability[i][2]
    g_for_stb2 = [glu]*len(stb2)
    g_for_unstb2 = [glu]*len(unstb2)
    f15.circle(g_for_stb1,stb1,size=1.5,fill_color="black",line_color="black")
    f15.circle(g_for_unstb1,unstb1,size=1.5,fill_color="grey",line_color="grey")
    f16.circle(g_for_stb2,stb2,size=1.5,fill_color="black",line_color="black")
    f16.circle(g_for_unstb2,unstb2,size=1.5,fill_color="grey",line_color="grey")

show(row(f15,f16))


# In[530]:


"""plot Ht,St as a function of D"""
D=np.linspace(0.02,5,30)
St,Ht = piecewise_glucose_s(D)
St_p,Ht_p = piecewise_glucose_s([0.02,0.1,2,5])
f12=figure(width=300,height=250,x_axis_label="St",y_axis_label="Ht")
f12.line(St,Ht,line_width=2,color="black")
f12.circle(St_p,Ht_p,size=3,fill_color="grey",line_color="grey")
f13=figure(width=300,height=250,x_axis_label="D",y_axis_label="level")
f13.line(D,St,line_width=2,legend_label="St")
f13.line(D,Ht,line_width=2,legend_label="Ht",color="red")
show(row(f12,f13))


# In[39]:


f12.output_backend = "svg"
export_svgs(f12, filename="Ht-St.svg")
f13.output_backend = "svg"
export_svgs(f13, filename="StHt_glucose.svg")


# In[548]:


arg=[0.01,0.52,0.1,0.01,0.62,0.3]
Ss=[]
Hs=[]
Su=[]
Hu=[]

for st in np.linspace(1.4,2.6,200):
    for ht in np.linspace(1.4,2.6,200):
        x=find_fixed_points(st,ht,arg,1)
        stable,unstable=stability(x,st,ht,arg)
        if stable:
            Ss.append(st)
            Hs.append(ht)     
        else:
            Su.append(st)
            Hu.append(ht) 
            
stable_points=[Ss,Hs,Su,Hu]   
pickle.dump(stable_points, open(".\sir2OE_model_stable_points.p", "wb+"))
# pickle.dump(stable_points, open(".\model_stable_points_shift.p", "wb+"))


# In[549]:


# Ss,Hs,Su,Hu=pickle.load(open(".\model_stable_points.p", "rb"))
Ss,Hs,Su,Hu=pickle.load(open(".\sir2OE_model_stable_points.p", "rb"))
# Ss,Hs,Su,Hu=pickle.load(open(".\model_stable_points_shift.p", "rb"))
D=np.linspace(0.02,5,100)
StOE,HtOE = piecewise_glucose_OE(D)
St_p,Ht_p = piecewise_glucose_s([0.02,0.1,2,3,5])
# St_p_shift,Ht_p_shift = piecewise_glucose_s2(np.linspace(0.02,2,100))
# St_p_shift_d,Ht_p_shift_d = piecewise_glucose_s2([0.02,2])
f14 = figure(width=400,height=400,x_axis_label="St",y_axis_label="Ht")
f14.grid.visible=False
f14.square(Ss,Hs,size=4,fill_color="#2980B9",line_color="#2980B9")
f14.square(Su,Hu,size=4,fill_color="#F4D03F",line_color="#F4D03F")
# f14.line(St_p,Ht_p,line_width=2,color="darkred")
# f14.line(St_p_shift,Ht_p_shift,line_dash='dashed',line_width=2,color="grey")
f14.line(StOE,HtOE,line_width=2,color="darkred")
f14.circle(St_p,Ht_p,size=4,fill_color="black",line_color="black")
# f14.circle(St_p_shift_d,Ht_p_shift_d,size=4,fill_color="black",line_color="black")
f14.x_range = Range1d(1.34, 2.64)
f14.y_range = Range1d(1.34, 2.64)
show(f14)


# In[550]:


f14.output_backend = "svg"
export_svgs(f14, filename="2xSir2_bifur_phase_w_g.svg")


# In[557]:


"""make movie from phase plane for glucose concentration gradient"""
glu_gradience = np.hstack((np.linspace(5,0.2,70),np.linspace(0.2,0.02,30)))
for i,D in enumerate(glu_gradience):
    St,Ht = piecewise_glucose(D)
    x_max,y_max = 1.8,1.8
    X,Y = np.linspace(0,x_max,10),np.linspace(0,y_max,10)
    scale=0.04
    m=120
    s,h=nullcline([0,x_max],[0,y_max],D)
    s_x=s[0] 
    s_y=s[1]
    h_x=h[0]
    h_y=h[1]

    arg=[0.01,0.52,0.1,0.01,0.62,0.3] #parameters for ode
    x0=find_fixed_points(St,Ht,arg,0)
    sta,unsta=stability(x0,St,Ht,arg)
    sta=np.array(sta)
    unsta=np.array(unsta)

    f19=figure(width=400,height=400,
           x_axis_label="Sir2",y_axis_label="HAP",
           x_range=(0,x_max),y_range=(0,y_max))
    f19.title.text = 'glucose concentration: '+ '{0:.3f}'.format(D) +'%'

    for x0 in X:
        for y0 in Y:
            delta=gly_ode(x0,y0,D)
            dist=np.sqrt(delta[0]**2+delta[1]**2)
            x1,y1=scale*delta[0]/dist+x0,scale*delta[1]/dist+y0
            color=(255-m*dist**(1/3),255-m*dist**(1/3),255-m*dist**(1/3))
            f19.add_layout(Arrow(end=VeeHead(size=8,fill_color=color,line_color=color), 
                             line_color=color,
                   x_start=x0, y_start=y0, x_end=x1, y_end=y1))
    f19.circle(x=s_x,y=s_y,size=1.5,fill_color="#2E86C1",line_color="#2E86C1")
    f19.circle(x=h_x,y=h_y,size=1.5,fill_color="#B03A2E",line_color="#B03A2E")
    f19.circle(x=sta[:,0].tolist(),y=sta[:,1].tolist(),size=9,fill_color="black",line_color="black")
    f19.circle(x=unsta[:,0].tolist(),y=unsta[:,1].tolist(),size=8,fill_color="white",line_color="black")
    f19.grid.visible=False
    export_png(f19, filename=".\movie_phase_glu_grad\{}.png".format(i))


# In[8]:


"""polt quiver and phase plane for 1xSir2"""
D=2
x_max,y_max = 1.8,1.8
X,Y = np.linspace(0,x_max,10),np.linspace(0,y_max,10)
#start points matrix
xline=np.linspace(0,x_max,10).T
yline=np.linspace(0,y_max,10).T
x_start,y_start=[],[]
W,V = np.meshgrid(X,Y)
x_start,y_start = W.flatten(),V.flatten()
traj=solver(x_start,y_start,T,D,switch_fre,0)
x=traj[0].transpose()
y=traj[1].transpose()
x=x.tolist()
y=y.tolist()

scale=0.04
m=120

s,h=nullcline([0,x_max],[0,y_max],D)
s_x=s[0] 
s_y=s[1]
h_x=h[0]
h_y=h[1]


f11=figure(width=400,height=400,
           x_axis_label="Sir2",y_axis_label="HAP",
           x_range=(0,x_max),y_range=(0,y_max))

for x0 in X:
    for y0 in Y:
        delta=gly_ode(x0,y0,D)
        dist=np.sqrt(delta[0]**2+delta[1]**2)
        x1,y1=scale*delta[0]/dist+x0,scale*delta[1]/dist+y0
        color=(255-m*dist**(1/3),255-m*dist**(1/3),255-m*dist**(1/3))
        f11.add_layout(Arrow(end=VeeHead(size=8,fill_color=color,line_color=color), 
                             line_color=color,
                   x_start=x0, y_start=y0, x_end=x1, y_end=y1))
f11.circle(x=s_x,y=s_y,size=1.5,fill_color="#2E86C1",line_color="#2E86C1",legend_label="Sir2 nullcline")
f11.circle(x=h_x,y=h_y,size=1.5,fill_color="#B03A2E",line_color="#B03A2E",legend_label="HAP nullcline")
# f11.multi_line(xs=x,ys=y,color="grey",alpha=0.3)
# f11.circle(x=x_start,y=y_start,fill_color="#8E44AD",line_color="#8E44AD",size=2.5)
f11.grid.visible=False
# f11.title="D1=D2=1.01"
show(f11)


# In[9]:


f11.output_backend="svg"
export_svgs(f11,filename="1xSir2_D2_phase_plane.svg")


# In[5]:


"""phase plane for 2xSir2"""
D=0.02
St,Ht = piecewise_glucose(D)
St=1.2*St
x_max,y_max = 2,1.8
X,Y = np.linspace(0,x_max,10),np.linspace(0,y_max,10)
scale=0.04
m=120
s,h=nullcline([0,x_max],[0,y_max],D)
s_x=s[0] 
s_y=s[1]
h_x=h[0]
h_y=h[1]

arg=[0.01,0.52,0.1,0.01,0.62,0.3] #parameters for ode
x0=find_fixed_points(St,Ht,arg,0)
sta,unsta=stability(x0,St,Ht,arg)
sta=np.array(sta)
unsta=np.array(unsta)

f20=figure(width=400,height=400,
           x_axis_label="Sir2",y_axis_label="HAP",
           x_range=(0,x_max),y_range=(0,y_max))
#     f20.title.text = 'glucose concentration: '+ '{0:.3f}'.format(D) +'%'

for x0 in X:
    for y0 in Y:
        delta=gly_ode(x0,y0,D)
        dist=np.sqrt(delta[0]**2+delta[1]**2)
        x1,y1=scale*delta[0]/dist+x0,scale*delta[1]/dist+y0
        color=(255-m*dist**(1/3),255-m*dist**(1/3),255-m*dist**(1/3))
        f20.add_layout(Arrow(end=VeeHead(size=8,fill_color=color,line_color=color), 
                             line_color=color,
                   x_start=x0, y_start=y0, x_end=x1, y_end=y1))
f20.circle(x=s_x,y=s_y,size=1.5,fill_color="#2E86C1",line_color="#2E86C1")
f20.circle(x=h_x,y=h_y,size=1.5,fill_color="#B03A2E",line_color="#B03A2E")
f20.circle(x=sta[:,0].tolist(),y=sta[:,1].tolist(),size=9,fill_color="black",line_color="black")
f20.circle(x=unsta[:,0].tolist(),y=unsta[:,1].tolist(),size=8,fill_color="white",line_color="black")
f20.grid.visible=False
show(f20)


# In[22]:


f20.output_backend="svg"
export_svgs(f20,filename="2xSir2_D2_phase_plane.svg")


# In[29]:


"""phase plane with traj simulation for 2xSir2"""
D=0.1
T=100
ts=np.linspace(0,T,T*10)
switch_fre=0
groups=10
initial=[0.49*1.2,0.86]
x_max,y_max = 2,1.8
mu1,sigma1,mu2,sigma2 = 0,0.05,0,0.05
s_init=np.random.normal(mu1,sigma1,groups)+initial[0]
h_init=np.random.normal(mu2,sigma2,groups)+initial[1]
St,Ht = piecewise_glucose(D)
St=1.2*St

traj=solver(s_init,h_init,T,D,switch_fre,0.35)
x=traj[0].transpose()
y=traj[1].transpose()
x=x.tolist()
y=y.tolist()
t=np.vstack([ts]*groups)
X,Y = np.linspace(0,x_max,10),np.linspace(0,y_max,10)
scale=0.04
m=120
s,h=nullcline([0,x_max],[0,y_max],D)
s_x=s[0] 
s_y=s[1]
h_x=h[0]
h_y=h[1]

f21=figure(width=400,height=400,
           x_axis_label="Sir2",y_axis_label="HAP",
           x_range=(0,x_max),y_range=(0,y_max))
#     f20.title.text = 'glucose concentration: '+ '{0:.3f}'.format(D) +'%'

for x0 in X:
    for y0 in Y:
        delta=gly_ode(x0,y0,D)
        dist=np.sqrt(delta[0]**2+delta[1]**2)
        x1,y1=scale*delta[0]/dist+x0,scale*delta[1]/dist+y0
        color=(255-m*dist**(1/3),255-m*dist**(1/3),255-m*dist**(1/3))
        f21.add_layout(Arrow(end=VeeHead(size=8,fill_color=color,line_color=color), 
                             line_color=color,
                   x_start=x0, y_start=y0, x_end=x1, y_end=y1))
f21.circle(x=s_x,y=s_y,size=1.5,fill_color="#2E86C1",line_color="#2E86C1")
f21.circle(x=h_x,y=h_y,size=1.5,fill_color="#B03A2E",line_color="#B03A2E")
f21.multi_line(xs=x,ys=y,color=magma(groups),alpha=0.3,line_width=2)
f21.circle(x=s_init,y=h_init,fill_color="#8E44AD",line_color="#8E44AD",size=2.5)
f21.grid.visible=False
show(f21)


# In[30]:


f21.output_backend="svg"
export_svgs(f21,filename="2xSir2_D01_phase_plane_traj.svg")


# In[27]:


"""calculation of ratio of modeS"""
T=100
groups=5000
switch_fre=np.linspace(0.4,0,50)
delayed_mean=[]
delayed_std=[]
for i in switch_fre:
    delayed_ratio_list=[]
    for repeat in range(10): #repeat in 10 times
        s_init=(np.random.rand(1,groups)-0.5)*0.15+initial[0]
        h_init=(np.random.rand(1,groups)-0.5)*0.15+initial[1]
        osci_mode2_occur_time=[]
        S,H = solver(s_init,h_init,T,0.02,i,noise)
        mode1_ID,mode2_ID,modeS_ID=mode_seperator(S,H)

        delayed_ratio = len(modeS_ID)/groups
            
        delayed_ratio_list.append(delayed_ratio)
        
    delayed_mean.append(np.mean(delayed_ratio_list))
    delayed_std.append(np.std(delayed_ratio_list))

delayed_data=[delayed_mean,delayed_std]
pickle.dump(delayed_data, open(".\delayed_data.p", "wb+"))


# In[28]:


glu_ratio=pickle.load(open(".\delayed_data.p", "rb"))
delayed_mean=glu_ratio[0]
delayed_std=glu_ratio[1]
f22=figure(width=400, height=300,x_axis_label="Period",y_axis_label="Delayed_ratio")
f22.xaxis.ticker = [0.4,0.3,0.17,0.2,0.083,0.042,0.1,0.02]
f22.xaxis.major_label_overrides={0.4:'2.5',0.3:'3.3',0.17:'6',0.2:'5',0.083:'12',0.042:'24',0.1:'10',0.02:'50'}
f22.x_range.flipped=True
f22.varea(x=switch_fre,
        y1=np.array(delayed_mean)+np.array(delayed_std),
        y2=np.array(delayed_mean)-np.array(delayed_std),
        color="#CAB2D6",
        alpha=0.5)
f22.line(x=switch_fre,y=delayed_mean,color="red",line_width=2)  
show(f22)


# In[29]:


f22.output_backend="svg"
export_svgs(f22,filename="osci_delayed_ratio.svg")


# In[33]:


"""phase plane with traj simulation for glucose switch"""
D1=0.02
D2=2
T=90
ts=np.linspace(0,T,T*10)
switch_fre=0.174

initial=[0.49,0.86]
x_max,y_max = 1.8,1.8

s1,h1=nullcline([0,x_max],[0,y_max],D1)
s_x1=s1[0] 
s_y1=s1[1]
h_x1=h1[0]
h_y1=h1[1]

s2,h2=nullcline([0,x_max],[0,y_max],D2)
s_x2=s2[0] 
s_y2=s2[1]
h_x2=h2[0]
h_y2=h2[1]

X,Y = np.linspace(0,x_max,10),np.linspace(0,y_max,10)
# W,V = np.meshgrid(X,Y)
# x_start,y_start = W.flatten(),V.flatten()
x_start=[0.55,0.45,0.8]
y_start=[0.9,1.05,0.55]
traj=solver(x_start,y_start,T,D1,switch_fre,0.3)
x=traj[0].transpose()
y=traj[1].transpose()
n=lifespan(x,y,T)
colors=[]
for i in range(len(n)):
    colors.append((rgb_to_hex(255,255-int(n[i]/900*255),255-int(n[i]/900*255))))
tuple(colors)

x=x.tolist()
y=y.tolist()

f23=figure(width=400,height=400,
           x_axis_label="Sir2",y_axis_label="HAP",
           x_range=(0,x_max),y_range=(0,y_max))


f23.circle(x=s_x1,y=s_y1,size=1.5,fill_color="#21618C",line_color="#21618C")
f23.circle(x=h_x1,y=h_y1,size=1.5,fill_color="#943126",line_color="#943126")
f23.circle(x=s_x2,y=s_y2,size=1.5,fill_color="#5DADE2",line_color="#5DADE2")
f23.circle(x=h_x2,y=h_y2,size=1.5,fill_color="#EC7063",line_color="#EC7063")
f23.multi_line(xs=x,ys=y,color=("grey","red","blue"),alpha=1,line_width=1)
f23.circle(x=x_start,y=y_start,fill_color="#8E44AD",line_color="#8E44AD",size=2.5)
f23.grid.visible=False
show(f23)


# In[35]:


f23.output_backend="svg"
export_svgs(f23,filename="osci_phase_plane_noise.svg")


# In[50]:


"""plot glucose switch time trace"""
T=100 #total time
frequency=0.37#0.2
f,s,h,LS=damage_solver(T,2,frequency) #damage,sir2,hap4,lifetime
dt = 0.1
N = int(T/dt)
ts = np.linspace(0,T,N)
points,samples=np.shape(f)
F=f.transpose()
S=s.transpose()
H=h.transpose()
F=F.tolist()
S=S.tolist()
H=H.tolist()
t=np.vstack([ts]*samples)
t=t.tolist()


f25=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Sir2 Level(A.U.)",x_range=(0,T),y_range=(0,2))
f25.multi_line(xs=t,ys=S)

f26=figure(width=400,height=300,x_axis_label="Time",y_axis_label="HAP Level(A.U.)",x_range=(0,T),y_range=(0,2.1))
f26.multi_line(xs=t,ys=H,line_color="red")

# f27=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Damage(A.U.)",
#           y_range=(0,10),
#           x_range=(0,T))    
# f27.multi_line(t,F)
show(row(f25,f26))


# In[45]:


mode1_ID,mode2_ID,modeS_ID=mode_seperator(s,h)
colors=[]
for i in range(samples):
    colors.append([255-LS[i]/T*255,255-LS[i]/T*255,255-LS[i]/T*255])
    
mode1_S = s[:,mode1_ID].transpose()
mode1_H = h[:,mode1_ID].transpose()
mode2_S = s[:,mode2_ID].transpose()
mode2_H = h[:,mode2_ID].transpose()
modeS_S = s[:,modeS_ID].transpose()
modeS_H = h[:,modeS_ID].transpose()
F = f.transpose()
F=F.tolist()
modeS_F = f[:,modeS_ID].transpose()
mode1_S = mode1_S.tolist()
mode1_H = mode1_H.tolist()
mode2_S = mode2_S.tolist()
mode2_H = mode2_H.tolist()
modeS_S = modeS_S.tolist()
modeS_H = modeS_H.tolist()
modeS_F = modeS_F.tolist()
f29=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Sir2 Level(A.U.)",x_range=(0,T),y_range=(0,2))
f29.multi_line(xs=np.vstack([ts]*len(mode1_S)).tolist(),ys=mode1_S,line_color="#C91C00")
f29.multi_line(xs=np.vstack([ts]*len(mode2_S)).tolist(),ys=mode2_S,line_color="#1F77B4")
f29.multi_line(xs=np.vstack([ts]*len(modeS_S)).tolist(),ys=modeS_S,line_color="#8E44AD")

f30=figure(width=400,height=300,x_axis_label="Time",y_axis_label="HAP Level(A.U.)",x_range=(0,T),y_range=(0,2.2))
f30.multi_line(xs=np.vstack([ts]*len(mode1_H)).tolist(),ys=mode1_H,line_color="#C91C00")
f30.multi_line(xs=np.vstack([ts]*len(mode2_H)).tolist(),ys=mode2_H,line_color="#1F77B4")
f30.multi_line(xs=np.vstack([ts]*len(modeS_H)).tolist(),ys=modeS_H,line_color="#8E44AD")

f31=figure(width=400,height=300,x_axis_label="Time",y_axis_label="Damage(A.U.)",
          y_range=(0,10),
          x_range=(0,T))  
f31.multi_line(xs=np.vstack([ts]*len(F)).tolist(),ys=F,color=colors)
f31.multi_line(xs=np.vstack([ts]*len(modeS_F)).tolist(),ys=modeS_F,color="#8E44AD")
f31.line(x=[0,T],y=[5,5],line_width=1,color=[0,0,0])
citation = Label(x=2, y=5.1, 
                 text='Threshold', text_font_size='10pt',
                 background_fill_color='white', background_fill_alpha=1.0)
f31.add_layout(citation)
show(row(f29,f30,f31))


# In[46]:


f29.output_backend="svg"
export_svgs(f29,filename="bypass_osc_sir2_trace.svg")
f30.output_backend="svg"
export_svgs(f30,filename="bypass_osc_HAP_trace.svg")
f31.output_backend="svg"
export_svgs(f31,filename="bypass_osc_damage_trace.svg")


# In[232]:


"""define ModeS time threshold"""
T=100
groups=20000
switch_fre=0
D=0.02
s_init=(np.random.rand(1,groups)-0.5)*0.15+initial[0]
h_init=(np.random.rand(1,groups)-0.5)*0.15+initial[1]
S1,H1 = solver(s_init,h_init,T,D,switch_fre,noise)
mode2=sort_mode2_occur_time(H1)
mode1=sort_mode1_occur_time(S1)

p1=figure(height=200,width=300,x_range=(15,350))
hist1, edges1 = np.histogram(mode1, density=True, bins=50)
#fit to gamma distribution
x1 = np.linspace (15, 350, 100) 
param1 = stats.gamma.fit(mode1, floc=0)
pdf_fitted_1 = stats.gamma.pdf(x1, *param1)
cdf_fitted_1 = stats.gamma.cdf(x1, *param1)
p1.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
         fill_color="red", line_color="white",
        )
p1.line(x=x1,y=pdf_fitted_1,color="black",line_width=2)
p1.y_range = Range1d(0, .019)
#right y-axis
p1.extra_y_ranges = {"probability": Range1d(start=0, end=1.02)}
p1.add_layout(LinearAxis(y_range_name="probability"), 'right')
p1.line(x1,cdf_fitted_1,color="grey",line_width=2,y_range_name="probability")
#mark 95% point
IC=0.95
IDs=np.where(cdf_fitted_1>=IC)
p1.circle(x=x1[IDs[0][0]],y=IC,y_range_name="probability")
p1.xaxis.major_label_overrides={50:'5',100:'10',150:'15',200:'20',250:'25'}
p1.grid.visible=False
p1.title="mode1(n={2}) a={0:.3f},scale={1:.3f}".format(param1[0],param1[2],len(mode1))


# In[233]:


T=100
groups=20000
switch_fre=0
D=2
initial=[0.49,0.86]
s_init=(np.random.rand(1,groups)-0.5)*0.15+initial[0]
h_init=(np.random.rand(1,groups)-0.5)*0.15+initial[1]
noise=0.32
S1,H1 = solver(s_init,h_init,T,D,switch_fre,noise)
mode2=sort_mode2_occur_time(H1)
mode1=sort_mode1_occur_time(S1)



p2=figure(height=200,width=300,x_range=(25,400))
hist2, edges2 = np.histogram(mode2, density=True, bins=30)
x2 = np.linspace (40, 400, 100) 
param2 = stats.gamma.fit(mode2, floc=0)
pdf_fitted_2 = stats.gamma.pdf(x2, *param2)
cdf_fitted_2 = stats.gamma.cdf(x2, *param2)

p2.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:],
         fill_color="skyblue", line_color="white",alpha=0.7
        )
p2.line(x=x2,y=pdf_fitted_2,color="black",line_width=2)
p2.y_range = Range1d(0, .0119)
#right y-axis
p2.extra_y_ranges = {"probability": Range1d(start=0, end=1.02)}
p2.add_layout(LinearAxis(y_range_name="probability"), 'right')
p2.line(x2,cdf_fitted_2,color="grey",line_width=2,y_range_name="probability")
#mark 95% point
IDs=np.where(cdf_fitted_2>=IC)
p2.circle(x=x2[IDs[0][0]],y=IC,y_range_name="probability")
p2.xaxis.major_label_overrides={50:'5',100:'10',150:'15',200:'20',250:'25',300:'30',350:'35',400:'40'}
p2.grid.visible=False
p2.title="mode2(n={2}) a={0:.3f},scale={1:.3f}".format(param2[0],param2[2],len(mode2))
show(row(p1,p2))


# In[236]:


p1.output_backend="svg"
export_svgs(p1,filename="mode1_gamma.svg")
p2.output_backend="svg"
export_svgs(p2,filename="mode2_gamma.svg")


# In[4]:


piecewise_glucose(0.02)


# In[13]:


stable=[]
unstable=[]
all_points=[]
for i in range(10000):
    args=generate_parameters(param_value,param)
    St=args[6]
    Ht=args[7]
    arg=args[0:6]
    if len(stability(find_fixed_points(St,Ht,arg,0),St,Ht,arg)[1])!=0:
        stable.append(args)
    else:
        unstable.append(args)
    all_points.append(args)
    


# In[17]:


data1=np.array(stable)
data2=np.array(unstable)
data3=np.array(all_points)

pca = PCA(n_components=2)
# result1 = pca.fit_transform(data1)
# result2 = pca.fit_transform(data2)
result3 = pca.fit_transform(np.concatenate((data1,data2)))


# In[18]:


len(data1)


# In[26]:


p_pca = figure(title="2-Component PCA Analysis", x_axis_label="Principal Component 1 (PC1)", y_axis_label="Principal Component 2 (PC2)", width=800, height=600)

# Convert data to ColumnDataSource for each dataset
source1 = ColumnDataSource(data=dict(x=result3[0:len(data1), 0], y=result3[0:len(data1), 1], label=[str(row) for row in data1]))
source2 = ColumnDataSource(data=dict(x=result3[len(data1):, 0], y=result3[len(data1):, 1], label=[str(row) for row in data2]))

# Scatter plots for each dataset
p_pca.scatter(x='x', y='y', size=10, color='grey', alpha=0.3, legend_label='unstable (85%)', source=source2)
p_pca.scatter(x='x', y='y', size=10, color='darkred', alpha=0.5, legend_label='stable (15%)', source=source1)


# Hover tool to display original values on hover
# hover = HoverTool(tooltips=[("Original Values", " @label")])
# p_pca.add_tools(hover)

# Add legend
p_pca.legend.title = '10000 groups:'
p_pca.legend.label_text_font_size = '10pt'

# Show the plot
show(p_pca)


# In[55]:


# D = io.loadmat('E:\\Aging\\20220810_270_3_6_osc\\iRFP_for_sorting.mat')
D = io.loadmat('E:\\Aging\\201iRFPcombined_v2.mat')
# D = io.loadmat('E:\\Aging\\20190301\\60min_norm.mat')
# D = io.loadmat('E:\\Aging\\glu01iRFP.mat')


# In[56]:


traj=D['all_data_traj'][0]
points=[]
for i in range(0,len(traj)):
    points.append(traj[i][0][0])
X=np.array(points)
model = KElbowVisualizer(TimeSeriesKMeans(), k=10,timings=False)
model.fit(X)
plt.savefig("Elbow_clustering.pdf")
model.show()


# In[61]:


# Scale the time series data to have zero mean and unit variance
X_train = TimeSeriesScalerMeanVariance().fit_transform(X)
# Perform time series clustering using k-means
n_cluster = 4
km = TimeSeriesKMeans(n_clusters=n_cluster, metric="dtw")
y_pred = km.fit_predict(X_train)

# Plot the clustered time series data
plt.figure(figsize=(10, 10))

for yi in range(n_cluster):
    plt.subplot(2, 2, yi + 1)
    for xx in X_train[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(np.mean(X_train[y_pred == yi], 0), "r-")
    plt.xlim(0, X_train.shape[1])
    plt.ylim(-4, 4)
    plt.title("Cluster %d" % (yi))

plt.tight_layout()
plt.savefig("clustered_time_series.pdf")
plt.show()


# In[6]:


from platform import python_version
print(python_version())


# In[ ]:




