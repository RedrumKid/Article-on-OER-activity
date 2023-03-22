# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:56:56 2022

@author: Ožbej
"""

import numpy as np
import scipy.integrate as sciint
import scipy.optimize as sciop
import scipy.stats as scist
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import numba
from numba import jit,njit
from numba import prange
import time
import Fourier_analysis_file as ff
import sys
import cmath


    
@jit(nopython=True)
def current_f(c,k,thermo,E):
    k01,E01,kr,k02,E02,k03,E03,k04,E04,kc1,kc2,kad1,kad2,n0,gama,sa,sb,sc,sd,se,sg,sn,alfa1,alfa2,alfa3,alfa4=k
    alfa,f,ah,rho=thermo
    a,b,c,d,e,g,n,Ep,jp=c
    
    j=(k01*np.exp(-alfa1*f*(Ep-E01))*a*np.exp(sa*a)*ah-k01*np.exp((1-alfa1)*f*(Ep-E01))*b*np.exp(sb*b)+k02*(b*np.exp(sb*b)*ah*np.exp(-alfa2*f*(Ep-E02))-c*np.exp(sc*c)*np.exp((1-alfa2)*f*(Ep-E02)))+
        k04*(e*np.exp(se*e)*np.exp(-alfa4*f*(Ep-E04))*ah-g*np.exp(sg*g)*np.exp((1-alfa4)*f*(Ep-E04)))+k03*(d*np.exp(sd*d)*ah*np.exp(-alfa3*f*(Ep-E03))-e*np.exp(se*e)*np.exp((1-alfa3)*f*(Ep-E03)))-kr*a*np.exp(sa*a))
    return -j*96485*n0
    
@jit(nopython=True)
def time_step(x,t,xp,k,thermo,E1,E2,v,dt,amp,frequ,diff,jp,E_exact):
    k01,E01,kr,k02,E02,k03,E03,k04,E04,kc1,kc2,kad1,kad2,n0,gama,sa,sb,sc,sd,se,sg,sn,alfa1,alfa2,alfa3,alfa4=k
    alfa,f,ah,rho=thermo
    a,b,c,d,e,g,n,E,gc=x
    ap,bp,cp,dp,ep,gp,npr,Ep,gcp=xp
    
    f1=k01*np.exp(-alfa1*f*(E-E01))*a*ah*np.exp(sa*a)-k01*np.exp((1-alfa1)*f*(E-E01))*b*np.exp(sb*b)+(a-ap)/dt+0*kr*a*np.exp(sa*a)
    f2=(b-bp)/dt-(k01*np.exp(-alfa1*f*(E-E01))*a*ah*np.exp(sa*a)-k01*np.exp((1-alfa1)*f*(E-E01))*b*np.exp(sb*b))+k02*(b*np.exp(sb*b)*ah*np.exp(-alfa2*f*(E-E02))-c*np.exp(sc*c)*np.exp((1-alfa2)*f*(E-E02)))
    f3=-k02*(b*np.exp(sb*b)*ah*np.exp(-alfa2*f*(E-E02))-c*np.exp(sc*c)*np.exp((1-alfa2)*f*(E-E02)))+(c-cp)/dt-kc1*d*np.exp(sd*d)+kc2*c*np.exp(sc*c)
    
    f4=k03*(d*np.exp(sd*d)*ah*np.exp(-alfa3*f*(E-E03))-e*np.exp(se*e)*np.exp((1-alfa3)*f*(E-E03)))+(d-dp)/dt+kc1*d*np.exp(sd*d)-kc2*c*np.exp(sc*c)
    f5=-k03*(d*np.exp(sd*d)*ah*np.exp(-alfa3*f*(E-E03))-e*np.exp(se*e)*np.exp((1-alfa3)*f*(E-E03)))+(e-ep)/dt+k04*(e*np.exp(se*e)*ah*np.exp(-alfa4*f*(E-E04))-g*np.exp(sg*g)*np.exp((1-alfa4)*f*(E-E04)))
    f6=-k04*(e*np.exp(se*e)*ah*np.exp(-alfa4*f*(E-E04))-g*np.exp(sg*g)*np.exp((1-alfa4)*f*(E-E04)))+(g-gp)/dt+kad1*g*np.exp(sg*g)-kad2*n*np.exp(sn*n)
    f7=-(kad1*g*np.exp(sg*g)-kad2*n*np.exp(sn*n)+0*kr*a*np.exp(sa*a))+(n-npr)/dt
    
    f8=E-E_exact+rho*(current_f(x,k,thermo,E)-gc)
    f9=-gc+gama*diff-gama*rho*(1/dt*(current_f(x,k,thermo,E)-jp)+1/dt*(gc-gcp))
    
    return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9])

@jit(nopython=True)
def current_f_out(c,k,thermo):
   k01,E01,kr,k02,E02,k03,E03,k04,E04,kc1,kc2,kad1,kad2,n0,gama,sa,sb,sc,sd,se,sg,sn,alfa1,alfa2,alfa3,alfa4=k
   alfa,f,ah,rho=thermo
   a,b,c,d,e,g,n,Ep,jp=c
   
   j=(k01*np.exp(-alfa1*f*(Ep-E01))*a*np.exp(sa*a)*ah-k01*np.exp((1-alfa1)*f*(Ep-E01))*b*np.exp(sb*b)+k02*(b*np.exp(sb*b)*ah*np.exp(-alfa2*f*(Ep-E02))-c*np.exp(sc*c)*np.exp((1-alfa2)*f*(Ep-E02)))+
       k04*(e*np.exp(se*e)*np.exp(-alfa4*f*(Ep-E04))*ah-g*np.exp(sg*g)*np.exp((1-alfa4)*f*(Ep-E04)))+k03*(d*np.exp(sd*d)*ah*np.exp(-alfa3*f*(Ep-E03))-e*np.exp(se*e)*np.exp((1-alfa3)*f*(Ep-E03)))-kr*a*np.exp(sa*a))
   return -j*96485*n0

def calc_current(t,const,non_fit,potential_const,E_exp):
    E1,E2,v,amp,frequ=potential_const
    dt=t[1]
    
    x0=[0,0,0,0,0,0,1,E1,0]
    
    dedt=np.diff(E_exp)/dt
    
    sol=[x0]
    gc=[0]
    g=[0]

    for i in range(1,len(t)):
        root=sciop.root(time_step,sol[i-1],args=(t[i],np.array(sol[i-1]),const,non_fit,E1,E2,v,dt,amp,frequ,dedt[i-1],g[-1],E_exp[i]))
        sol.append(root.x)
        g.append(current_f_out(root.x,const,non_fit))
        gc.append(root.x[-1])
    
    sol=np.array(sol)
    g=np.array(g)
    gc=np.array(gc)

    return g+gc

def loss_func(Fit_params,x,y_data,fun,extra_params,E_params,N_har,w,E_data):
    y_sim=fun(x,Fit_params,extra_params,E_params,E_data)
    a_sim=np.array([E_data,y_sim,x])
    sp,freq,I_har_sim=ff.FFT_analysis(a_sim, frequency, N_har, w)
    Phi=0.0
    i=5

    Phi+=np.sqrt(np.sum((y_data[i][5000:-6000]-I_har_sim[i][5000:-6000])**2)/np.sum(y_data[i][5000:-6000]**2))
    return Phi
#Important to define the integer in the output array, as that defines the harmonic used for simulations
def y_sim(x,k01,E01,kr,k02,E02,k03,E03,k04,E04,kc1,kc2,kad1,kad2,n0,gama,sa,sb,sc,sd,se,sg,sn,alfa1,alfa2,alfa3,alfa4):
    const=np.array([k01,E01,kr,k02,E02,k03,E03,k04,E04,kc1,kc2,kad1,kad2,n0,gama,sa,sb,sc,sd,se,sg,sn,alfa1,alfa2,alfa3,alfa4])
    j_sim=calc_current(x, const, non_fit, E_c,E_data)

    a_sim=np.array([E_data,j_sim,t_data]).T
    sp,freq,I_har=ff.FFT_analysis(a_sim.T, frequency, N_har, wind)
    return I_har[5]

#Uncompensated resistance found by EIS
rho= 20

#Parameters used for simulatiomn
alfa=0.5
f=38.0
ah=0.1

#Initial guess for fitting
const=np.array([6.04247497e-01, 1.42025193e+00, 1.03886429e+00, 1.27230406e-01,
       1.44071004e+00, 6.41201735e-01, 2.81657482e-01, 2.00475561e+00,
       6.74568674e-01, 1.68628226e+02, 2.28338029e+01, 6.30073810e+01,
       1.53305399e+01, 1.88894184e-11, 1.35304295e-05, 4.71022467e+01,
       1.27211843e-01, 1.31713118e+00, 4.33170756e-02, 7.05400012e-03,
       1.19940095e+00, 1.75729882e+00, 1.38982559e+00, 2.82599037e-01,
       4.36865430e-01, 1.94337611e+00])
const=np.abs(const)

non_fit=np.array([alfa,f,ah,rho])

E1=0.4
E2=1.6
v=0.001
amplitude=0.
frequency=0.127#*10

E_c=[E1,E2,v,amplitude,frequency]
tmax=2*abs(E2-E1)/v

a,b=ff.open_single_file()
# pas=-1
pas=int(len(a[:,0])/2)
# pas=110900
# pas=1050
# pas1=1350
a=a[:pas,:]
a[:,1]=a[:,1]*10**-3
E_data=a[:,0]#318088
# E_data=smooth(E_data,smothf)
I_data=a[:,1]
t_data=a[:,2]-a[0,2]
dt=t_data[2]-t_data[1]
# sys.exit()

N_har=7
wind=0.009*np.ones(N_har+1)

Label="Material name"
Title="OER polarisation curve comparison between different materials"

sp,freq,I_har_data=ff.FFT_analysis(a.T, frequency, N_har, wind)
ff.Harmonic_plots(I_har_data, t_data,w=0,dt=a[10,2]-a[9,2],label=Label,pas=pas,Title=Title)

ff.FT_plot(freq, sp,label=Label)

ff.Plot_measure(E_data, I_data,label=Label)

sys.exit()

start_time = time.process_time()

#Different fitting routines available, the integer in the I_har_data indicates which harmonic will be used for fitting
popt,popcov=sciop.curve_fit(y_sim,t_data,I_har_data[5],p0=const,xtol=10**-8,method="lm")
poperror=np.sqrt(np.diag(popcov))

print(loss_func(popt,t_data,I_har_data,calc_current,non_fit,E_c,N_har,wind,E_data))

j_sim=calc_current(t_data, popt, non_fit, E_c,E_data)

# res=sciop.minimize(loss_func,const,method="Nelder-Mead",tol=10**-8,args=(t_data,I_har_data,calc_current,non_fit,E_c,N_har,wind,E_data))
# j_sim=calc_current(t_data, res.x, non_fit, E_c,E_data)
# print(res.fun)

# j_sim=calc_current(t_data, const, non_fit, E_c,E_data)
# print(loss_func(const,t_data,I_har_data,calc_current,non_fit,E_c,N_har,wind,E_data))


print()
print()
print("Program time")
print("--- %s seconds ---" % (time.process_time() - start_time))

a_sim=np.array([E_data,j_sim,t_data]).T
sp1,freq,I_har=ff.FFT_analysis(a_sim.T, frequency, N_har, wind)
ff.Harmonic_plots(I_har, a[:,2],w=0,dt=a[10,2]-a[9,2],label="simulation",Title=Title)
ff.FT_plot(freq, sp1,label="simulation")
# # 
# ff.Plot_measure(E_data, j_sim)




