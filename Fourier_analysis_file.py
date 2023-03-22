# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:01:16 2022

@author: ozbejv
"""

import numpy as np
import scipy.signal as scisi
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from galvani import BioLogic
import sys
import numba
from numba import jit
import matplotlib as mpl

plt.rc("font", size=14)

def FFT_analysis(a,f,N,w):
    ### input parameters:
    #a-ndarray of data from voltammetric experimet in form E,I,t
    #f-base frequency of ACV potential
    #N-number of harmonics to be processed
    #w-array or list of window function parameters
    
    ### output parameters:
    #sp-ndarray of FT values of the input current
    #freq-ndarray of frequencies; to be used in a window function or as x-axis
    #I_harmonics-list of ndarrays of analytical signals of the harmonic currents; already procesed
    
    def rectangular(f,w0,w):
        return np.where(abs(f-w0)<=w,1,0)
    
    I_harmonics=[]
    dt=a[2,10]-a[2,9]
    freq=np.fft.fftfreq(a[2].shape[-1],d=dt)
    sp=np.fft.fft(a[1])
    for i in range(N+1):
    #     #kopiram FFT
        if i==0:
            filter_sp=sp.copy()
            window=rectangular(freq,i*f,w[i])
            filter_sp=window*filter_sp
            Inew=np.fft.ifft(filter_sp).real
            I_harmonics.append(Inew)
        else:
            filter_sp=sp.copy()
            window=rectangular(freq,i*f,w[i])+rectangular(freq,-i*f,w[i])
            filter_sp=window*filter_sp
            Inew=np.fft.ifft(filter_sp).real
            anal_signal=np.abs(scisi.hilbert(Inew))
            I_harmonics.append(anal_signal)
    return sp,freq,I_harmonics

def Harmonic_plots(I_harmonics,x_axis,w=0,dt=0,label="",Title="",pas=-1,col=0):
    ###input parameters:
    #I_harmonics-list of ndarrays of analytical harmonic currents to be plotted
    #x_axis-a list or ndarray to plot currents against
    #can be either time or experimental potential
    #w-width of rectangular window function to be used when plotting against potential, dc potential found in FT of experimental potential
    #dt-time step used in experiment; used when finding potential to plot against
    #label-label data in the plot
    #Title-Adding to the title of the plot, format is: "nth harmonic+Title"
    
    titles=["0th","1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]
    
    def rectangular(f,w0,w):
        return np.where(abs(f-w0)<=w,1,0)
    
    if w==0:
        x_axis=x_axis-x_axis[0]
        for i in range(len(I_harmonics)):
            # plt.figure(str(i)+"_harmonic",figsize=(11,7))
            # plt.title(str(i)+"th harmonic "+Title)
            # plt.plot(x_axis[5000:-6000]*10**-3+0.4,I_harmonics[i][5000:-6000],label=label)
            # plt.xlabel("$E_{DC}$ [V]")
            # plt.ylabel("I [A]")
            
            if col==0:
                plt.figure(str(i)+"_harmonic",figsize=(11,7))
                plt.title(str(i)+"th harmonic "+Title)
                plt.plot(x_axis[5000:-6000]*10**-3+0.4,I_harmonics[i][5000:-6000],label=label)
                plt.xlabel("$E_{DC}$ [V]")
                plt.ylabel("I [A]")
            elif col==1:
                plt.figure(str(i)+"_harmonic",figsize=(11,7))
                plt.title(str(i)+"th harmonic "+Title)
                plt.plot(x_axis[5000:-6000]*10**-3+0.4,I_harmonics[i][5000:-6000],label=label,color="#ff7f0e")
                plt.xlabel("$E_{DC}$ [V]")
                plt.ylabel("I [A]")
            else:
                plt.figure(str(i)+"_harmonic",figsize=(11,7))
                plt.title(str(i)+"th harmonic "+Title)
                plt.plot(x_axis[5000:-6000]*10**-3+0.4,I_harmonics[i][5000:-6000],label=label,color="0")
                plt.xlabel("$E_{DC}$ [V]")
                plt.ylabel("I [A]")
                plt.legend()
    
    elif w<0:
        print("Error in ploting function, w < 0")
    
    elif dt<=0:
        print("Error in ploting function, dt <= 0")
    
    elif w>0 and dt>0:
        sp=np.fft.fft(x_axis)
        freq=np.fft.fftfreq(x_axis.shape[-1],d=dt)
        E_sim=np.fft.ifft(rectangular(freq,0,w)*sp).real
        
        # N = 6
        # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.CMRmap(np.linspace(0,1,N)))
        
        for i in range(len(I_harmonics)):
            
            if col==0:
                plt.figure(str(i)+"_harmonic_c",figsize=(11,7))
                plt.title(titles[i]+" harmonic "+Title)
                plt.plot(E_sim[:pas],I_harmonics[i],label=label)
                plt.xlabel("$E_{DC}$ [V]")
                plt.ylabel("I [A]")
            elif col==1:
                plt.figure(str(i)+"_harmonic_c",figsize=(11,7))
                plt.title(titles[i]+" harmonic "+Title)
                plt.plot(E_sim[:pas],I_harmonics[i],label=label,color="#d62728",linestyle="--")
                plt.xlabel("$E_{DC}$ [V]")
                plt.ylabel("I [A]")
            else:
                plt.figure(str(i)+"_harmonic_c",figsize=(11,7))
                plt.title(titles[i]+" harmonic "+Title)
                plt.plot(E_sim[:pas],I_harmonics[i],label=label,color="000")
                plt.xlabel("$E_{DC}$ [V]")
                plt.ylabel("I [A]")
            # plt.vlines(0.63,0,1.5*max(I_harmonics[i]),colors="Black",linestyles="dashed")
            # plt.vlines(1.015,0,1.5*max(I_harmonics[i]),colors="Black",linestyles="dashed")
            # plt.vlines(1.4,0,1.5*max(I_harmonics[i]),colors="Black",linestyles="dashed")
            # plt.vlines(1.7,0,1.5*max(I_harmonics[i]),colors="Black",linestyles="dashed")
            
            # plt.axvspan(0.63-0.11, 0.63+0.11, color='gray', alpha=0.1, lw=0)
            # plt.axvspan(1.015-0.11, 1.015+0.11, color='gray', alpha=0.1, lw=0)
            # plt.axvspan(1.4-0.11, 1.4+0.11, color='gray', alpha=0.1, lw=0)
            # plt.axvspan(1.7-0.11, 1.7+0.11, color='gray', alpha=0.1, lw=0)

def FT_plot(freq,sp,Title="",label=""):
    plt.figure("Fourier_Transform",figsize=(11,7))
    plt.title("Fourier Transform "+Title)
    plt.plot(freq,np.log10(np.abs(sp)),label=label)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("$log_{10}$($I_{FT}$) [dB]")
    
def Plot_measure(x_axis,y_axis,Title="",label="",x_label=""):
    plt.figure("Base_signal2",figsize=(11,7))
    plt.title("Measured Signal "+Title)
    plt.plot(x_axis,y_axis,label=label)
    plt.xlabel(x_label)
    plt.ylabel("I [A]")

def open_single_file():
    ###Nonstandard libraries used:
    #galvani-library found on: https://pypi.org/project/galvani/ or https://github.com/echemdata/galvani
    #
    ###input:
    #None
    ###functionality:
    #function opens a tkinter dialog window to find a file to be imported into a program
    #file can be either ".txt" or ".mpr" type 
    ###output:
    #a-ndarray of experimental data in shape: E,I,t
    #b-dictionary of added parameters in case file is type ".txt", else is empty
    a=[]
    b={}    
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename()
    

    if file_path[-4:]==".txt":
        with open(file_path) as f:
            for line in f:
                if len(line.split())==3:
                    try:
                        a1,a2,a3=line.split()
                        a1=float(a1.replace(',','.'))
                        a2=float(a2.replace(',','.'))
                        a3=float(a3.replace(',','.'))
                        a.append([a1,a2,a3])
                    except:
                        try:
                            a1,a2,a3=line.split()
                            b[a1+' '+a2]=float(a3.replace(',','.'))
                        except:
                            b[a1+' '+a2]=a3.replace(',','.')
                if len(line.split())==2:
                    try:
                        a1,a2=line.split()
                        b[a1]=float(a2.replace(',','.'))
                    except:
                        pass
            f.close()
            a=np.array(a)
            return a,b
    
    elif file_path[-4:]==".mpr":
        mpr_file = BioLogic.MPRfile(file_path)
        df = pd.DataFrame(mpr_file.data)
        a=np.array([df["Ewe/V"],df["I/mA"],df["time/s"]]).T
        return a,b
    
    else:
        print("Selection error: incorrect file selected for opening")
        return False, False

# frequency=0.127
# w=np.ones(8)*0.01
# a,b=open_single_file()
# a=a[1000:,:]
# a=a[2:112590,:]
# a[:,2]=a[:,2]-a[0,2]
# wind=np.ones(7)
# sp,freq,I_har=FFT_analysis(a.T, frequency, 7, w)
# Harmonic_plots(I_har, a[:,2],w=0,dt=a[10,2]-a[9,2])
# FT_plot(freq, sp)
# Plot_measure(a[:,0], a[:,1])