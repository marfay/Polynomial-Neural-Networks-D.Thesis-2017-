# -*- coding: utf-8 -*-
import os, sys


import numpy as np
from matplotlib.pyplot import *
import neural_network as nn


path = 'evaluation\\'
e = np.loadtxt(path+'e.txt')
yr = np.loadtxt(path+'yr.txt')
yn = np.loadtxt(path+'yn.txt')
EA = np.loadtxt(path+'EA.txt')
EAP = np.loadtxt(path+'EAP.txt')
MAE = np.loadtxt(path+'MAE.txt')
RMSE = np.loadtxt(path+'RMSE.txt')
error_percentage = np.loadtxt(path+'error_percentage.txt')
err_settings = np.loadtxt(path+'error_settings.txt')
plot_what = np.loadtxt(path+'what_to_plot.txt')
#matplotlib.rcParams.update({'font.size': 14})

error_signal = np.zeros(error_percentage.shape[0])
for i in range(error_percentage.shape[0]):
    if err_settings[1]<error_percentage[i]:
        error_signal[i]=1


def eval_signal(signal):
    """
    Funkce, ktera z detekcniho signalu vytahne intervaly pro preruseni uceni
    :param signal:
    :return:
    """
    # -Podstata -> Nalezt hrany prechodu
    inter=np.array([])
    signal[0]=0
    signal[-1]=0
    for i in range(0, signal.shape[0]-1):
        if signal[i]==0 and signal[i+1]>0:
            inter = np.append(inter, i+1)
        if signal[i]>0 and signal[i+1]==0:
            inter = np.append(inter, i)
    interval_array = inter.reshape(inter.shape[0]/2,2)
    inter_string = ''
    for i in inter:
        inter_string += str(int(i))+' '
    return inter_string

intervals = eval_signal(error_signal)
print(intervals)
path = 'evaluation\\'
with open(path + 'signal_intervals.txt', "w") as text_file:
    text_file.write(intervals)


#Pro diplomku


"""
##############
matplotlib.rcParams.update({'font.size': 14})

subplot(1,1,1)
xlabel('Vzorky [15 min.]')
ylabel('kW')
title(u'Umělá data')
plot(yr[1500:3500],'k',label = u'Umělá data')
show()
"""


if np.sum(plot_what) == 2:
    splots = 3
if np.sum(plot_what) == 1:
    splots = 2
if np.sum(plot_what) == 0:
    splots= 1


matplotlib.rcParams.update({'font.size': 14})

subplot(splots,1,1)
xlabel('Vzorky [15 min.]',fontsize = 16)
ylabel(u'Spotřeba energie [kW]',fontsize = 16)
#title(u'Predikce spotřeby energie')
plot(yr,'k',label = u'Spotřeba energie')
plot(yn,'g',label = 'Predikce')
plot(abs(e),'r', label = 'Chyba',linewidth=0.5,linestyle = '--')
if err_settings[0] == 1:
    plot(error_signal*np.amax(yr)*0.5,label = u'Detekční signál')

legend(loc=2)
if plot_what[1]==1:
    subplot(splots,1,2)
    #title(u'Pruměrná chyba')
    xlabel('Vzorky[15 min.]')
    ylabel('Chyba [%]')
    #plot(np.sqrt((e**2)),'r',label ='error')
    plot(error_percentage,'r',label =u'Chyba')
    line_max = np.ones(error_percentage.shape[0])*err_settings[1]
    plot(line_max,'k',linestyle='--',label = u'Detekční limit')
    legend(loc=2)
if plot_what[0]==1:
    if plot_what[1]==0:
        subplot(splots,1,2)
    else:
        subplot(splots, 1, 3)
    title('RMSE - MAE')
    xlabel('samples')
    ylabel('Data units')
    #der_RMSE = np.zeros(RMSE.shape[0])
    #for i in range(2000,RMSE.shape[0]-1):
    #    der_RMSE[i+1]=np.abs(RMSE[i+1]-RMSE[i])
    #der_RMSE = der_RMSE/np.amax(der_RMSE)
    plot(RMSE,'b',label ='RMSE')
    plot(MAE,'y',label ='MAE')
    #plot(der_RMSE,label = 'der RMSE')
    legend(loc=2)

show()