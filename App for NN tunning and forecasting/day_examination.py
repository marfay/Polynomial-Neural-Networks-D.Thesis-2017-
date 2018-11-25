# -*- coding: utf-8 -*-
import os, sys


import numpy as np
from matplotlib.pyplot import *
import load_ene as ee
data = np.load('ENE_data.npy')
#date,time,_=ee.load_data()
X = data[:,7]
X = X[::47]
C = X
for i in range(4):
    X = np.append(X,C)
matplotlib.rcParams.update({'font.size': 16})

subplot(1,1,1)
ylabel(u'Spotřeba energie [kW]')
xlabel('Vzorky[15 min.]')
plot(X,'k',linewidth=0.5)
show()
