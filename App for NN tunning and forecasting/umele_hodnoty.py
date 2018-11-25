import numpy as np
import load_ene as ee
days = (np.load('log_days.npy'))
time = np.sin(np.load('the_time.npy'))
time = np.load('the_time.npy')
date_X, time_X, _ = ee.load_data()
X = np.load('ENE_data.npy')
from matplotlib.pyplot import *

print(date_X[96*250],time_X[96*250],time[96*250])
samples = np.linspace(0,1,days.shape[0])

samples = np.linspace(0,1,days.shape[0])
weather = np.sin(2*np.pi*samples/2)*100

real_sum = np.random.rand(days.shape[0])*10

samples = np.linspace(0,1,days.shape[0])
yu = np.sin(2*np.pi*samples*(days.shape[0]/(2*96)))
yu[2315:2385]=0.35
yu[3310:3316]=0.85

sum1= np.sin(2*np.pi*samples*(days.shape[0]/(5)))/40
sum2= np.sin(2*np.pi*samples*(days.shape[0]/(4)))/40
sum3= np.sin(2*np.pi*samples*(days.shape[0]/(3)))/40
subplot(1,1,1)
yu=days*yu**2 + sum1 + sum2 + sum3
# Chyba

yu = yu *400+350+weather+ real_sum
plot(yu)
show()
import neural_network as nn
X = nn.stack_input(X,yu)

#weather = np.savetxt('weather.txt',weather)
#X = np.savetxt('ENE_data_test.txt',X)