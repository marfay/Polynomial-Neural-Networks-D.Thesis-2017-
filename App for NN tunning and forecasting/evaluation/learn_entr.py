# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib.lines as mlines


from matplotlib.pyplot import *

path = 'evaluation\\'
e = np.loadtxt(path+'e.txt')
yr = np.loadtxt(path+'yr.txt')
yn = np.loadtxt(path+'yn.txt')
EA = np.loadtxt(path+'EA.txt')
EAP = np.loadtxt(path+'EAP.txt')
#matplotlib.rcParams.update({'font.size': 14})




subplot(3,1,1)
title('Neural output')
xlabel('samples')
ylabel('Data units')
plot(yr,'b',label = 'Real data')
plot(yn,'g',label = 'Neural output')
plot(abs(e),'r', label = 'Error')
legend()
subplot(3,1,2)
title('Change in Entropy of Learning')
xlabel('samples')
ylabel('[-]')
plot(EA,label ='EA')
ylim((0,1))
legend()
subplot(3,1,3)
title('Entropy')
xlabel('samples')
ylabel('[-]')
plot(EAP,label ='EAP')

legend()

show()



