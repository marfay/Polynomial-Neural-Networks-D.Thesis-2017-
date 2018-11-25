import numpy as np
from matplotlib.pyplot import *
import cv2
import neural_network as nn
I = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0],[1,1,1],[1,1,0],[0,1,1]])



a=[0,0,1]
b=[1,0,0]
c=[0,1,0]
I=np.array([0,0,0])
for i in range(3):
    I=np.vstack((I,a+np.random.rand(3)/15.))
    I =np.vstack((I, b + np.random.rand(3)/15.))
    I =np.vstack((I, c + np.random.rand(3)/15.))
I=I/np.amax(I)

#imshow(I.T)
#show()
print(I)
n=30
m=30
som = nn.SOM(n,m,I)
weights = som.weights
for ep in range(100):
    weights = som.som_one(ep,n,m,(n+m)/0.3,1,100,30,weights)
    #som.plot_2dnet(weights)
    #som.plot_net(weights)
    a=som.u_matrix(weights)
    #imshow(a)
    #show()
    resized_image = cv2.resize(weights, (n*10, m*10))
    cv2.imshow('f', resized_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #imshow(weights)
    #show()
def find_class(weights,W):
    winner = np.zeros((weights.shape[0], weights.shape[1]))  # saves the Carthese coordinates of actual winner which will be saved in winner_z
    def distance( inp, w):
        d = np.sqrt(np.sum((inp - w) ** 2))
        return d
    for k in range(n):
        for l in range(m):
            winner[k, l] = distance(W, weights[k, l, :])

    win = np.where(winner == np.amin(winner))
    print(win)
find_class(weights,[1.,0,0])

raw_input()

