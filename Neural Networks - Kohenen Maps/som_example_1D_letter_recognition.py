import numpy as np
from matplotlib.pyplot import *
import cv2
import neural_network as nn
o=cv2.imread('a.jpg')

I = nn.translate_image(o)
print(I.shape)
#I = np.random.shuffle(I)
#I = I[::11]
np.random.shuffle(I)
print(I.shape)
show()
print(I.shape)
plot(I[:,0],I[:,1],'ro')
show()
#imshow(o)
#show()





import neural_network as nn
n=8#o.shape[0]
m=8#o.shape[1]
print(I.shape)
print(n)
som = nn.SOM(n,m,I)
weights=som.weights
print(weights)
epochs=100
for ep in range(epochs):
    #"""
    axis([0, 35, 0, 35])
    ion()
    cla()

    for i in range(n):
        for j in range(m):
            scatter(weights[i, j, 0], weights[i, j, 1], color='blue')
    scatter(I[:, 0], I[:, 1], color='black')
    pause(0.01)
    #"""
    weights = som.som_one(ep,n,m,(n+m)/2,0.5,15,300,weights,type='mexican') #/4 nebo 6
    #resized_image = cv2.resize(weights, (n * 30, m * 30))
    #som.plot_2dnet(weights)



    """
    #cv2.imshow('f', o)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    """
    print(100.*ep/epochs)

a = som.plot_2dnet_data(weights,I,type='ko')
som.plot_2dnet(weights)
imshow(som.u_matrix(weights))
show()

