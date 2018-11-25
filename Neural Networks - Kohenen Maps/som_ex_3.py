import numpy as np
from matplotlib.pyplot import *
import cv2
import neural_network as nn
o = cv2.imread('a.jpg') / 255.


I = nn.translate_image(o)
print(I.shape)
# I = np.random.shuffle(I)
np.random.shuffle(I)
# I = np.random.rand(20,2)
plot(I[:, 0], I[:, 1], 'ro')
show()
# imshow(o)
# show()

axis([0, 10, 0, 10])
ion()

import neural_network as nn

n = 2  # o.shape[0]
m = 50  # o.shape[1]
print(I.shape)
print(n)
som = nn.SOM(n, m, I)
weights = som.weights
epochs = 50
for ep in range(epochs):
    axis([0, 10, 0, 10])
    ion()
    cla()

    weights = som.som_one(ep, n, m, (n + m) / 5, 0.5, 400, 400, weights)
    # resized_image = cv2.resize(weights, (n * 30, m * 30))
    for i in range(n):
        for j in range(m):
            scatter(weights[i, j, 0], weights[i, j, 1])
    print(weights[i, j, 0], weights[i, j, 1])
    pause(0.01)

    cv2.imshow('f', o)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(100. * ep / epochs)
a = som.plot_2dnet_data(weights, I, type='ko')
som.plot_2dnet(weights)
imshow(som.u_matrix(weights))
show()

