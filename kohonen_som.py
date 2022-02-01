# Q1_graded
# Do not change the above line.

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

img = load_img('input.jpg', color_mode="grayscale")
X = img_to_array(img)

# Q1_graded
# Do not change the above line.

class Kohonen:
    def __init__(self, epochs, lr, radius, img_size):
        self.map = np.random.randint(0, 255, (img_size, img_size, 1)).astype(float)
        self.epochs = epochs
        self.learning_rate = lr
        self.radius = radius
    
    def find_winning_neuron(self, x):
        euclidean_distances = ((self.map - x) ** 2).sum(axis=2)
        return np.unravel_index(np.argmin(euclidean_distances, axis=None), euclidean_distances.shape)

    def update_weights(self, x, d):
        a, b = self.find_winning_neuron(x)
        a_neighbor_start = a - d if x > d else 0
        a_neighbor_stop = a + d if a + d < self.map.shape[0] else self.map.shape[0]
        b_neighbor_start = b - d if b > d else 0
        b_neighbor_stop = b + d if b + d < self.map.shape[0] else self.map.shape[1]
        for i in range(a_neighbor_start, a_neighbor_stop):
            for j in range(b_neighbor_start, b_neighbor_stop):
                euclidean_distance = ((i - a) ** 2) + ((j - b) ** 2)
                h = np.exp(-(euclidean_distance/ (2 * (self.radius ** 2))))
                self.map[i,j,:] += self.learning_rate * h * (x - self.map[i,j,:])

    def fit(self, X):
        for epoch in range(self.epochs):
            np.random.shuffle(X)
            for x in X:
                self.update_weights(x, 3)
    

# Q1_graded
# Do not change the above line.

model = Kohonen(10, 0.3, 1, 64)
model.fit(X.reshape(-1, 1))

# Q1_graded
# Do not change the above line.

output = array_to_img(model.map)
output

