#Autoencoder = another wy to train in cases of image classification

import tensorflow as tf
from tensorflow.keras.datasets import mnist # type: ignore
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_trainclass), (x_test, y_testclass) = mnist.load_data()

def plotn(n,x):
    fig, ax = plt.subplots(1,n)
    for i, z in enumerate(x[0:n]):
        ax[i].imshow(z.reshape(28,28) if z.size==28*28 else z.reshape(14,14) if z.size==14*14 else z)
    plt.show()
    
plotn(10,x_train)