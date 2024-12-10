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
    
plotn(5,x_train)

#Training some samples
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy,mse

input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

encoder = Model(input_img,encoded)

input_rep = Input(shape=(4,4,8))

x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_rep)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(input_rep,decoded)

autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
     

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
     

autoencoder.fit(x_train, x_train,
                epochs=25,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

y_test = autoencoder.predict(x_test[0:5])
plotn(5,x_test)
plotn(5,y_test)

encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test[0:5])

plotn(5,encoded_imgs.reshape(5,-1,8))

print(encoded_imgs.max(),encoded_imgs.min())
res = decoder.predict(7*np.random.rand(7,4,4,8))
plotn(7,res)

#Denoising
def noisify(data):
    return np.clip(data+np.random.normal(loc=0.5,scale=0.5,size=data.shape),0.,1.)
x_train_noise = noisify(x_train)
x_test_noise = noisify(x_test)

plotn(5,x_train_noise)

autoencoder.fit(x_train_noise, x_train,
                epochs=25,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noise, x_test))

y_test = autoencoder.predict(x_test_noise[0:5])
plotn(5,x_test_noise)
plotn(5,y_test)

#High-resolution
#Train the encoder to increase the resolution
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, AveragePooling2D, Conv2D, MaxPooling2D, UpSampling2D
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Carregar e pré-processar dados
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Ajustar a dimensão para (samples, height, width, channels)
x_train = x_train[..., tf.newaxis]  # Adiciona canal
x_test = x_test[..., tf.newaxis]

# Aplicar AveragePooling2D aos dados
x_train_lr = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x_train).numpy()
x_test_lr = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x_test).numpy()

# Função para exibir imagens
def plotn(n, images):
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.axis("off")
    plt.show()

# Visualizar dados reduzidos
plotn(5, x_train_lr)

# Definir o modelo codificador
input_img = Input(shape=(14, 14, 1))  # Entrada após pooling
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

encoder = Model(input_img, encoded)

# Definir o modelo decodificador
input_rep = Input(shape=(4, 4, 8))  # Saída do codificador
x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_rep)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(input_rep, decoded)

# Criar o modelo autoencoder
autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Treinar o modelo
autoencoder.fit(
    x_train_lr, x_train,  # x_train_lr reduzido, x_train original
    epochs=25,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_lr, x_test)
)

# Testar o modelo
y_test_lr = autoencoder.predict(x_test_lr[:5])

# Visualizar resultados
plotn(5, x_test_lr[:5])  # Entradas reduzidas
plotn(5, y_test_lr)      # Saídas reconstruídas
