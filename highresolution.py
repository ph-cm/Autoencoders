import tensorflow as tf
from tensorflow.keras.layers import (
    Input, AveragePooling2D, Conv2D, UpSampling2D
)
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Carregar e pré-processar dados
(x_train, y_trainclass), (x_test, y_testclass) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Ajustar a dimensão para (samples, height, width, channels)
x_train = x_train[..., tf.newaxis]  # Adiciona canal
x_test = x_test[..., tf.newaxis]

# Aplicar AveragePooling2D para reduzir as dimensões
x_train_lr = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x_train).numpy()
x_test_lr = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x_test).numpy()

# Função para exibir imagens
def plotn(n, x, title=""):
    fig, ax = plt.subplots(1, n, figsize=(15, 5))
    for i, z in enumerate(x[:n]):
        ax[i].imshow(z.squeeze(), cmap="Greens")  # Tons de verde
        ax[i].axis("off")
        if title:
            ax[i].set_title(title, fontsize=10)
    plt.show()

# Visualizar dados reduzidos
plotn(5, x_train_lr, title="Input (Low Resolution)")

# Definir o modelo codificador
input_img = Input(shape=(14, 14, 1))  # Entrada de baixa resolução
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

encoder = Model(input_img, encoded)

# Definir o modelo decodificador
input_rep = Input(shape=(14, 14, 64))  # Saída do codificador
x = UpSampling2D((2, 2))(input_rep)  # Aumenta para (28, 28, 64)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # Final: (28, 28, 1)

decoder = Model(input_rep, decoded)

# Criar o modelo autoencoder para super-resolução
autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(optimizer='adam', loss='mse')  # Usar MSE como função de perda

# Treinar o modelo
autoencoder.fit(
    x_train_lr, x_train,  # Imagens de baixa resolução como entrada e originais como saída
    epochs=25,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_lr, x_test)
)

# Testar o modelo
y_test_sr = autoencoder.predict(x_test_lr[:5])

# Visualizar resultados
plotn(5, x_test_lr[:5], title="Input (Low Resolution)")
plotn(5, y_test_sr, title="Output (Super Resolution)")
