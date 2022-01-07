#Aufgrund der besonderen verdeckten Schicht zwischen dem
#Decoder und Encoder wurde hier der Programmcode der Keras
#Biblothek als Vorlage genommen und angepasst.
#Dieser ist unter https://keras.io/examples/generative/vae/
#am 16.09.2021 genommen und angepasst worden.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class Sampling(layers.Layer):
    #Benutze den Erwartungswert und die Varianz der Kodierungen
    #Um entsprechende Variationen zu generieren
    def call(self, inputs):
        h_erwartung, h_log_varianz = inputs
        traingsdimension = tf.shape(h_erwartung)[0]
        kodierungsdimension = tf.shape(h_erwartung)[1]
        epsilon = tf.keras.backend.random_normal(shape=(traingsdimension, kodierungsdimension))
        return h_erwartung + tf.exp(0.5 * h_log_varianz) * epsilon

#Hier tiefer Autoencoder konstruiert, damit die
#Rekonstruktionen im 2 dimensionalen gut sind
eingabe = keras.Input(shape=(784,))
x = layers.Dense(1024, activation="relu")(eingabe)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
h_erwartung = layers.Dense(2,activation='sigmoid', name="h_erwartung")(x)
h_log_varianz = layers.Dense(2, name="h_log_varianz")(x)
z = Sampling()([h_erwartung, h_log_varianz])
encoder = keras.Model(eingabe, [h_erwartung, h_log_varianz, z], name="encoder")

decoder_eingabe = keras.Input(shape=(2,))
x = layers.Dense(32, activation="relu")(decoder_eingabe)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(1024, activation="relu")(x)
decoder_ausgabe = layers.Dense(784, activation="relu")(x)
decoder = keras.Model(decoder_eingabe, decoder_ausgabe, name="decoder")

#Konstruiere eine Klasse zu Variational Autoencodern
#um passendes trainieren zu schaffen
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.rekonstruktions_verlust_tracker = keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [self.rekonstruktions_verlust_tracker]

    def train_step(self, data):
        #Bestimmte die Gradienten des aktuellen Autoencoders an anhand
        #der aktuellen Trainingsdaten und Parameter
        with tf.GradientTape() as tape:
            h_erwartung, h_log_varianz, h = self.encoder(data)
            rekonstruktion = self.decoder(h)
            mse_verlust = tf.reduce_mean(tf.reduce_sum(
                    keras.losses.mean_squared_error(data, rekonstruktion)))
        grads = tape.gradient(mse_verlust, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.rekonstruktions_verlust_tracker.update_state(mse_verlust)
        return {"rekonstruktions_verlust": self.rekonstruktions_verlust_tracker.result()}

#Impotiere den Trainingsdatensatz
(x_train, y_train), (x_test, y_test) =  keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0

#Initialisiere und trainiere den VAE
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(x_train, epochs=200, batch_size=2048)

#Erzeuge 20x20 Bilder je 28x28 Pixel, da der
#Kodierungsraum der I_2 ist erzeuge glvt. diese
#in [0,1] und erzeuge damit Abb. 10
digit_size = 28
figure = np.zeros((digit_size * 20, digit_size * 20))
grid_x = np.linspace(0, 1, 20)
grid_y = np.linspace(0, 1, 20)[::-1]

for i, yi in enumerate(grid_y):
    for j, xi in enumerate(grid_x):
        #Nimm das Sample aus [0,1]^2 und decodiere es
        z_sample = np.array([[xi, yi]])
        x_decoded = vae.decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size : (i + 1) * digit_size,
            j * digit_size : (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
start_range = digit_size // 2
end_range = 20 * digit_size + start_range
pixel_range = np.arange(start_range, end_range, digit_size)
sample_range_x = np.round(grid_x, 1)
sample_range_y = np.round(grid_y, 1)
plt.xticks(pixel_range, sample_range_x)
plt.yticks(pixel_range, sample_range_y)
plt.xlabel("off")
plt.ylabel("off")
plt.imshow(figure, cmap="Greys_r")
plt.show()

