import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#Importiere den MNIST-Datensatz und vektorisiere
#die 28 x 28 Bilder zu einem Vektor der Dimension 784.
#Dann normiere die Graustufenwerte
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# Trainiere nur mit den Ziffern 0 und 1
x_train = x_train[np.where(y_train < 2)]
y_train = y_train[np.where(y_train < 2)]
x_test = x_test[np.where(y_test < 2)]
y_test = y_test[np.where(y_test< 2)]

#Bilde einen tieferen Autoencoder mit weiteren Schichten im Encoder und Decoder
#damit die Resultate besser sind.
#Es wird ein Netzwerk mit den Dimensionen
#784 -> 1024 -> 64 -> 2 -> 64 -> 1024 -> 784
#erzeugt.
inputs = keras.Input(shape=(784))
x = layers.Dense(1024, activation="relu")(inputs)
x = layers.Dense(64, activation="relu")(x)
encoder = layers.Dense(2, activation="relu")(x)
z = layers.Dense(64, activation="relu")(encoder)
z = layers.Dense(1024, activation="relu")(z)
decoder = layers.Dense(784, activation="sigmoid")(z)
Autoencoder= keras.Model(inputs=inputs, outputs=decoder)

#encoded ist die kodierte Ausgabe des Encoders
encoded = keras.Model(inputs=inputs, outputs=encoder)

#Kompiliere und trainiere den Autoencoder
Autoencoder.compile(optimizer="adam", loss="MSE")
Autoencoder.fit(x_train, x_train, batch_size=4096, epochs=200, verbose=1, shuffle=True, validation_data=(x_test, x_test))

#Die vom Encoder kodierte Daten werden nun zweidimensional
#Und geplotet, sodass Abb. 7 erzeugt wird.
kodiert = encoded.predict(x_test)
fig, ax = plt.subplots(figsize=(6,6))
scat = ax.scatter(kodiert[:,0], kodiert[:,1], c=y_test, cmap=plt.cm.jet, edgecolors='black')
cb = plt.colorbar(scat)
ax.axis('off')
plt.show()

