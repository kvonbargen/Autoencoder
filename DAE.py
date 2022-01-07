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

#Funktion die ein Rauschen epsilon ~ N(0, sigma* E_d)
#erzeugt und auf die Eingabe addiert und wieder in das
#vordefinierte Interval passt.
#Mit x_i<0 -> x_i=0 und x_i>1 -> x_i=1
def verrauschen(eingabe):
    sigma = 0.5
    epsilon = np.random.normal(loc=0.0, scale=sigma, size=eingabe.shape)
    rauschen = eingabe + epsilon
    normlisiert = np.clip(rauschen, 0.0, 1.0)
    return normlisiert

x_train_verrauscht = verrauschen(x_train)
x_test_verrauscht = verrauschen(x_test)

#Definiere den Autoencoder
def Autoencoder():
    eingabe = keras.Input(shape=(784))
    encoder = layers.Dense(100, activation="relu")(eingabe)
    decoder = layers.Dense(784, activation="sigmoid")(encoder)
    return keras.Model(inputs=eingabe, outputs=decoder)

#Kompiliere und trainiere den Autoencoder mit der
#Eingabe der verrauschten Daten und dem Ziel der Orginale
autoencoder= Autoencoder()
autoencoder.compile(optimizer="adam", loss="MSE")
autoencoder.fit(x_train_verrauscht, x_train, batch_size=2048, epochs=500, verbose=1, shuffle=True, validation_data=(x_test_verrauscht, x_test))

#Erzeuge die Rekonstruktionen des Test-Datensatzes
#und vergleiche diese mit den Orginalen und
#mit den verrauschten Eingabedaten in Abb. 9.
rekonstruktion = autoencoder.predict(x_test_verrauscht)
indices = np.random.randint(len(x_test), size=10)
images1 = x_test[indices, :]
images2 = x_test_verrauscht[indices, :]
images3 = rekonstruktion[indices, :]

fig, axarr = plt.subplots(3,10)
for i, (image1, image2, image3) in enumerate(zip(images1, images2,images3)):
    plt.gray()
    axarr[0,i].imshow(image1.reshape(28, 28))
    axarr[0,i].get_xaxis().set_visible(False)
    axarr[0,i].get_yaxis().set_visible(False)

    plt.gray()
    axarr[1,i].imshow(image2.reshape(28, 28))
    axarr[1,i].get_xaxis().set_visible(False)
    axarr[1,i].get_yaxis().set_visible(False)

    plt.gray()
    axarr[2,i].imshow(image3.reshape(28, 28))
    axarr[2,i].get_xaxis().set_visible(False)
    axarr[2,i].get_yaxis().set_visible(False)

plt.show()

