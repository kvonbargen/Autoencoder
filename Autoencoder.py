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

#Erzeuge ein leeres NN mit der Eingabedimension 784,
#einem Kodierungsraum der Dimension 'kodierungsdimension'
#und einer Dimension des Bildraumes von 784. Hierbei werden
#jeweils die Parameter W_1, b_1  und W_2, b_2 erzeugt.
#Mit den Aktivierungsfunktionen ReLU und der Sigmoidfunktion
def model(kodierungsdimension):
    inputs = keras.Input(shape=(784))
    encoder = layers.Dense(kodierungsdimension, activation="relu")(inputs)
    decoder = layers.Dense(784, activation="sigmoid")(encoder)
    return keras.Model(inputs=inputs, outputs=decoder)

#Erstelle Modelle mit verschiedenen Dimensionen des
#Kodierungsraumes, trainiere diese mit der Verlustfunktion
#des Standardmodelles und einer adaptiven Lernrate in 200
#Zyklen und speichere die Traingsfortschritte in der jeweiligen history.
Autoencoder2 = Autoencoder(2)
Autoencoder2.compile(optimizer="adam", loss="MSE")
history2 = Autoencoder2.fit(x_train, x_train, batch_size=2048, epochs=200, verbose=1, shuffle=True, validation_data=(x_test, x_test))
Autoencoder5 = Autoencoder(5)
Autoencoder5.compile(optimizer="adam", loss="MSE")
history5 = Autoencoder5.fit(x_train, x_train, batch_size=2048, epochs=200, verbose=1, shuffle=True, validation_data=(x_test, x_test))
Autoencoder10 = Autoencoder(10)
Autoencoder.compile(optimizer="adam", loss="MSE")
history10 = Autoencoder10.fit(x_train, x_train, batch_size=2048, epochs=200, verbose=1, shuffle=True, validation_data=(x_test, x_test))
Autoencoder20 = Autoencoder(20)
Autoencoder20.Autoencoder(optimizer="adam", loss="MSE")
history20 = Autoencoder20.fit(x_train, x_train, batch_size=2048, epochs=200, verbose=1, shuffle=True, validation_data=(x_test, x_test))
Autoencoder50 = Autoencoder(50)
Autoencoder50.compile(optimizer="adam", loss="MSE")
history50 = Autoencoder50.fit(x_train, x_train, batch_size=2048, epochs=200, verbose=1, shuffle=True, validation_data=(x_test, x_test))

#Lege auf den Plot die Ergebnisse des Trainings
#der Autoencoder und erzeuge damit Abb.6
plt.plot(history2.history['val_loss'], label='MSE m=2')
plt.plot(history5.history['val_loss'], label='MSE m=5')
plt.plot(history10.history['val_loss'], label='MSE m=10')
plt.plot(history20.history['val_loss'], label='MSE m=20')
plt.plot(history50.history['val_loss'], label='MSE m=50')
plt.title('Vergleich des MSE mit variabler Dimension des Kodierungsraumes')
plt.ylabel('MSE')
plt.xlabel('Anzahl der Trainingsschritte')
plt.legend(loc="upper left")
plt.show()

#Werte AE(x_test) aus und nimm 10 Zufalls-Datenpunkte
#mit den jeweiligen Rekonstruktionen.Daraufhin werden
#diese wie in Abb. 5 verglichen und zuvor wieder
#in das Format 28x28 konvertiert, sowie die Werte in [0,1]
#als Graustufenwerte aufgefasst.
x_rekonstruiert = model20.predict(x_test)
indices = np.random.randint(len(x_test), size=10)
images1 = x_test[indices, :]
images2 = x_rekonstruiert[indices, :]
plt.figure(figsize=(20, 4))
for i, (image1, image2) in enumerate(zip(images1, images2)):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(image1.reshape(28, 28))
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 11)
    plt.imshow(image2.reshape(28, 28))
    plt.title("rekonstruiert")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



