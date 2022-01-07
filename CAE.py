import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#Lege eine Autoencoder-Klasse an
#Es wird ein AE mit einer verdeckten Schicht der Dimension 50 angelegt.
class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.flatten_layer = tf.keras.layers.Flatten()
        self.encoder = tf.keras.layers.Dense(50, activation=tf.nn.relu)
        self.decoder = tf.keras.layers.Dense(784,activation=tf.nn.relu)

    #Erzeuge in Eingabe, die Kodierung und die Rekonstruktion des Autoencoders
    def eval(self, inp):
        eingaben_formatiert = self.flatten_layer(inp)
        x = self.encoder(eingaben_formatiert)
        x_kodiert = x
        x_decodiert = self.decoder(x)
        return x_decodiert, eingaben_formatiert, x_kodiert

    #Erzeuge die Kodierung des Autoencoders
    def encode(self, eingabe):
        eingaben_formatiert = self.flatten_layer(eingabe)
        x_kodiert = self.encoder(eingaben_formatiert)
        return x_kodiert

#Definiere die Verlustfunktion des CAE
#Diese besteht im ersten Teil aus dem MSE
#Im zweiten Teil brauchen wir die Gewichtsmatrix W_1
#Sowie die Kodierungen um die Ableitung von h_j in x_i zu bestimmen
#lambd stellt hierbei das lambda dar und wurde empirisch so bestimmt,
#dass ein Sparsenessfaktor von ca. 0.05 erhalten wird.
lambd = 0.0095
def loss(x, x_bar, h, model):
    rekonstruktions_verlust = tf.reduce_mean(
        tf.keras.losses.mse(x, x_bar)
    )
    W = tf.Variable(model.encoder.weights[0])
    dh = h * (1 - h)
    W = tf.transpose(W)
    kontraktiv_verlust = lambd * tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)
    verlust = rekonstruktions_verlust + kontraktiv_verlust
    return verlust

#Bestimmt die Gradienten des Autoencoders mit der Eingabe,
#sowie die Werte der Verlustfunktion
def grad(autoencoder, eingabe):
    with tf.GradientTape() as tape:
        rekonstruktion, eingaben_formatiert, kodiert = autoencoder.eval(eingabe)
        verlust = loss(eingaben_formatiert, rekonstruktion, kodiert, autoencoder)
    return verlust, tape.gradient(verlust, autoencoder.trainable_variables), rekonstruktion

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255

#Initialisiere den Autoencoder
autoencoder = AutoEncoder()
optimizer = tf.optimizers.Adam(learning_rate=0.001)
num_epochs = 300
batch_size = 2048
#Im folgenden Block wird der Autoencoder trainiert.
#Durch die speziellere Verlustfunktion werden die Werte mittels
#grad im aktuellen Trainingsschritt bestimmt und in den Optimizier gegeben,
#sodass dieser die Anpassungen der Parameter vornimmt.
for epoch in range(num_epochs):
    for x in range(0, len(x_train), batch_size):
        x_eingabe = x_train[x: x + batch_size]
        verlust, grads,_ = grad(autoencoder, x_eingabe)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))
    print("Epoch: {}, Loss: {}".format(epoch, tf.reduce_sum(verlust)))

#Kodiere die ersten 50 Daten des Test-Datensatzes
#Plotte diese spaltenweise nebeneinander,
#sodass Abb. 8 erzeugt wird.
kodiert = autoencoder.encode(x_test[0:50])
plt.figure(figsize=(8,8))
plt.imshow( np.transpose(np.array(kodiert).reshape((50,50))))
plt.gray()
plt.axis("off")
plt.show()








