# from tensorflow.python.client import device_lib
#
# print(device_lib.list_local_devices())

import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from tensorflow.keras.datasets import mnist


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(tf.test.gpu_device_name())

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0


model = keras.Sequential()
model.add(keras.Input(shape=(None, 28)))
# model.add(layers.SimpleRNN(256, return_sequences=True, activation='tanh'))
# model.add(layers.SimpleRNN(256, activation='tanh'))
# model.add(layers.GRU(256, return_sequences=True, activation='tanh'))
# model.add(layers.GRU(256, activation='tanh'))
# model.add(layers.LSTM(256, return_sequences=True, activation='tanh'))
# model.add(layers.LSTM(256, activation='tanh'))
model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True, activation='tanh')))
model.add(layers.Bidirectional(layers.LSTM(256, activation='tanh')))
model.add(layers.Dense(10))

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=1)
