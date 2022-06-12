# Prerequisites for codes

import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.datasets import mnist

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(tf.test.gpu_device_name())

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)


x_train = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28*28).astype("float32")/255.0

# sequential API
# not flexible but convinient

model = keras.Sequential(
        [
            keras.Input(shape=(28*28)),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(10)
        ]
)


# Another way to code
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu'))
print(model.summary())
model.add(layers.Dense(256, activation='relu', name='selected_layer'))
model.add(layers.Dense(10))

# print(model.summary())
# sys.exit()

# ***one way to debug***
# model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output])
# model.layers[-2].output --> is equivalent to --> model.getlayer('selected_layer').output
# feature = model.predict(x_train)
# print(feature)
# you can also get all features in a single line - watch video
# model = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
# features = model,predict(x_train)
# for feature in features:
#       print(feature)
# sys.exit()

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    #from_logits --> output activation here or no (no assumes you already have it in model) default - false
    optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    metrics = ['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)

sys.exit()



#Functional API
#very flexible
inputs = keras.Input(shape=(28*28))
x = layers.Dense(512, activation='relu', name='first_dense_layer')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

pruint(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    optimizer = keras.optimizers.Adam(learning_rate=0.0001),
    metrics = ['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)