import os
import sys
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.datasets import cifar10
# Human level accuracy is 94%

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print(tf.test.gpu_device_name())

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def my_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10)(x)
    model1 = keras.Model(inputs=inputs, outputs=outputs)
    return model1


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape)


x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='valid', activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ]
)


# using the function
model = my_model()

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=1)



