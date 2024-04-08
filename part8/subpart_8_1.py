import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from part7.example_7_17 import get_mnist_dataset, get_raw_mnist_dataset

inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(filters=32, padding="same", kernel_size=3, activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

(images, labels), (test_images, test_labels) = get_raw_mnist_dataset()
train_images = images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(train_images, labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)
