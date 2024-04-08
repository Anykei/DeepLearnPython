import pathlib

from tensorflow import keras
from tensorflow.keras import layers


new_base_dir = pathlib.Path("cat_vs_dog_small")


inputs = keras.Input(shape=(180, 180, 3))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, padding="same", kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

train = keras.utils.image_dataset_from_directory(new_base_dir/"train", image_size=(180, 180), batch_size=32)
validation = keras.utils.image_dataset_from_directory(new_base_dir/"train", image_size=(180, 180), batch_size=32)
test = keras.utils.image_dataset_from_directory(new_base_dir/"train", image_size=(180, 180), batch_size=32)
