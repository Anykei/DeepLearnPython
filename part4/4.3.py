from tensorflow import keras
from tensorflow.keras import layers

from part4.model_wrapper import ModelWrapper

(train_data, train_labels), (test_data, test_labels) = keras.datasets.boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


def build_model():
    model = keras.Sequential([layers.Dense(64, activation="relu"),
                              layers.Dense(64, activation="relu"),
                              layers.Dense(1)])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


k = 4

num_val_samples = len(train_data) // k
