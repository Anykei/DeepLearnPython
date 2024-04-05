from tensorflow import keras
from tensorflow.keras import layers


def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)

    model = keras.Model(inputs, outputs)

    return model


def get_mnist_dataset():
    (images, labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    images = images.reshape((60000, 28 * 28)).astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255
    train_images, val_images = images[10000:], images[10000:]
    train_labels, val_labels = labels[10000:], labels[10000:]

    return train_images, val_images, train_labels, val_labels, test_images, test_labels
