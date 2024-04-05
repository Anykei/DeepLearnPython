from tensorflow import keras
from tensorflow.keras import layers


class Classifier(keras.Model):
    def __init__(self, _num_classes=2):
        super().__init__()
        if _num_classes == 2:
            num_units = 1
            activation = "sigmoid"
        else:
            num_units = _num_classes
            activation = "softmax"

        self.dense = layers.Dense(num_units, activation=activation)

    def call(self, _inputs):
        return self.dense(_inputs)


inputs = keras.Input(shape=(3,))
features = layers.Dense(64, activation="relu")(inputs)
outputs = Classifier(_num_classes=10)(features)
model = keras.Model(inputs=inputs, outputs=outputs)

# -------------------------------

inputs = keras.Inputs(shape=(64,))
outputs = layers.Dense(1, activation="sigmoid")
binary_classifier = keras.Model(inputs=inputs, outputs=outputs)


class MyModel(keras.Model):
    def __init__(self, num_classes=2):
        super().__init__()
        self.dense = layers.Dense(64, activation="relu")
        self.classifier = binary_classifier

    def call(self, inputs):
        features = self.dense(inputs)
        return self.classifier(features)

model = MyModel()
