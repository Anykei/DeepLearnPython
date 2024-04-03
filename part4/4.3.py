import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from part4.model_wrapper import ModelWrapper

(train_data, train_targets), (test_data, test_targets) = keras.datasets.boston_housing.load_data()

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
num_epoch = 100
all_scores = []
all_history = []

for i in range(k):
    print(f"Progressing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                            train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    history = model.fit(x=partial_train_data, y=partial_train_targets, epochs=num_epoch, batch_size=16, verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    all_scores.append(val_mae)

    all_history.append(history.history["mae"])

print(all_scores)
print(np.mean(all_scores))

average_history = [np.mean([x[i] for x in all_history]) for i in range(num_epoch)]

plt.plot(range(1, len(average_history) + 1), average_history)
plt.xlabel("Эпохи")
plt.ylabel("Оценка МАЕ")
plt.show()
