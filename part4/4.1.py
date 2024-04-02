import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def plot_history(_history):
    history_dict = _history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epoch = range(1, len(loss_values) + 1)

    plt.plot(epoch, loss_values, 'bo', label="Потери на этапе обучения")
    plt.plot(epoch, val_loss_values, 'b', label="Потери на этапе проверки")
    plt.title("Потери на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.ylabel("Потери")
    plt.legend()
    plt.show()

    plt.clf()
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    plt.plot(epoch, acc, "bo", label="Точность на этапе обучения")
    plt.plot(epoch, val_acc, "b", label="Точность на этапе проверки")
    plt.title("Точность на этапах обучения и проверки")
    plt.xlabel("Эпохи")
    plt.ylabel("Точность")
    plt.legend()
    plt.show()


def vectorize_seq(_seq, _dim=10000):
    result = np.zeros((len(_seq), _dim))
    for i, seq in enumerate(_seq):
        for j in seq:
            result[i, j] = 1
    return result


(traint_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

x_train = vectorize_seq(traint_data)
x_test = vectorize_seq(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

x_val = x_train[:10000]
part_x_train = x_train[10000:]

y_val = y_train[:10000]
part_y_train = y_train[10000:]

history = model.fit(part_x_train, part_y_train, epochs=8, batch_size=512, validation_data=(x_val, y_val))
r = model.evaluate(x_test, y_test)
print(r)

plot_history(history)

pass
