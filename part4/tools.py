import numpy as np

import matplotlib.pyplot as plt


def plot_history(_history):
    history_dict = _history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epoch = range(1, len(loss_values) + 1)

    _, axs = plt.subplots(1, 2, layout='constrained')

    axs[0].plot(epoch, loss_values, 'bo', label="Потери на этапе обучения")
    axs[0].plot(epoch, val_loss_values, 'b', label="Потери на этапе проверки")

    axs[0].set(title="Потери на этапах обучения и проверки")
    axs[0].set_xlabel("Эпохи")
    axs[0].set_ylabel("Потери")
    axs[0].grid(True)
    axs[0].legend()

    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    axs[1].plot(epoch, acc, "bo", label="Точность на этапе обучения")
    axs[1].plot(epoch, val_acc, "b", label="Точность на этапе проверки")
    axs[1].set(title="Точность на этапах обучения и проверки")
    axs[1].set_xlabel("Эпохи")
    axs[1].set_ylabel("Точность")
    axs[1].grid(True)
    axs[1].legend()

    plt.show()


def vectorize_seq(_seq, _dim=10000):
    result = np.zeros((len(_seq), _dim))
    for i, seq in enumerate(_seq):
        for j in seq:
            result[i, j] = 1
    return result


def to_one_hot(_labels, _dim=46):
    result = np.zeros((len(_labels), _dim))
    for i, label in enumerate(_labels):
        result[i, label] = 1.
    return result
