import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from part4.model_wrapper import ModelWrapper
from part4.tools import vectorize_seq, plot_history


def data_to_words_convert(_indexes):
    word_index = keras.datasets.reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in _indexes])
    print(decoded_newswire)


def random_class(_labels):
    import copy

    test_labels_copy = copy.copy(_labels)
    np.random.shuffle(test_labels_copy)
    hits_array = np.array(_labels) == np.array(test_labels_copy)
    print(hits_array.mean())


(train_data, train_labels), (test_data, test_labels) = keras.datasets.reuters.load_data(num_words=10000)

data_to_words_convert(train_data[0])

x_train = vectorize_seq(train_data)
x_test = vectorize_seq(test_data)

# y_train = keras.utils.to_categorical(train_labels)
# y_test = keras.utils.to_categorical(test_labels)

y_train = np.array(train_labels)
y_test = np.array(test_labels)

random_class(test_labels)

mw = ModelWrapper()

mw.add_model(keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax'),
]), "norm")

mw.add_model(keras.Sequential([
    layers.Dense(132, activation='relu'),
    layers.Dense(132, activation='relu'),
    layers.Dense(46, activation='softmax'),
]), "big_lay")

mw.add_model(keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(46, activation='softmax'),
]), "")

x_val = x_train[:1000]
part_x_train = x_train[1000:]

y_val = y_train[:1000]
part_y_train = y_train[1000:]

#  loss='categorical_crossentropy',
mw.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

mw.fit(x=part_x_train, y=part_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
mw.plot_hist()
