import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from part4.tools import vectorize_seq, plot_history

(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

x_train = vectorize_seq(train_data)
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
