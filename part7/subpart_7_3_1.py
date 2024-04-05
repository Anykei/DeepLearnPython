import keras.metrics
import tensorflow as tf


from part7.example_7_17 import get_mnist_model, get_mnist_dataset


class RootMeanSquare(keras.metrics.Metric):
    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="total_samples", initializer="zeros")
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros", dtype="int32")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)


# my_callbacks = {
#     keras.callbacks.EarlyStopping(patience=2),
#     keras.callbacks.ModelCheckpoint(filepath='~/repo/CNN/part7/model.{epoch:02d}-{val_loss:.2f}.keras'),
#     keras.callbacks.TensorBoard(log_dir='./logs'),
# }


train_images, val_images, train_labels, val_labels, test_images, test_labels = get_mnist_dataset()

model = get_mnist_model()
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy", RootMeanSquare()])

model.fit(x=train_images, y=train_labels, epochs=3, validation_data=(val_images, val_labels))
          # , callbacks=my_callbacks)

test_metrics = model.evaluate(test_images, test_labels)

pass
