import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# time = tf.Variable(2.)
# with tf.GradientTape() as outer_tape:
#     with tf.GradientTape() as inner_tape:
#         position = 4.9 * time ** 2
#     speed = inner_tape.gradient(position, time)
# acceleration = outer_tape.gradient(speed, time)

def get2classes(_num_samples_per_class):
    num_samples_per_class = _num_samples_per_class

    negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)
    positive_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)

    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
    targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                         np.ones((num_samples_per_class, 1), dtype="float32")))

    # plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
    # plt.show()

    return inputs, targets


def square_loss(_targets, _predictions):
    per_samples_loss = tf.square(_targets - _predictions)
    return tf.reduce_mean(per_samples_loss)


class LinearClassificator:
    def __init__(self, _input_dim, _output_dim, _learn_rate):
        self.W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
        self.b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))
        self.learn_rate = _learn_rate

    def model(self, _inputs):
        return tf.matmul(_inputs, self.W) + self.b

    def training_step(self, _inputs, _targets):
        with tf.GradientTape() as tape:
            predictions = self.model(_inputs)
            loss = square_loss(_predictions=predictions, _targets=_targets)
        grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [self.W, self.b])

        self.W.assign_sub(grad_loss_wrt_W * self.learn_rate)
        self.b.assign_sub(grad_loss_wrt_b * self.learn_rate)
        return loss


inputs, targets = get2classes(1000)

input_dim = 2
output_dim = 1

lc = LinearClassificator(input_dim, output_dim, 0.1)

for step in range(140):
    loss = lc.training_step(inputs, targets)
    print(f"Loss at step {step}: {loss:.4f}")

x = np.linspace(-1, 4, 100)
y = -lc.W[0] / lc.W[1] * x + (0.5 - lc.b) / lc.W[1]

plt.plot(x, y, "-r")
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])

pass
