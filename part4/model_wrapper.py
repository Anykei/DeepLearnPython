from matplotlib import pyplot as plt
from tensorflow import keras


class ModelWrapper:
    def __init__(self):
        self.models = {}
        self.hist = {}

    def compile(self, **kwargs):
        for k in self.models.keys():
            self.models[k].compile(**kwargs)

    def add_model(self, _model: keras.Sequential, _name: str = ''):
        if _name == '':
            _name = str(len(self.models) + 1)
        self.models[_name] = _model

    def fit(self, **kwargs):
        for k in self.models.keys():
            self.hist[k] = self.models[k].fit(**kwargs)

    def plot_hist(self):
        _, axs = plt.subplots(len(self.hist), 2, layout='constrained')
        for i, key in enumerate(self.hist.keys()):
            history_dict = self.hist[key].history
            loss_values = history_dict["loss"]
            val_loss_values = history_dict["val_loss"]
            epoch = range(1, len(loss_values) + 1)

            axs[i][0].plot(epoch, loss_values, 'bo', label="Потери на этапе обучения")
            axs[i][0].plot(epoch, val_loss_values, 'b', label="Потери на этапе проверки")

            axs[i][0].set(title=f"Потери на этапах обучения и проверки {key}")
            axs[i][0].set_xlabel("Эпохи")
            axs[i][0].set_ylabel("Потери")
            axs[i][0].grid(True)
            axs[i][0].legend()

            acc = history_dict["accuracy"]
            val_acc = history_dict["val_accuracy"]
            axs[i][1].plot(epoch, acc, "bo", label="Точность на этапе обучения")
            axs[i][1].plot(epoch, val_acc, "b", label="Точность на этапе проверки")
            axs[i][1].set(title=f"Точность на этапах обучения и проверки {key}")
            axs[i][1].set_xlabel("Эпохи")
            axs[i][1].set_ylabel("Точность")
            axs[i][1].grid(True)
            axs[i][1].legend()

        plt.show()
