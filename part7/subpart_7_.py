from tensorflow import keras
from tensorflow.keras import layers


def model_functional():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    model.build(input_shape=(None, 3))

    print(model.weights)
    print(model.summary())

    model = keras.Sequential(name="test")
    model.add(layers.Input(shape=(3,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(10, activation="softmax"))
    # model.build((None, 3))
    print(model.summary())


def example_7_9(_vocabulary_size, _num_tags, _num_department):
    title = keras.Input(shape=(_vocabulary_size,), name="title")
    text_body = keras.Input(shape=(_vocabulary_size,), name="text_body")
    tags = keras.Input(shape=(_num_tags,), name="tags")

    features = layers.Concatenate()([title, text_body, tags])
    features = layers.Dense(64, activation="relu")(features)

    priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
    department = layers.Dense(_num_department, activation="softmax", name="department")(features)

    model = keras.Model(inputs=[title, text_body, tags], outputs=[priority, department])
    return model


def gen_department_data(num_samples, vocabulary_size, num_tags, num_department):
    import numpy as np
    title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
    tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

    priority_data = np.random.random(size=(num_samples, 1))
    department_data = np.random.randint(0, 2, size=(num_samples, num_department))

    return title_data, text_body_data, tags_data, priority_data, department_data


def example_7_10():
    vocabulary_size = 10000
    num_tags = 100
    num_department = 4

    model = example_7_9(vocabulary_size, num_tags, num_department)

    num_samples = 1280

    title_data, text_body_data, tags_data, priority_data, department_data = (
        gen_department_data(num_samples, vocabulary_size, num_tags, num_department))

    model.compile(optimizer="rmsprop",
                  loss={"priority": "mean_squared_error", "department": "categorical_crossentropy"},
                  metrics={"priority": ["mean_absolute_error"], "department": ["accuracy"]})


    model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
              {"priority": priority_data, "department": department_data},
              epochs=1)

    model.evaluate({"title": title_data, "text_body": text_body_data, "tags": tags_data},
                   {"priority": priority_data, "department": department_data})

    priority_preds, department_preds = model.predict(
        {"title": title_data, "text_body": text_body_data, "tags": tags_data})

    keras.utils.plot_model(model)
    pass


example_7_10()
