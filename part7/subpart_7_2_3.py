from tensorflow import keras
from tensorflow.keras import layers

from part7 import subpart_7_


class CustomTicketModel(keras.model):
    def __init__(self, _num_department):
        super().__init__()
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(_num_department, activation="softmax")

    def call(self, inputs):
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]

        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)

        priority = self.priority_scorer(features)
        department = self.department_classifier(features)

        return priority, department


vocabulary_size = 10000
num_tags = 100
num_department = 4
num_samples = 1280

title_data, text_body_data, tags_data, priority_data, department_data = (
    subpart_7_.gen_department_data(num_samples, vocabulary_size, num_tags, num_department))

model = CustomTicketModel(_num_department=num_department)

model.compile(optimizer="rmsprop",
                  loss={"priority": "mean_squared_error", "department": "categorical_crossentropy"},
                  metrics={"priority": ["mean_absolute_error"], "department": ["accuracy"]})

model.fit({"title": title_data, "text_body": text_body_data, "tags": tags_data},
          {"priority": priority_data, "department": department_data},
          epochs=1)

# model.evaluate([title_data, text_body_data, tags_data], [priority_data, department_data])
model.evaluate({"title": title_data, "text_body": text_body_data, "tags": tags_data},
               {"priority": priority_data, "department": department_data})

priority_preds, department_preds = model.predict(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data})
