import json
import tensorflow as tf


def test_model_can_be_loaded():

    model = tf.keras.models.load_model("image_classifier.keras")

    assert model is not None


def test_classes_file_exists():

    with open("classes.json", encoding="utf-8") as file:
        classes = json.load(file)

    assert len(classes) > 0


def test_model_input_size():

    model = tf.keras.models.load_model("image_classifier.keras")

    assert model.input_shape[1:] == (224, 224, 3)
