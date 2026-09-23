import tensorflow as tf


def test_model_can_be_loaded():

    model = tf.keras.models.load_model("image_classifier.keras")

    assert model is not None


def test_classes_file_exists():

    with open("classes.txt") as file:
        classes = file.read().splitlines()

    assert len(classes) > 0


def test_model_input_size():

    model = tf.keras.models.load_model("image_classifier.keras")

    assert model.input_shape[1:] == (128, 128, 3)
