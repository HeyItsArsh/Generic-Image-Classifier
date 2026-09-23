import tensorflow as tf
import sys

MODEL_PATH = "image_classifier.keras"
IMG_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)

class_names = open("classes.txt").read().splitlines()

image_path = sys.argv[1]

img = tf.keras.utils.load_img(
    image_path,
    target_size=IMG_SIZE
)

img_array = tf.keras.utils.img_to_array(img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)

predicted_index = tf.argmax(prediction[0]).numpy()

print("Prediction values:", prediction[0])
print("Classes:", class_names)
print("Predicted Class:", class_names[predicted_index])
