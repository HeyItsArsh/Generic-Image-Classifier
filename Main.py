import tensorflow as tf
import matplotlib.pyplot as plt
from Model import create_model


DATASET_PATH = "dataset"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


class_names = train_ds.class_names

print("Classes:", class_names)


train_ds = train_ds.map(
    lambda x, y: (x / 255.0, y)
)

val_ds = val_ds.map(
    lambda x, y: (x / 255.0, y)
)


model = create_model(len(class_names))


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


loss, accuracy = model.evaluate(val_ds)

print("Validation Accuracy:", accuracy)


model.save("image_classifier.keras")


plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Training", "Validation"])

plt.show()


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Training", "Validation"])

plt.show()