import numpy as np
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, preprocessing
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import RandomFlip

img_height = 150
img_width = 150
comp_count = 8
batch_size = comp_count

images_path = 'circuits/cnn/train'

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    images_path,
    labels="inferred",
    label_mode="int",
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
)

ds_valid = tf.keras.preprocessing.image_dataset_from_directory(
    images_path,
    labels="inferred",
    label_mode="int",
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height,img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
)

print(len(ds_train))
print(len(ds_valid))

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("vertical", input_shape=(img_height, img_width,1)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomTranslation(0.1, 0.1),
    ])


model = keras.Sequential([
    data_augmentation,
    layers.Input((img_height,img_width, 1)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(comp_count),
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics=["accuracy"],
)

model.fit(ds_train, validation_data=ds_valid, epochs=40, verbose=2)

model.save("symbolsModel.h5")