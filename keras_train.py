import numpy as np
from cv2 import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_height = 100
img_width = 100
batch_size = 2

train_path = 'circuits/train'
valid_path = 'circuits/valid'

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
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

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
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

model = keras.Sequential([
    layers.Input((100,100, 1)),
    layers.Conv2D(16, 3),
    layers.Conv2D(32, 3),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(2),
])

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=[
        keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    ],
    metrics=["accuracy"],
)

model.fit(ds_train, epochs=5, verbose=2)

model.save("symbolsModel.h5")