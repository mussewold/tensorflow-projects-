import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import download_dataset
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import pathlib
import glob
from PIL import Image

dataset_dir = os.path.join("/Users/Fasil/Downloads/data")

'''from pathlib import Path
import imghdr

image_extensions = [".png", ".jpg"]  # add there all your images file extensions

img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for filepath in Path(dataset_dir).rglob("*"):
    if filepath.suffix.lower() in image_extensions:
        img_type = imghdr.what(filepath)
        if img_type is None:
            print(f"{filepath} is not an image")
            im = Image.open(filepath)
            rgb_im = im.convert("RGB")
            # exporting the image
            rgb_im.save(filepath)
        elif img_type not in img_type_accepted_by_tf:
            print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
            im = Image.open(filepath)
            rgb_im = im.convert("RGB")
            # exporting the image
            rgb_im.save(filepath)
'''

####----let's load the files using tensorflow----########
train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    seed=123,
    batch_size=16,
    validation_split = 0.2,
    subset='training',
    image_size=(300,300)
)
val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    seed=123,
    batch_size=16,
    validation_split = 0.2,
    subset='validation',
    image_size=(300,300)
)
#########-----building the model-------##############
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(300,300,3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
              ,metrics=['accuracy'])

history = model.fit(train_data, validation_data=val_data, epochs=10)

#########---plots-----##########
epochs = range(len(history.history['loss']))
plt.title('Accuracy')
plt.plot(epochs, history.history['accuracy'])
plt.plot(epochs, history.history['val_accuracy'])
plt.show()

plt.title('Loss')
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.show()



#######-----saving the model----------######
file_dir = './saved_model'
tf.keras.models.save_model(model, file_dir)
