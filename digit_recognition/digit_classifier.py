import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = 'C:/Users/Fasil/Downloads/train.csv'
test = 'C:/Users/Fasil/Downloads/test.csv'


def parse_data_from_input(filename):
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        labels = []
        imgs = []
        next(csv_reader)
        for row in csv_reader:
            labels.append(row[0])
            pixels = row[1:]
            img = np.array(pixels).reshape((28, 28))
            imgs.append(img)
        images = np.array(imgs, dtype='float64')
        labels = np.array(labels).astype('float64')

        return images, labels

# writing an early stop callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch, log={}):
        if(log.get('accuracy')>=0.99):
            print("\nAn accuracy of 0.99 achieved...haulting training")
            self.model.stop_training = True
mycallback = myCallback()
images, labels=parse_data_from_input(train)
###---train_val split
percent=0.8
split_size = int(len(labels)*percent)
print(split_size)
train_images=images[:split_size]
train_labels=labels[:split_size]
val_images=images[split_size:]
val_labels=labels[split_size:]
#creating the generator
train_images = np.expand_dims(train_images, axis=3)
train_datagen = ImageDataGenerator(rescale=1./255)
val_images = np.expand_dims(val_images, axis=3)
val_datagen = ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 64
train_generator = train_datagen.flow(x=train_images, y=train_labels, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(x=val_images, y=val_labels, batch_size=BATCH_SIZE)
## building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'],
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[mycallback])
model.save('./digit_classifier_model/digits.h5')
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, 'loss')
plot_graphs(history, 'accuracy')