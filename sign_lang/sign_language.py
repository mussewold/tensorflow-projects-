import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import zipfile
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator

zip_ref = zipfile.ZipFile('C:/Users/Fasil/Downloads/archive (6).zip', 'r')
zip_ref.extractall('C:/Users/Fasil/PycharmProjects/testProject/sign_language')

dataset_dir = 'C:/Users/Fasil/PycharmProjects/testProject/sign_language/'
train_dir = os.path.join(dataset_dir,'sign_mnist_train/sign_mnist_train.csv')
val_dir = os.path.join(dataset_dir, 'sign_mnist_test/sign_mnist_test.csv')
######----loading the data----#####
with open(train_dir) as file:
    reader = csv.reader(file)
    train_labels=[]
    imgs=[]
    next(reader)
    for row in reader:
        train_labels.append(row[0])
        pixels = row[1:]
        img = np.array(pixels).reshape((28,28))
        imgs.append(img)
    train_labels = np.array(train_labels).astype('float64')
    train_images = np.array(imgs).astype('float64')

with open(val_dir) as file:
    reader = csv.reader(file)
    val_labels = []
    imgs = []
    next(reader)
    for row in reader:
        val_labels.append(row[0])
        pixels = row[1:]
        img = np.array(pixels).reshape((28,28))
        imgs.append(img)
    val_labels = np.array(val_labels).astype('float64')
    val_images = np.array(imgs).astype('float64')


######3----------Making the data generators----------##########
train_images = np.expand_dims(train_images, axis=3)
val_images = np.expand_dims(val_images, axis=3)
train_data = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)
train_gen = train_data.flow(
    x=train_images,
    y=train_labels,
    batch_size=32
)

val_data = ImageDataGenerator(
    rescale=1./255)
val_gen = val_data.flow(
    x=val_images,
    y=val_labels,
    batch_size=32
)

########--------Model_-------#######
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])
history = model.fit(train_gen,
                    epochs=15,
                    validation_data=val_gen)


# Plot the chart for accuracy and loss on both training and validation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()









