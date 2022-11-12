import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
test = 'C:/Users/Fasil/Downloads/test.csv'

def parse_data_from_input(filename):
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        imgs = []
        next(csv_reader)
        for row in csv_reader:
            pixels = row[0:]
            img = np.array(pixels).reshape((28, 28))
            imgs.append(img)
        images = np.array(imgs, dtype='float64')
        return images
test_images = parse_data_from_input(test)

model = tf.keras.models.load_model('./digit_classifier_model/digits.h5')
'''epochs = range(len(model.history.history['loss']))
plt.plot(epochs, model.history.history['loss'])
plt.plot(epochs, model.history.history['accuracy'])
plt.show()'''

prediction = model.predict(test_images)
prediction = np.argmax(prediction,axis = 1)
df = pd.array(prediction)
results = pd.Series(prediction,name="Label")
prediction = pd.DataFrame(prediction, columns=['Label'])

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),prediction],axis = 1)

submission.to_csv("submission.csv",index=False)
