import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#loading the happy_vs_sad model


model = tf.keras.models.load_model('./saved_model', compile=True)
##########-----Let's predict on new photo---------########
path = "/Users/Fasil/Downloads/woman-g9fdf7d73a_640.jpg"
img = tf.keras.preprocessing.image.load_img(path, target_size=(300,300))
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])

if classes[0]>0.5:
    print("is a sad")
else:
    print("is a happy")
plt.imshow(img)
plt.show()