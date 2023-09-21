import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model('handwritten.model')
image_number = 1

while os.path.isfile(f"numbers/num{image_number}.png"):
    try:
        img = cv2.imread(f"numbers/num{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        number = np.argmax(prediction)
        print(f"Written digit is {number}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number += 1