import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import TFSMLayer

#inference_layer = TFSMLayer('C:/Users/aadis/Documents/PROJECTS/Digit-Recognizer-AI/handwritten2.keras', call_endpoint='serving_default')

# Create a Keras model using the loaded inference-only layer
#model = tf.keras.Sequential([inference_layer])
model = tf.keras.models.load_model('handwritten2.keras')
image_number = 3

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