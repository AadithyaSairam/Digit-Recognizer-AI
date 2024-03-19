import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

model = tf.keras.models.load_model('handwritten2.keras')

def prediction(image, model):
    img = cv2.resize(image, (28, 28))
    img = img / 255
    img = img.reshape(1, 28, 28, 1)
    predict = model.predict(img)
    prob = np.amax(predict)
    class_index = np.argmax(model.predict(img), axis=-1)
    result = class_index[0]
    #if prob < 0.75:
    #    result = 0
    #    prob = 0
    return result, prob


while True:

    _, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    frame_copy = frame.copy()

    bbox_size = (40, 40)
    bbox = [(int(WIDTH // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
            (int(WIDTH // 2 + bbox_size[0] // 2), int(HEIGHT // 2 + bbox_size[1] // 2))]

    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (200, 200))
    cv2.imshow("cropped", img_gray)

    (result, probability) = prediction(img_gray, model)
    cv2.putText(frame_copy, f"Prediction: {result}", (40, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_copy, f"Probability: {probability}", (40, 80), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.rectangle(frame_copy, bbox[0], bbox[1], (0, 255, 0), 3)

    cv2.imshow("input", frame_copy)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()