import cv2
import numpy as np
import tensorflow as tf

# Check if GPU is available and configure TensorFlow to use it
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Load the pre-trained model
model = tf.keras.models.load_model('handwritten2.keras')

# Initialize the webcam
cap = cv2.VideoCapture(0 + cv2.CAP_DSHOW)

# Set desired frame rate (fps)
desired_fps = 10

# Variables for frame rate control
frame_interval = int(1000 / desired_fps)
prev_time = 0

# Define the minimum area threshold for contours
min_area_threshold = 100

# Define the minimum solidity threshold for contours
min_solidity_threshold = 0.50

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours on the thresholded image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Filter contours based on size and solidity
    digit_contours = []
    if cnts:
        for c in cnts:
            area = cv2.contourArea(c)
            if area > min_area_threshold:  # Set a minimum area threshold
                # Calculate the solidity
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                if solidity > min_solidity_threshold:  # Set a minimum solidity threshold
                    digit_contours.append(c)

    # Draw contours on the original frame
    if digit_contours:
        for c in digit_contours:
            # Get bounding box coordinates
            x, y, w, h = cv2.boundingRect(c)
            
            # Extract digit region of interest (ROI)
            digit_roi = gray[y:y + h, x:x + w]

            # Resize ROI to 28x28
            digit_roi_resized = cv2.resize(digit_roi, (28, 28))

            # Prepare ROI for prediction
            digit_roi_resized = np.expand_dims(digit_roi_resized, axis=0)
            digit_roi_resized = np.expand_dims(digit_roi_resized, axis=-1)
            digit_roi_rescaled = digit_roi_resized / 255.0

            # Make prediction
            prediction = model.predict(digit_roi_rescaled)
            prob = np.amax(prediction)
            predicted_label = np.argmax(prediction)

            # Display predicted label on the frame
            if prob > 0.9:
                cv2.putText(frame, str(predicted_label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #cv2.putText(frame, str(prob), (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('frame', frame)

    # Control frame rate
    current_time = cv2.getTickCount()
    if cv2.waitKey(max(1, frame_interval - int((current_time - prev_time) * 1000 / cv2.getTickFrequency()))) & 0xFF == ord('q'):
        break
    prev_time = current_time

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
