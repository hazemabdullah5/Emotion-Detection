import cv2
import numpy as np
import tensorflow as tf

# 1. Load your trained Keras model
model = tf.keras.models.load_model("emotion_model60.h5")
# Ensure "emotion_model.h5" is in the same directory or provide the full path.

# 2. Define the class labels (adjust if your dataset has different labels)
# For FER-2013, typical order might be: [Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral]
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# 3. Initialize face detector (Haar cascade).
# You need the cascade XML file; if not installed, download from:
# https://github.com/opencv/opencv/tree/master/data/haarcascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 4. Open webcam
cap = cv2.VideoCapture(0)  # 0 = default laptop camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (for face detection, if using Haar Cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 5. Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        # Crop and preprocess the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize to the same size as your training images (48x48 if FER)
        face_roi_resized = cv2.resize(face_roi, (48, 48))

        # Scale pixel values (if your model was trained with normalization)
        face_roi_resized = face_roi_resized.astype("float32") / 255.0

        # Expand dims to match model's expected shape: (batch_size, height, width, channels)
        face_roi_resized = np.expand_dims(face_roi_resized, axis=-1)  # Add channel dimension (grayscale)
        face_roi_resized = np.expand_dims(face_roi_resized, axis=0)  # Add batch dimension

        # 6. Predict emotion
        preds = model.predict(face_roi_resized)
        emotion_index = np.argmax(preds[0])  # Index of max confidence
        emotion_label = class_labels[emotion_index]  # Map to label

        # Optionally get confidence
        confidence = preds[0][emotion_index]

        # 7. Draw bounding box and label on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{emotion_label} ({confidence * 100:.1f}%)"
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,  # font scale
            (0, 255, 0),  # color
            2  # thickness
        )

    # Show the output frame
    cv2.imshow("Emotion Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
