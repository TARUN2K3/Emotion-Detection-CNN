from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import json

app = Flask(__name__)

# Define the dictionary mapping emotion labels to their corresponding emotions
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON configuration and create the model
with open(r'C:\Users\Nitin\Desktop\CNN-Github\emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Create a Keras model from the JSON configuration
emotion_model = model_from_json(loaded_model_json)

# Load the model weights
emotion_model.load_weights(r'C:\Users\Nitin\Desktop\CNN-Github\emotion_model.h5')
print("Loaded model from disk")

# Function to perform emotion detection
def detect_emotions():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces available on camera
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Take each face available on the camera and preprocess it
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

# Route to stream video with emotion detection
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotions(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
