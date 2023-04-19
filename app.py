import cv2
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load the Haar cascade classifier for face detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open the webcam
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame using the Haar cascade
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        # Loop over each face and draw a bounding box around it
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Update the number of people detected
        num_people = len(faces)

        # Draw the number of people detected on the frame
        cv2.putText(frame, "Number of people: " + str(num_people), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Convert the frame to a byte array
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
