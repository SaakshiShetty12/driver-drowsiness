import cv2
import numpy as np
import dlib
from imutils import face_utils
from pygame import mixer

mixer.init()
mixer.music.load("music.wav")

cap = cv2.VideoCapture(0)  # Initialize the camera capture

detector = dlib.get_frontal_face_detector()  # Initialize face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Initialize landmark predictor

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

def compute(pta, ptb):
    dist = np.linalg.norm(pta - ptb)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:
        break

    # Convert frame to grayscale if necessary
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)
    face_frame = frame.copy()  # Make a copy of the frame for drawing purposes

    left_blink = right_blink = -1  # Initialize blink variables
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # Determine status based on blink detection
    if left_blink == 0 or right_blink == 0:
        sleep += 1
        drowsy = 0
        active = 0
        if sleep > 6:
            status = "SLEEPING !!!"
            color = (255, 0, 0)
            if not mixer.music.get_busy():
                mixer.music.play(-1)

    elif left_blink == 1 or right_blink == 1:
        sleep = 0
        active = 0
        drowsy += 1
        if drowsy > 6:
            status = "Drowsy !"
            color = (0, 0, 255)
            if not mixer.music.get_busy():
                mixer.music.play(-1)

    elif left_blink == 2 or right_blink == 2:
        drowsy = 0
        sleep = 0
        active += 1
        if active > 6:
            status = "Active :)"
            color = (0, 255, 0)
            if mixer.music.get_busy():
                mixer.music.stop()

    cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Frame", frame)  # Display the original frame
    cv2.imshow("Result of detector", face_frame)  # Display the frame with detections

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
