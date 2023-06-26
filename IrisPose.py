import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = mp_face_mesh.process(rgb_frame)

    # Check if face landmarks were detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)

            # Get eye landmarks
            left_eye_landmarks = face_landmarks.landmark[mp_face_mesh.LEFT_EYE]
            right_eye_landmarks = face_landmarks.landmark[mp_face_mesh.RIGHT_EYE]

            # Get pupil coordinates for left and right eyes
            left_pupil = (left_eye_landmarks[0].x, left_eye_landmarks[0].y)
            right_pupil = (right_eye_landmarks[0].x, right_eye_landmarks[0].y)

            # Print the pupil direction coordinates
            print("Left Pupil Direction: ({}, {})".format(left_pupil[0], left_pupil[1]))
            print("Right Pupil Direction: ({}, {})".format(right_pupil[0], right_pupil[1]))

    # Display the frame
    cv2.imshow('Pupil Direction Detection', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
