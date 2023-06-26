import cv2 as cv
import numpy as np
import mediapipe as mp
import math

def get_eye_position(eye_center, iris_center, eye_radius):
    eye_vector = np.subtract(iris_center, eye_center)
    eye_angle = math.atan2(eye_vector[1], eye_vector[0])
    eye_direction = ""
    
    if eye_radius < 3:  # Eye closed or not detected
        eye_direction = "Closed or Not Detected"
    elif -0.25 * math.pi < eye_angle < 0.25 * math.pi:
        eye_direction = "Looking Forward"
    elif 0.25 * math.pi <= eye_angle < 0.75 * math.pi:
        eye_direction = "Looking Left"
    elif -0.75 * math.pi <= eye_angle < -0.25 * math.pi:
        eye_direction = "Looking Right"
    else:
        eye_direction = "Looking Up or Down"
    
    return eye_direction

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
cap = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA)
            
            left_eye_center = np.mean(mesh_points[LEFT_EYE], axis=0).astype(int)
            right_eye_center = np.mean(mesh_points[RIGHT_EYE], axis=0).astype(int)
            cv.circle(frame, tuple(left_eye_center), 2, (0, 255, 0), -1)
            cv.circle(frame, tuple(right_eye_center), 2, (0, 255, 0), -1)
            
            left_eye_direction = get_eye_position(left_eye_center, center_left, l_radius)
            right_eye_direction = get_eye_position(right_eye_center, center_right, r_radius)
            
            if left_eye_direction == "Looking Left" or right_eye_direction == "Looking Left":
                cv.putText(frame, "Looking Left", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif left_eye_direction == "Looking Right" or right_eye_direction == "Looking Right":
                cv.putText(frame, "Looking Right", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv.putText(frame, "Looking Forward", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
