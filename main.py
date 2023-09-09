import cv2
import sys
import numpy as np
import mediapipe as mp
import pandas as pd
sys.path.insert(0, '/Users/thomasli/Desktop/development/hackProject/GazeTracking/')
from gaze_tracking import GazeTracking
import time
import csv
import datetime
import os
import vlc

# Load facial landmarks model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

face_landmark_points = [1, 152, 33, 263, 57, 287]

# 3D model points from a canonical facial model
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left Mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

# 2D facial landmarks we'll use from MediaPipe model
face_landmark_points = [1, 152, 33, 263, 57, 287]

def get_head_pose(image, landmarks):
    image_pts = np.array([
        landmarks[1],  # Nose tip
        landmarks[152],  # Chin
        landmarks[33],  # Left eye left corner
        landmarks[263],  # Right eye right corner
        landmarks[57],  # Left mouth corner
        landmarks[287]  # Right mouth corner
    ], dtype="double")

    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_pts, camera_matrix, dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    # Project a 3D point (0, 0, 1000.0) onto the image plane to draw a line sticking out of the nose
    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                                              camera_matrix, dist_coeffs)

    p1 = (int(image_pts[0][0]), int(image_pts[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    return p1, p2

def get_eye_gaze(eye_points, gray):
    
    # Get the region of interest for the eye
    eye_region = np.array(eye_points, dtype=np.int32)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    # Crop the eye region
    gray_eye = gray[min_y: max_y, min_x: max_x]

    # Apply binary thresholding
    _, thresholded_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour, assuming it's the iris
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            # Get the center of the contour
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Use the centroid of the contour
            cx, cy = np.mean(largest_contour[:, 0], axis=0).astype(int)

        # Convert the pupil coordinates relative to the original image
        pupil = tuple(np.add((cx, cy), (min_x, min_y)))
        return pupil
    else:
        return None


#Time variables
start_time = int(time.time() * 1000)
cur_time = start_time
last_attention_time = start_time
last_distracted_time = start_time

#Timestamp varisbles
start_distracted = ''
end_distracted = ''

gaze = GazeTracking() #Init Gaze Tracking
cap = cv2.VideoCapture(0) # Capture video

lost_attention_threshold = 5000 #time distracted before being marked as not paying attention (ms)
paying_attention_threshold = 500 #time of paying attention before being marked as paying attention (ms)
paying_attention = True #boolean representing attention - true by default

gaze_threshold = 0.8 #how low gaze can be before being marked as distracted

#create file for logging timestamps
current_datetime = datetime.datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
folder_path = os.getcwd() + '/logs/'
filename = os.path.join(folder_path, f"log_{formatted_datetime}.csv")
header = ['id', 'start_distracted', 'end_distracted']
with open(filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(header)

#User ID.
user_id = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # results = face_mesh.process(frame_rgb)
    # if results.multi_face_landmarks:
    #     for facial_landmarks in results.multi_face_landmarks:
    #         landmarks = {i: (
    #         int(facial_landmarks.landmark[i].x * frame.shape[1]), int(facial_landmarks.landmark[i].y * frame.shape[0]))
    #                      for i in range(468)}

    #         left_eye = [landmarks[i] for i in range(133, 143)]
    #         right_eye = [landmarks[i] for i in range(362, 372)]

    #         left_pupil = get_eye_gaze(left_eye,gray)
    #         right_pupil = get_eye_gaze(right_eye,gray)

    #         cv2.circle(frame, left_pupil, 3, (0, 0, 255), -1)
    #         cv2.circle(frame, right_pupil, 3, (0, 0, 255), -1)

    #         p1, p2 = get_head_pose(frame, landmarks)
    #         cv2.line(frame, p1, p2, (255, 0, 0), 2)

    gaze.refresh(frame)

    text = ""

    vertical_ratio = gaze.vertical_ratio()
    cur_time = int(time.time() * 1000)
    frame_time = str(cur_time - start_time)

    print(vertical_ratio)

    if (paying_attention == False):
        if (cur_time % 5000 == 0): 
            p = vlc.MediaPlayer('notif.mp3')
            p.play()

    if (paying_attention):
        if (vertical_ratio != None and vertical_ratio < gaze_threshold):
            last_attention_time = cur_time
        elif (cur_time - last_attention_time > lost_attention_threshold):
            paying_attention = False
            start_distracted = cur_time
    else:
        if (vertical_ratio == None or vertical_ratio > gaze_threshold):
            last_distracted_time = cur_time
        elif (cur_time - last_distracted_time > paying_attention_threshold):
            paying_attention = True
            end_distracted = cur_time
            row = [user_id, start_distracted, end_distracted]
            with open(filename, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)

    cv2.putText(frame, str(vertical_ratio), (60, 120), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
    cv2.putText(frame, str(paying_attention), (60, 600), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)

    # print(f'Last Attention Time: {last_attention_time}')
    # print(f'Cur Time: ')

    cv2.imshow("Attention Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if (paying_attention == False):
            end_distracted = cur_time
            row = [user_id, start_distracted, end_distracted]
            with open(filename, 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)
        break

cap.release()
cv2.destroyAllWindows()

#Process timestamps
df = pd.read_csv(filename)

