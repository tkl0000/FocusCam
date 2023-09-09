import cv2
import numpy as np
import mediapipe as mp

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


# Capture video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for facial_landmarks in results.multi_face_landmarks:
            landmarks = {i: (
            int(facial_landmarks.landmark[i].x * frame.shape[1]), int(facial_landmarks.landmark[i].y * frame.shape[0]))
                         for i in range(468)}

            left_eye = [landmarks[i] for i in range(133, 143)]
            right_eye = [landmarks[i] for i in range(362, 372)]

            left_pupil = get_eye_gaze(left_eye,gray)
            right_pupil = get_eye_gaze(right_eye,gray)

            cv2.circle(frame, left_pupil, 3, (0, 0, 255), -1)
            cv2.circle(frame, right_pupil, 3, (0, 0, 255), -1)

            p1, p2 = get_head_pose(frame, landmarks)
            cv2.line(frame, p1, p2, (255, 0, 0), 2)

    cv2.imshow("Attention Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()