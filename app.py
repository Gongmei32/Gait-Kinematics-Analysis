import time
import csv
from datetime import datetime
import cv2
import numpy as np
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# ---------------------------------------------------------
# 1. Pipeline Initializations
# ---------------------------------------------------------
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Lite model for max FPS
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def compute_joint_angle(a, b, c):
    a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
    vec_ba = a - b
    vec_bc = c - b
    cosine_angle = np.dot(vec_ba, vec_bc) / (np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

def compute_symmetry_index(left_val, right_val):
    mean_val = 0.5 * (left_val + right_val)
    if mean_val < 1e-4:
        return 0.0
    return float((abs(left_val - right_val) / mean_val) * 100.0)

# ---------------------------------------------------------
# 2. Gait State Tracking Variables
# ---------------------------------------------------------
step_count = 0
last_step_time = time.time()
cadence_spm = 0.0

# Leg states: 'STANCE' vs 'SWING'
l_leg_state = "STANCE"
r_leg_state = "STANCE"

# Data logging buffer
session_records = []
start_session_time = time.time()

# ---------------------------------------------------------
# 3. Stream & Window Configuration
# ---------------------------------------------------------
VIDEO_SOURCE = 1  # 0 for webcam, 1 for phone USB, or phone stream URL
cap = cv2.VideoCapture(VIDEO_SOURCE)

WINDOW_NAME = "MTech Gait Kinematics & Rehabilitation Telemetry"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

prev_frame_time = time.time()
fps = 0.0

print("[INFO] Gait Engine running. Stand back so full legs are visible. Press 'q' to stop & export.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (960, 540))

    # FPS Calculation
    curr_time = time.time()
    dt = curr_time - prev_frame_time
    prev_frame_time = curr_time
    fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else 0.0

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = pose.process(rgb_frame)
    rgb_frame.flags.writeable = True

    l_knee_angle = 0.0
    r_knee_angle = 0.0
    l_hip_angle = 0.0
    r_hip_angle = 0.0
    knee_si = 0.0

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # Extract normalized coordinates
        l_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_hip      = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee     = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle    = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        r_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_hip      = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_knee     = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_ankle    = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate joint kinematics
        l_knee_angle = compute_joint_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = compute_joint_angle(r_hip, r_knee, r_ankle)
        l_hip_angle  = compute_joint_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle  = compute_joint_angle(r_shoulder, r_hip, r_knee)
        knee_si      = compute_symmetry_index(l_knee_angle, r_knee_angle)

        # -----------------------------------------------------
        # Gait Phase Segmentation & Step / Cadence Tracking
        # Knee angles < 145° indicate active swing / flexion phase
        # -----------------------------------------------------
        # Left Leg Phase Transition
        if l_knee_angle < 145.0 and l_leg_state == "STANCE":
            l_leg_state = "SWING"
        elif l_knee_angle >= 160.0 and l_leg_state == "SWING":
            l_leg_state = "STANCE"
            step_count += 1
            step_interval = curr_time - last_step_time
            if 0.2 < step_interval < 3.0:
                cadence_spm = 60.0 / step_interval
            last_step_time = curr_time

        # Right Leg Phase Transition
        if r_knee_angle < 145.0 and r_leg_state == "STANCE":
            r_leg_state = "SWING"
        elif r_knee_angle >= 160.0 and r_leg_state == "SWING":
            r_leg_state = "STANCE"
            step_count += 1
            step_interval = curr_time - last_step_time
            if 0.2 < step_interval < 3.0:
                cadence_spm = 60.0 / step_interval
            last_step_time = curr_time

        # Record metrics for dissertation report export
        session_records.append([
            round(curr_time - start_session_time, 2),
            round(l_knee_angle, 1),
            round(r_knee_angle, 1),
            round(knee_si, 1),
            l_leg_state,
            r_leg_state,
            step_count,
            round(cadence_spm, 1)
        ])

        # Draw anatomical skeletal lines
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # ---------------------------------------------------------
    # 4. Clinical Heads-Up Display (HUD)
    # ---------------------------------------------------------
    # Telemetry card background
    cv2.rectangle(frame, (15, 15), (370, 245), (18, 18, 18), -1)
    cv2.rectangle(frame, (15, 15), (370, 245), (0, 255, 180), 1)

    cv2.putText(frame, "CLINICAL GAIT TELEMETRY", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 180), 2)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (295, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # Angles
    cv2.putText(frame, f"Left Knee : {l_knee_angle:5.1f} deg ({l_leg_state})", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    cv2.putText(frame, f"Right Knee: {r_knee_angle:5.1f} deg ({r_leg_state})", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

    # Spatiotemporal metrics
    cv2.putText(frame, f"Total Steps  : {step_count}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
    cv2.putText(frame, f"Cadence (SPM): {cadence_spm:4.1f} steps/min", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

    # Symmetry Index
    si_color = (0, 255, 0) if knee_si < 10.0 else ((0, 255, 255) if knee_si < 20.0 else (0, 0, 255))
    cv2.putText(frame, f"Symmetry Dev : {knee_si:5.1f}%", (30, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.52, si_color, 2)
    
    cv2.putText(frame, "Press 'q' to stop & export CSV", (30, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# 5. Export Clinical Session Data
# ---------------------------------------------------------
if session_records:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"gait_session_{timestamp}.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time_Sec", "L_Knee_Angle", "R_Knee_Angle", "Knee_SI_Pct", "L_State", "R_State", "Steps", "Cadence_SPM"])
        writer.writerows(session_records)
    print(f"\n[SUCCESS] Session telemetry saved to: {csv_filename}")