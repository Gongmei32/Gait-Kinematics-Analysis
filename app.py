import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import time
import csv
import os
from datetime import datetime
import cv2
import numpy as np
import mediapipe.python.solutions.pose as mp_pose
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

# ---------------------------------------------------------
# 1. Pipeline Initializations & Mathematical Models
# ---------------------------------------------------------
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Lite model optimized for real-time throughput
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def compute_joint_angle(a, b, c):
    """Computes planar angle at vertex b using standard vector dot product."""
    a, b, c = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), np.array(c, dtype=np.float32)
    vec_ba = a - b
    vec_bc = c - b
    cosine_angle = np.dot(vec_ba, vec_bc) / (np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))

def compute_symmetry_index(left_val, right_val):
    """Calculates Robinson's Symmetry Index (SI %)."""
    mean_val = 0.5 * (left_val + right_val)
    if mean_val < 1e-4:
        return 0.0
    return float((abs(left_val - right_val) / mean_val) * 100.0)

# ---------------------------------------------------------
# 2. Futuristic HUD Rendering Engine
# ---------------------------------------------------------
class BiometricHUD:
    def __init__(self):
        # Color Palette in OpenCV BGR format
        self.OBSIDIAN     = (14, 12, 10)       # Deep translucent slate
        self.NEON_CYAN    = (255, 230, 0)      # Primary active metric (#00E6FF)
        self.CYAN_MUTED   = (180, 150, 40)     # Dim cyan for borders/grids
        self.ALERT_AMBER  = (0, 179, 255)      # Cautionary asymmetry (#FFB300)
        self.ALERT_CORAL  = (80, 60, 255)      # Critical asymmetry (#FF3C50)
        self.TEXT_WHITE   = (245, 245, 245)
        self.TEXT_MUTED   = (150, 150, 150)
        self.BORDER_GRAY  = (45, 45, 45)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_glass_card(self, frame, x, y, w, h, alpha=0.72):
        """Renders an alpha-blended glassmorphic panel with tactical notched corners."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.OBSIDIAN, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Subtle outline
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.BORDER_GRAY, 1, cv2.LINE_AA)

        # Tactical corner brackets
        k_len = 10
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for cx, cy in corners:
            dx = k_len if cx == x else -k_len
            dy = k_len if cy == y else -k_len
            cv2.line(frame, (cx, cy), (cx + dx, cy), self.NEON_CYAN, 2, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + dy), self.NEON_CYAN, 2, cv2.LINE_AA)

    def draw_joint_target(self, frame, center_px, angle_deg, label="KNEE", side="L"):
        """Draws a concentric ring target and callout vector directly at the joint."""
        cx, cy = center_px
        
        # Concentric rings
        cv2.circle(frame, (cx, cy), 14, self.CYAN_MUTED, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 6, self.NEON_CYAN, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, self.TEXT_WHITE, -1, cv2.LINE_AA)

        # Angled leader line & callout box
        offset_x = -110 if side == "L" else 30
        bx = cx + offset_x
        by = cy - 25

        cv2.line(frame, (cx, cy), (bx + (70 if side == "L" else 0), by), self.NEON_CYAN, 1, cv2.LINE_AA)
        cv2.line(frame, (bx, by), (bx + 80, by), self.NEON_CYAN, 1, cv2.LINE_AA)

        cv2.putText(frame, f"{label}: {int(angle_deg)} DEG", (bx + 2, by - 6),
                    self.font, 0.38, self.TEXT_WHITE, 1, cv2.LINE_AA)

    def render_hud(self, frame, patient_name, fps, l_knee, r_knee, l_state, r_state, 
                   step_count, cadence, knee_si, l_peak_rom, r_peak_rom, legs_visible):
        """Assembles the primary top-left biometric telemetry HUD."""
        x, y, w, h = 20, 20, 340, 275
        self.draw_glass_card(frame, x, y, w, h, alpha=0.75)

        # Header with pulsating active tag
        cv2.putText(frame, "BIOMETRIC GAIT TELEMETRY", (x + 16, y + 26), 
                    self.font, 0.50, self.NEON_CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS {fps:4.1f}", (x + w - 75, y + 26), 
                    self.font, 0.40, self.TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.line(frame, (x + 16, y + 36), (x + w - 16, y + 36), self.BORDER_GRAY, 1, cv2.LINE_AA)

        # Patient Identifier
        cv2.putText(frame, f"SUBJECT // {patient_name.upper()}", (x + 16, y + 56), 
                    self.font, 0.40, self.CYAN_MUTED, 1, cv2.LINE_AA)

        if not legs_visible:
            cv2.putText(frame, "TARGET LOCK REQUIRED", (x + 16, y + 90), 
                        self.font, 0.55, self.ALERT_CORAL, 2, cv2.LINE_AA)
            cv2.putText(frame, "Lower limbs out of frame bounds", (x + 16, y + 115), 
                        self.font, 0.38, self.TEXT_MUTED, 1, cv2.LINE_AA)
        else:
            # Joint State & Flexion Angles
            l_info = f"L-KNEE: {l_knee:5.1f} DEG [{l_state[:2]}]"
            r_info = f"R-KNEE: {r_knee:5.1f} DEG [{r_state[:2]}]"
            cv2.putText(frame, l_info, (x + 16, y + 84), self.font, 0.42, self.TEXT_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, r_info, (x + 16, y + 106), self.font, 0.42, self.TEXT_WHITE, 1, cv2.LINE_AA)

            # Peak ROM Metric (Option B integration)
            rom_txt = f"PEAK ROM  L: {l_peak_rom:4.1f} | R: {r_peak_rom:4.1f} DEG"
            cv2.putText(frame, rom_txt, (x + 16, y + 130), self.font, 0.38, self.CYAN_MUTED, 1, cv2.LINE_AA)

        cv2.line(frame, (x + 16, y + 144), (x + w - 16, y + 144), self.BORDER_GRAY, 1, cv2.LINE_AA)

        # Cadence & Total Step Counter
        cv2.putText(frame, "CADENCE", (x + 16, y + 168), self.font, 0.36, self.TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{cadence:4.1f} SPM", (x + 16, y + 196), self.font, 0.80, self.TEXT_WHITE, 2, cv2.LINE_AA)

        cv2.putText(frame, "TOTAL STEPS", (x + 180, y + 168), self.font, 0.36, self.TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{step_count:03d}", (x + 180, y + 196), self.font, 0.80, self.NEON_CYAN, 2, cv2.LINE_AA)

        # Symmetry Index (SI) Color-Coded Ring & Alert
        if knee_si <= 8.0:
            si_color = self.NEON_CYAN
            si_status = "SYMMETRY: NOMINAL"
        elif knee_si <= 15.0:
            si_color = self.ALERT_AMBER
            si_status = "SYMMETRY: DRIFT DETECTED"
        else:
            si_color = self.ALERT_CORAL
            si_status = "SYMMETRY: ASYMMETRIC BIAS"

        cv2.line(frame, (x + 16, y + 210), (x + w - 16, y + 210), self.BORDER_GRAY, 1, cv2.LINE_AA)
        cv2.circle(frame, (x + 24, y + 235), 5, si_color, -1, cv2.LINE_AA)
        cv2.putText(frame, f"SI: {knee_si:4.1f}% - {si_status}", (x + 38, y + 239), 
                    self.font, 0.38, si_color, 1, cv2.LINE_AA)

        cv2.putText(frame, "[Q] SECURE TERMINATION & LOG", (x + 16, y + 262), 
                    self.font, 0.34, self.TEXT_MUTED, 1, cv2.LINE_AA)

# ---------------------------------------------------------
# 3. Patient Configuration & Runtime State Setup
# ---------------------------------------------------------
hud = BiometricHUD()

patient_input = input("\nEnter Patient Name / ID (Press Enter for 'Anonymous'): ").strip()
patient_name = patient_input.replace(" ", "_") if patient_input else "Anonymous"
display_patient_name = patient_input if patient_input else "Anonymous"

step_count = 0
last_step_time = 0.0
cadence_spm = 0.0

l_leg_state = "STANCE"
r_leg_state = "STANCE"

# Option B: Peak ROM trackers (minimum extension to peak swing flexion)
l_peak_rom = 0.0
r_peak_rom = 0.0

MIN_STEP_INTERVAL = 0.35    # Min interval between step events (seconds)
MAX_STEP_INTERVAL = 3.5     # Upper bound for active stride timing
VISIBILITY_THRESHOLD = 0.65 # Confidence gate for knee/ankle keypoints

session_records = []
start_session_time = time.time()

# ---------------------------------------------------------
# 4. Stream Source Allocation
# ---------------------------------------------------------
print("\n--- Video Input Configuration ---")
print("Options:")
print("  [Enter]   : Live Phone Camera (Device index 1)")
print("  0         : Built-in Webcam")
print("  <file>    : Path to video (e.g., demo_walk.mp4)")
source_in = input("Select input source: ").strip()

is_video_file = False
if not source_in:
    VIDEO_SOURCE = 1
elif source_in.isdigit():
    VIDEO_SOURCE = int(source_in)
else:
    VIDEO_SOURCE = source_in
    is_video_file = True

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"[ERROR] Could not open video source: {VIDEO_SOURCE}")
    exit(1)

WINDOW_NAME = "MTech Gait Kinematics & Rehabilitation Telemetry"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 540)

prev_frame_time = time.time()
fps = 0.0

print(f"\n[INFO] Session initiated for: {display_patient_name}")
print(f"[INFO] Source: {'Video File (' + str(VIDEO_SOURCE) + ')' if is_video_file else 'Live Stream'}")
print("[INFO] Stand back so lower limbs are framed. Press 'q' to stop & export.")

# ---------------------------------------------------------
# 5. Core Kinematic Execution Loop
# ---------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()

    if not success:
        if is_video_file:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()
            if not success:
                break
        else:
            break

    frame = cv2.resize(frame, (960, 540))
    fh, fw, _ = frame.shape

    # Real-time smoothed FPS
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
    knee_si = 0.0
    legs_visible = False
    l_knee_px = None
    r_knee_px = None

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        l_hip_lm   = lm[mp_pose.PoseLandmark.LEFT_HIP.value]
        l_knee_lm  = lm[mp_pose.PoseLandmark.LEFT_KNEE.value]
        l_ankle_lm = lm[mp_pose.PoseLandmark.LEFT_ANKLE.value]

        r_hip_lm   = lm[mp_pose.PoseLandmark.RIGHT_HIP.value]
        r_knee_lm  = lm[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        r_ankle_lm = lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

        left_visible = (l_knee_lm.visibility > VISIBILITY_THRESHOLD and l_ankle_lm.visibility > VISIBILITY_THRESHOLD)
        right_visible = (r_knee_lm.visibility > VISIBILITY_THRESHOLD and r_ankle_lm.visibility > VISIBILITY_THRESHOLD)

        if left_visible and right_visible:
            legs_visible = True

            l_hip   = [l_hip_lm.x, l_hip_lm.y]
            l_knee  = [l_knee_lm.x, l_knee_lm.y]
            l_ankle = [l_ankle_lm.x, l_ankle_lm.y]

            r_hip   = [r_hip_lm.x, r_hip_lm.y]
            r_knee  = [r_knee_lm.x, r_knee_lm.y]
            r_ankle = [r_ankle_lm.x, r_ankle_lm.y]

            # Pixel conversions for anchored HUD targets
            l_knee_px = (int(l_knee[0] * fw), int(l_knee[1] * fh))
            r_knee_px = (int(r_knee[0] * fw), int(r_knee[1] * fh))

            l_knee_angle = compute_joint_angle(l_hip, l_knee, l_ankle)
            r_knee_angle = compute_joint_angle(r_hip, r_knee, r_ankle)
            knee_si      = compute_symmetry_index(l_knee_angle, r_knee_angle)

            # Option B: Track Peak Flexion during active swing
            l_peak_rom = max(l_peak_rom, 180.0 - l_knee_angle)
            r_peak_rom = max(r_peak_rom, 180.0 - r_knee_angle)

            # -----------------------------------------------------
            # Gait Phase Segmentation & Step Cadence Detection
            # -----------------------------------------------------
            step_triggered = False

            # Left Limb Event Triggers
            if l_knee_angle < 135.0 and l_leg_state == "STANCE":
                l_leg_state = "SWING"
            elif l_knee_angle >= 160.0 and l_leg_state == "SWING":
                if (curr_time - last_step_time) >= MIN_STEP_INTERVAL:
                    l_leg_state = "STANCE"
                    step_count += 1
                    step_triggered = True

            # Right Limb Event Triggers
            if r_knee_angle < 135.0 and r_leg_state == "STANCE":
                r_leg_state = "SWING"
            elif r_knee_angle >= 160.0 and r_leg_state == "SWING":
                if (curr_time - last_step_time) >= MIN_STEP_INTERVAL:
                    r_leg_state = "STANCE"
                    step_count += 1
                    step_triggered = True

            if step_triggered:
                if last_step_time > 0.0:
                    step_interval = curr_time - last_step_time
                    if MIN_STEP_INTERVAL <= step_interval <= MAX_STEP_INTERVAL:
                        instant_spm = 60.0 / step_interval
                        cadence_spm = (0.6 * cadence_spm + 0.4 * instant_spm) if cadence_spm > 0 else instant_spm
                last_step_time = curr_time

            # Steady-state cadence fallback
            if step_count >= 2 and cadence_spm == 0.0:
                elapsed_walk = curr_time - start_session_time
                if elapsed_walk > 0:
                    cadence_spm = (step_count / elapsed_walk) * 60.0

            # Dynamic decay when stationary
            if last_step_time > 0.0 and (curr_time - last_step_time) > MAX_STEP_INTERVAL and cadence_spm > 0:
                cadence_spm = max(0.0, cadence_spm - 8.0 * dt)

            # Telemetry logging record
            session_records.append([
                round(curr_time - start_session_time, 2),
                round(l_knee_angle, 1),
                round(r_knee_angle, 1),
                round(knee_si, 1),
                round(l_peak_rom, 1),
                round(r_peak_rom, 1),
                l_leg_state,
                r_leg_state,
                step_count,
                round(cadence_spm, 1)
            ])

        # MediaPipe skeletal overlay
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Render floating targets over patient knees if visible
        if legs_visible:
            if l_knee_px:
                hud.draw_joint_target(frame, l_knee_px, l_knee_angle, label="L-KNEE", side="L")
            if r_knee_px:
                hud.draw_joint_target(frame, r_knee_px, r_knee_angle, label="R-KNEE", side="R")

    # ---------------------------------------------------------
    # 6. Render Biometric Telemetry Glass HUD
    # ---------------------------------------------------------
    hud.render_hud(
        frame=frame,
        patient_name=display_patient_name,
        fps=fps,
        l_knee=l_knee_angle,
        r_knee=r_knee_angle,
        l_state=l_leg_state,
        r_state=r_leg_state,
        step_count=step_count,
        cadence=cadence_spm,
        knee_si=knee_si,
        l_peak_rom=l_peak_rom,
        r_peak_rom=r_peak_rom,
        legs_visible=legs_visible
    )

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------------------------------------------------
# 7. CSV Export Protocol
# ---------------------------------------------------------
if session_records:
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"gait_{patient_name}_{date_str}.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Time_Sec", "L_Knee_Angle", "R_Knee_Angle", "Knee_SI_Pct", 
            "L_Peak_ROM", "R_Peak_ROM", "L_State", "R_State", "Steps", "Cadence_SPM"
        ])
        writer.writerows(session_records)
    print(f"\n[SUCCESS] Diagnostic telemetry saved to: {csv_filename}")