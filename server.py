import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import time
import os
import csv
import glob
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import mediapipe.python.solutions.pose as mp_pose

app = Flask(__name__)

# ---------------------------------------------------------
# Biomechanical State & Math Engines
# ---------------------------------------------------------
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.35,
    min_tracking_confidence=0.35
)

def compute_joint_angle(a, b, c):
    """Computes planar interior joint angle at vertex b using vector dot product."""
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

# Global Telemetry Packet
telemetry_state = {
    "patient_name": "Subject_01",
    "fps": 0.0,
    "l_knee_angle": 0.0,
    "r_knee_angle": 0.0,
    "knee_si": 0.0,
    "l_peak_rom": 0.0,
    "r_peak_rom": 0.0,
    "l_state": "STANCE",
    "r_state": "STANCE",
    "step_count": 0,
    "cadence_spm": 0.0,
    "legs_visible": False,
    "is_recording": False
}

active_session_log = []
start_session_time = time.time()
last_step_time = 0.0
MIN_STEP_INTERVAL = 0.35
MAX_STEP_INTERVAL = 3.5
VISIBILITY_THRESHOLD = 0.35

# Holographic Palette (BGR)
NEON_CYAN = (255, 230, 0)
ALERT_AMBER = (0, 179, 255)
CYAN_MUTED = (180, 150, 40)
TEXT_WHITE = (245, 245, 245)

def render_holographic_limbs(frame, l_hip_px, l_knee_px, l_ankle_px,
                             r_hip_px, r_knee_px, r_ankle_px,
                             l_state, r_state):
    l_col = ALERT_AMBER if l_state == "SWING" else NEON_CYAN
    r_col = ALERT_AMBER if r_state == "SWING" else NEON_CYAN

    # Pelvic link
    cv2.line(frame, l_hip_px, r_hip_px, CYAN_MUTED, 2, cv2.LINE_AA)

    # Left leg (outer glow + inner core)
    cv2.line(frame, l_hip_px, l_knee_px, (l_col[0]//2, l_col[1]//2, l_col[2]//2), 5, cv2.LINE_AA)
    cv2.line(frame, l_knee_px, l_ankle_px, (l_col[0]//2, l_col[1]//2, l_col[2]//2), 5, cv2.LINE_AA)
    cv2.line(frame, l_hip_px, l_knee_px, l_col, 2, cv2.LINE_AA)
    cv2.line(frame, l_knee_px, l_ankle_px, l_col, 2, cv2.LINE_AA)

    # Right leg (outer glow + inner core)
    cv2.line(frame, r_hip_px, r_knee_px, (r_col[0]//2, r_col[1]//2, r_col[2]//2), 5, cv2.LINE_AA)
    cv2.line(frame, r_knee_px, r_ankle_px, (r_col[0]//2, r_col[1]//2, r_col[2]//2), 5, cv2.LINE_AA)
    cv2.line(frame, r_hip_px, r_knee_px, r_col, 2, cv2.LINE_AA)
    cv2.line(frame, r_knee_px, r_ankle_px, r_col, 2, cv2.LINE_AA)

    # Ankle contact rings
    for ankle_pt, state, col in [(l_ankle_px, l_state, l_col), (r_ankle_px, r_state, r_col)]:
        ax, ay = ankle_pt
        if state == "STANCE":
            cv2.ellipse(frame, (ax, ay + 6), (14, 4), 0, 0, 360, col, 1, cv2.LINE_AA)
            cv2.circle(frame, (ax, ay), 4, TEXT_WHITE, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (ax, ay), 5, col, 1, cv2.LINE_AA)

def auto_detect_camera():
    """Scans auxiliary device indices (1, 2, 3) for phone cams, then falls back to 0."""
    for idx in [0]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[INFO] Primary optical link established on Phone Camera (Index {idx})")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap
            cap.release()
    print("[WARN] External camera not found on indices 1-3. Falling back to webcam (Index 0)...")
    return cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera = auto_detect_camera()

def generate_frames():
    global last_step_time, start_session_time, camera
    prev_time = time.time()
    fps = 0.0

    while True:
        if not camera.isOpened():
            blank = np.zeros((540, 960, 3), dtype=np.uint8)
            cv2.putText(blank, "OPTICAL FEED OFFLINE // CHECK CAMERA", (200, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            time.sleep(0.01)
            continue

        frame = cv2.resize(frame, (960, 540))
        fh, fw, _ = frame.shape

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else 0.0

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True

        legs_visible = False
        l_knee_angle, r_knee_angle, knee_si = 0.0, 0.0, 0.0

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            l_hip_lm, l_knee_lm, l_ankle_lm = lm[23], lm[25], lm[27]
            r_hip_lm, r_knee_lm, r_ankle_lm = lm[24], lm[26], lm[28]

            l_k_vis = l_knee_lm.visibility
            r_k_vis = r_knee_lm.visibility

            if (l_knee_lm.visibility > VISIBILITY_THRESHOLD and l_ankle_lm.visibility > VISIBILITY_THRESHOLD and
                r_knee_lm.visibility > VISIBILITY_THRESHOLD and r_ankle_lm.visibility > VISIBILITY_THRESHOLD):

                legs_visible = True
                l_hip, l_knee, l_ankle = [l_hip_lm.x, l_hip_lm.y], [l_knee_lm.x, l_knee_lm.y], [l_ankle_lm.x, l_ankle_lm.y]
                r_hip, r_knee, r_ankle = [r_hip_lm.x, r_hip_lm.y], [r_knee_lm.x, r_knee_lm.y], [r_ankle_lm.x, r_ankle_lm.y]

                l_hip_px   = (int(l_hip[0] * fw), int(l_hip[1] * fh))
                l_knee_px  = (int(l_knee[0] * fw), int(l_knee[1] * fh))
                l_ankle_px = (int(l_ankle[0] * fw), int(l_ankle_lm.y * fh))

                r_hip_px   = (int(r_hip[0] * fw), int(r_hip[1] * fh))
                r_knee_px  = (int(r_knee[0] * fw), int(r_knee[1] * fh))
                r_ankle_px = (int(r_ankle[0] * fw), int(r_ankle_lm.y * fh))

                l_knee_angle = compute_joint_angle(l_hip, l_knee, l_ankle)
                r_knee_angle = compute_joint_angle(r_hip, r_knee, r_ankle)
                knee_si      = compute_symmetry_index(l_knee_angle, r_knee_angle)

                telemetry_state["l_peak_rom"] = max(telemetry_state["l_peak_rom"], 180.0 - l_knee_angle)
                telemetry_state["r_peak_rom"] = max(telemetry_state["r_peak_rom"], 180.0 - r_knee_angle)

                step_triggered = False
                if l_knee_angle < 135.0 and telemetry_state["l_state"] == "STANCE":
                    telemetry_state["l_state"] = "SWING"
                elif l_knee_angle >= 160.0 and telemetry_state["l_state"] == "SWING":
                    if (curr_time - last_step_time) >= MIN_STEP_INTERVAL:
                        telemetry_state["l_state"] = "STANCE"
                        telemetry_state["step_count"] += 1
                        step_triggered = True

                if r_knee_angle < 135.0 and telemetry_state["r_state"] == "STANCE":
                    telemetry_state["r_state"] = "SWING"
                elif r_knee_angle >= 160.0 and telemetry_state["r_state"] == "SWING":
                    if (curr_time - last_step_time) >= MIN_STEP_INTERVAL:
                        telemetry_state["r_state"] = "STANCE"
                        telemetry_state["step_count"] += 1
                        step_triggered = True

                if step_triggered:
                    if last_step_time > 0.0:
                        interval = curr_time - last_step_time
                        if MIN_STEP_INTERVAL <= interval <= MAX_STEP_INTERVAL:
                            spm = 60.0 / interval
                            telemetry_state["cadence_spm"] = (0.6 * telemetry_state["cadence_spm"] + 0.4 * spm) if telemetry_state["cadence_spm"] > 0 else spm
                    last_step_time = curr_time

                render_holographic_limbs(frame, l_hip_px, l_knee_px, l_ankle_px,
                                         r_hip_px, r_knee_px, r_ankle_px,
                                         telemetry_state["l_state"], telemetry_state["r_state"])

                if telemetry_state["is_recording"]:
                    active_session_log.append({
                        "Time_Sec": round(curr_time - start_session_time, 2),
                        "L_Knee_Angle": round(l_knee_angle, 1),
                        "R_Knee_Angle": round(r_knee_angle, 1),
                        "Knee_SI_Pct": round(knee_si, 1),
                        "L_Peak_ROM": round(telemetry_state["l_peak_rom"], 1),
                        "R_Peak_ROM": round(telemetry_state["r_peak_rom"], 1),
                        "L_State": telemetry_state["l_state"],
                        "R_State": telemetry_state["r_state"],
                        "Steps": telemetry_state["step_count"],
                        "Cadence_SPM": round(telemetry_state["cadence_spm"], 1)
                    })
            else:
                cv2.putText(frame, f"BODY IN FRAME // CONFIDENCE: L {l_k_vis:.2f} | R {r_k_vis:.2f}",
                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "STAND BACK // STEP INTO CAMERA FRAME",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        telemetry_state.update({
            "fps": round(fps, 1),
            "l_knee_angle": round(l_knee_angle, 1),
            "r_knee_angle": round(r_knee_angle, 1),
            "knee_si": round(knee_si, 1),
            "legs_visible": legs_visible
        })

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ---------------------------------------------------------
# Web Routes
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def get_telemetry():
    return jsonify(telemetry_state)

@app.route('/api/session/start', methods=['POST'])
def start_session():
    global active_session_log, start_session_time
    data = request.get_json() or {}
    telemetry_state["patient_name"] = data.get("patient_name", "Subject_01")
    telemetry_state["step_count"] = 0
    telemetry_state["l_peak_rom"] = 0.0
    telemetry_state["r_peak_rom"] = 0.0
    telemetry_state["is_recording"] = True
    start_session_time = time.time()
    active_session_log = []
    return jsonify({"status": "recording_started", "patient": telemetry_state["patient_name"]})

@app.route('/api/session/stop', methods=['POST'])
def stop_session():
    global active_session_log
    telemetry_state["is_recording"] = False
    
    csv_filename = None
    if active_session_log:
        p_clean = telemetry_state["patient_name"].replace(" ", "_")
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"gait_{p_clean}_{date_str}.csv"
        keys = active_session_log[0].keys()
        with open(csv_filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(active_session_log)
        print(f"[SUCCESS] Telemetry CSV logged: {csv_filename}")

    return jsonify({
        "status": "recording_stopped",
        "csv_file": csv_filename,
        "total_records": len(active_session_log),
        "data": active_session_log
    })

@app.route('/api/session/latest')
def get_latest_session():
    return jsonify({
        "patient_name": telemetry_state["patient_name"],
        "records": active_session_log
    })

@app.route('/api/sessions/list')
def list_saved_sessions():
    """Lists all saved gait_*.csv sessions on disk sorted by most recent."""
    files = glob.glob("gait_*.csv")
    files.sort(key=os.path.getctime, reverse=True)
    return jsonify({"sessions": [os.path.basename(f) for f in files]})

@app.route('/api/sessions/load/<filename>')
def load_specific_session(filename):
    """Parses a selected CSV file and returns the records."""
    clean_name = os.path.basename(filename)
    if not os.path.exists(clean_name):
        return jsonify({"error": "File not found"}), 404

    records = []
    with open(clean_name, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "Time_Sec": float(row.get("Time_Sec", 0.0)),
                "L_Knee_Angle": float(row.get("L_Knee_Angle", 0.0)),
                "R_Knee_Angle": float(row.get("R_Knee_Angle", 0.0)),
                "Knee_SI_Pct": float(row.get("Knee_SI_Pct", 0.0)),
                "L_Peak_ROM": float(row.get("L_Peak_ROM", 0.0)),
                "R_Peak_ROM": float(row.get("R_Peak_ROM", 0.0)),
                "L_State": row.get("L_State", "STANCE"),
                "R_State": row.get("R_State", "STANCE"),
                "Steps": int(row.get("Steps", 0)),
                "Cadence_SPM": float(row.get("Cadence_SPM", 0.0))
            })

    parts = os.path.splitext(clean_name)[0].split("_")
    name_parts = parts[1:]
    while name_parts and name_parts[-1].isdigit():
        name_parts.pop()
    patient_name = " ".join(name_parts) if name_parts else "Anonymous"

    return jsonify({
        "patient_name": patient_name,
        "filename": clean_name,
        "records": records
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)