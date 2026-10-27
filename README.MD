# AEGIS-GAIT // Biomechanical Gait Kinematics & Clinical Telemetry Suite

![AEGIS-GAIT Version](https://img.shields.io/badge/Version-1.0.0-00e5ff?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?style=for-the-badge&logo=flask)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange?style=for-the-badge)
![Chart.js](https://img.shields.io/badge/Chart.js-4.0%2B-red?style=for-the-badge&logo=chartdotjs)

**AEGIS-GAIT** is a computer-vision biomechanical telemetry and clinical dossier platform designed for non-invasive, markerless gait analysis. Built on MediaPipe Pose and Flask, the system streams real-time sagittal lower-limb kinematics, quantifies dynamic inter-limb asymmetries via Robinson's Symmetry Index (SI), and compiles interactive post-session diagnostic reports.

---

## Key Features

- **Markerless Optical Telemetry (30 Hz):**
  - Live video streaming using lightweight MJPEG conduits.
  - Automatic auxiliary camera auto-discovery prioritizing smartphone webcams (DroidCam, Link to Windows) with seamless laptop webcam fallback.
  - Holographic skeletal overlays rendered dynamically in OpenCV (outer neon glow, inner structural core, stance/swing state coloration, and ground-contact rings).

- **Biomechanical Metrics Engine:**
  - **Planar Knee Flexion/Extension:** Real-time interior joint angle calculations via dot-product vectors:
    $$\theta = \arccos\left(\frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|}\right)$$
  - **Dynamic Peak Range of Motion (ROM):** Dynamic tracking of maximum swing-phase knee flexion.
  - **Robinson's Symmetry Index (SI %):** Quantifies inter-limb asymmetry against standard therapeutic thresholds:
    $$\text{SI} = \frac{|X_{\text{Left}} - X_{\text{Right}}|}{0.5(X_{\text{Left}} + X_{\text{Right}})} \times 100\%$$
  - **Temporal Cadence Profiling:** Continuous stride-frequency tracking measured in Steps Per Minute (SPM).

- **Kinematic Dossier & Diagnostic Engine:**
  - **3-Panel Synchronized Kinematic Charts:** Interactive Chart.js panels displaying bilateral sagittal knee waveforms, SI deviations against 8% (Nominal) and 15% (Asymmetry) boundaries, and cadence profiles.
  - **Automated Pathology Report:** Rule-based clinical classifier that flags stiff-knee gait, antalgic guarding, hemiparetic asymmetry, and bradykinetic cadence.
  - **Historical Log Selector:** Instant drop-down access to parse and visualize historical session CSV files on disk.

---

## System Architecture