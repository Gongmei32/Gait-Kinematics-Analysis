import sys
import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def get_latest_session_csv():
    """Finds the most recent session CSV in the root directory."""
    files = glob.glob("gait_*.csv")
    if not files:
        return None
    return max(files, key=os.path.getctime)

def extract_patient_name(filepath):
    """
    Extracts the patient name from files named:
    - gait_<Name>_<Date>.csv
    - gait_<Name>_<Date>_<Time>.csv
    - gait_<Name>.csv
    """
    base = os.path.basename(filepath)
    # Strip extension
    stem = os.path.splitext(base)[0]
    
    parts = stem.split("_")
    # Expected structure: parts[0] is 'gait', parts[1] is name, trailing parts are date/time
    if len(parts) >= 2 and parts[0] == "gait":
        # If the last segment is purely numeric (date or timestamp), drop it
        name_parts = parts[1:]
        while name_parts and name_parts[-1].isdigit():
            name_parts.pop()
        
        if name_parts:
            return " ".join(name_parts)
            
    return "Anonymous"

def plot_gait_telemetry(csv_file):
    patient_title = extract_patient_name(csv_file)
    print(f"[INFO] Parsing session for: {patient_title} from {csv_file}")
    
    df = pd.read_csv(csv_file)
    if df.empty or "Time_Sec" not in df.columns:
        print("[ERROR] CSV file is empty or corrupted.")
        return

    time_pts = df["Time_Sec"]
    l_knee = df["L_Knee_Angle"]
    r_knee = df["R_Knee_Angle"]
    si_pct = df["Knee_SI_Pct"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    fig.patch.set_facecolor('#121212')

    for ax in (ax1, ax2):
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white')
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    # Subplot 1: Bilateral Knee Flexion Waves
    ax1.plot(time_pts, l_knee, label='Left Knee Angle', color='#00e5ff', linewidth=2)
    ax1.plot(time_pts, r_knee, label='Right Knee Angle', color='#ff4081', linewidth=2)
    ax1.set_ylabel('Angle (Degrees)', color='white', fontsize=11)
    ax1.set_title(f'Gait Kinematics - Patient: {patient_title} (Sagittal Knee Flexion)', color='white', fontsize=13, pad=12)
    ax1.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='none', labelcolor='white')

    # Subplot 2: Dynamic Symmetry Index (%)
    ax2.plot(time_pts, si_pct, label='Symmetry Index (SI %)', color='#76ff03', linewidth=1.8)
    ax2.axhline(10.0, color='#ffeb3b', linestyle=':', linewidth=1.5, label='Mild Deviation Threshold (10%)')
    ax2.axhline(20.0, color='#f44336', linestyle='--', linewidth=1.5, label='Pathological Asymmetry (20%)')
    ax2.set_xlabel('Elapsed Time (Seconds)', color='white', fontsize=11)
    ax2.set_ylabel('Symmetry Index (%)', color='white', fontsize=11)
    ax2.set_ylim(0, max(35.0, si_pct.max() + 5))
    ax2.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='none', labelcolor='white')

    # Clinical Summary Box
    avg_si = si_pct.mean()
    total_steps = df["Steps"].iloc[-1] if "Steps" in df.columns else 0
    avg_cadence = df["Cadence_SPM"].iloc[-1] if "Cadence_SPM" in df.columns else 0.0

    summary_text = (
        f"Subject: {patient_title}\n"
        f"Duration: {time_pts.iloc[-1]:.1f}s | Steps: {total_steps}\n"
        f"Cadence: {avg_cadence:.1f} SPM | Mean SI: {avg_si:.1f}%"
    )
    plt.gcf().text(0.14, 0.02, summary_text, fontsize=9.5, color='#e0e0e0',
                   bbox=dict(facecolor='#2a2a2a', edgecolor='#00e5ff', boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else get_latest_session_csv()
    if not csv_path:
        print("[WARNING] No gait_*.csv found. Run app.py first.")
    else:
        plot_gait_telemetry(csv_path)