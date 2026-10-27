let isRecording = false;

// 1. Live Knee Flexion Waveform Chart
const ctx = document.getElementById('liveChart').getContext('2d');
const maxPoints = 40;
const liveChart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: Array(maxPoints).fill(''),
    datasets: [
      {
        label: 'Left Knee (°)',
        data: Array(maxPoints).fill(180),
        borderColor: '#00e5ff',
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0
      },
      {
        label: 'Right Knee (°)',
        data: Array(maxPoints).fill(180),
        borderColor: '#ffb300',
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 0
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    scales: {
      y: { min: 80, max: 185, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#64748b' } },
      x: { grid: { display: false } }
    },
    plugins: {
      legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono' } } }
    }
  }
});

// 2. Start / Stop Session Recording
async function toggleSession() {
  const patient = document.getElementById('patient-input').value.trim() || 'Subject_01';
  const recBtn = document.getElementById('btn-record');
  const recBadge = document.getElementById('rec-status');

  if (!isRecording) {
    const res = await fetch('/api/session/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_name: patient })
    });
    if (res.ok) {
      isRecording = true;
      recBtn.innerText = 'STOP RECORDING & LOG';
      recBtn.classList.add('btn-stop');
      recBadge.innerText = 'LOGGING TELEMETRY';
      recBadge.style.color = '#ff3366';
    }
  } else {
    const res = await fetch('/api/session/stop', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      isRecording = false;
      recBtn.innerText = 'START RECORDING';
      recBtn.classList.remove('btn-stop');
      recBadge.innerText = data.csv_file ? `LOGGED: ${data.csv_file}` : 'SESSION COMPLETE';
      recBadge.style.color = '#00e5ff';
    }
  }
}

// 3. Telemetry Sync Loop (30 Hz)
setInterval(async () => {
  try {
    const res = await fetch('/api/telemetry');
    const data = await res.json();

    document.getElementById('si-val').innerText = `${data.knee_si}%`;
    document.getElementById('cadence-val').innerHTML = `${data.cadence_spm} <span class="unit">SPM</span>`;
    document.getElementById('steps-val').innerText = data.step_count;
    document.getElementById('l-rom').innerText = `${data.l_peak_rom}°`;
    document.getElementById('r-rom').innerText = `${data.r_peak_rom}°`;
    document.getElementById('l-phase').innerText = `L: ${data.l_state}`;
    document.getElementById('r-phase').innerText = `R: ${data.r_state}`;

    const siEl = document.getElementById('si-status');
    if (data.knee_si <= 8.0) {
      siEl.innerText = "NOMINAL"; siEl.style.color = "#00e5ff";
    } else if (data.knee_si <= 15.0) {
      siEl.innerText = "DRIFT DETECTED"; siEl.style.color = "#ffb300";
    } else {
      siEl.innerText = "ASYMMETRIC BIAS"; siEl.style.color = "#ff3366";
    }

    if (data.legs_visible) {
      liveChart.data.datasets[0].data.push(data.l_knee_angle);
      liveChart.data.datasets[0].data.shift();
      liveChart.data.datasets[1].data.push(data.r_knee_angle);
      liveChart.data.datasets[1].data.shift();
      liveChart.update();
    }
  } catch (err) {
    console.error("Telemetry sync error:", err);
  }
}, 33);