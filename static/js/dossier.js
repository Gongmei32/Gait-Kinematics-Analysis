let flexionChart = null;
let siChart = null;
let cadenceChart = null;

// -------------------------------------------------------------
// Historical Log Selection & Loading
// -------------------------------------------------------------
async function populateSessionDropdown() {
  try {
    const res = await fetch('/api/sessions/list');
    const data = await res.json();
    const select = document.getElementById('session-select');
    select.innerHTML = '<option value="">-- SELECT PREVIOUS LOG --</option>';

    if (data.sessions && data.sessions.length > 0) {
      data.sessions.forEach(file => {
        const opt = document.createElement('option');
        opt.value = file;
        opt.innerText = file;
        select.appendChild(opt);
      });
    }
  } catch (err) {
    console.error("Failed to fetch session list:", err);
  }
}

async function loadSelectedSession(filename) {
  if (!filename) return;

  try {
    const res = await fetch(`/api/sessions/load/${encodeURIComponent(filename)}`);
    const result = await res.json();
    processAndDisplaySession(result);
  } catch (err) {
    console.error(`Failed to load file ${filename}:`, err);
  }
}

async function loadLatestSession() {
  await populateSessionDropdown();
  try {
    const res = await fetch('/api/session/latest');
    const result = await res.json();

    // If no in-memory log exists from current run, auto-select most recent CSV file
    if (!result.records || result.records.length === 0) {
      const select = document.getElementById('session-select');
      if (select.options.length > 1) {
        select.selectedIndex = 1;
        await loadSelectedSession(select.value);
        return;
      }
    }
    processAndDisplaySession(result);
  } catch (err) {
    console.error("Dossier render failure:", err);
  }
}

function processAndDisplaySession(result) {
  const records = result.records || [];
  const patient = result.patient_name || 'Subject_01';

  document.getElementById('dossier-patient-tag').innerText = `SUBJECT DOSSIER: ${patient.toUpperCase()}`;

  if (!records || records.length === 0) {
    document.getElementById('dossier-meta-tag').innerText = "STATUS: NO RECORDS FOUND IN LOG.";
    return;
  }

  document.getElementById('dossier-meta-tag').innerText = `STATUS: LOADED ${records.length} TELEMETRY FRAMES ${result.filename ? '(' + result.filename + ')' : ''}`;

  const timeLabels = records.map(r => `${r.Time_Sec}s`);
  const lKnee = records.map(r => r.L_Knee_Angle);
  const rKnee = records.map(r => r.R_Knee_Angle);
  const si = records.map(r => r.Knee_SI_Pct);
  const cadence = records.map(r => r.Cadence_SPM);

  const dur = records[records.length - 1].Time_Sec;
  const totalSteps = records[records.length - 1].Steps;
  const meanCadence = (cadence.reduce((a, b) => a + b, 0) / cadence.length).toFixed(1);
  const meanSi = (si.reduce((a, b) => a + b, 0) / si.length).toFixed(1);
  const maxLrom = Math.max(...records.map(r => r.L_Peak_ROM)).toFixed(1);
  const maxRrom = Math.max(...records.map(r => r.R_Peak_ROM)).toFixed(1);

  document.getElementById('sum-dur').innerText = `${dur}s`;
  document.getElementById('sum-steps').innerText = totalSteps;
  document.getElementById('sum-cad').innerText = `${meanCadence} SPM`;
  document.getElementById('sum-si').innerText = `${meanSi}%`;
  document.getElementById('sum-lrom').innerText = `${maxLrom}°`;
  document.getElementById('sum-rrom').innerText = `${maxRrom}°`;

  evaluateGaitPathology(records, parseFloat(meanSi), parseFloat(maxLrom), parseFloat(maxRrom), parseFloat(meanCadence));
  renderDossierCharts(timeLabels, lKnee, rKnee, si, cadence);
}

// -------------------------------------------------------------
// Biomechanical Diagnostic Rule Engine
// -------------------------------------------------------------
function evaluateGaitPathology(records, meanSi, maxLrom, maxRrom, meanCadence) {
  const badgeEl = document.getElementById('diag-badge');
  const listEl = document.getElementById('findings-list');
  const impEl = document.getElementById('diag-impression');

  listEl.innerHTML = '';
  const findings = [];
  const suspectedConditions = [];
  let severity = "NOMINAL";

  // 1. Bilateral Flexion ROM Analysis
  const romDiff = Math.abs(maxLrom - maxRrom);
  if (maxLrom < 45.0 || maxRrom < 45.0) {
    const affected = maxLrom < maxRrom ? "Left" : "Right";
    findings.push(`<b>Deficient Swing Flexion (${affected} Limb):</b> Peak flexion failed to clear 45° (L: ${maxLrom}°, R: ${maxRrom}°). Indicates stiff-knee gait, reduced foot-ground clearance, and heightened trip risk.`);
    suspectedConditions.push("Stiff-Knee Gait / Quadriceps Spasticity / Joint Contracture");
    severity = "ALERT";
  }

  // 2. ROM Discrepancy & Guarding
  if (romDiff > 12.0) {
    const favored = maxLrom > maxRrom ? "Left" : "Right";
    const guarded = maxLrom > maxRrom ? "Right" : "Left";
    findings.push(`<b>Bilateral Disparity:</b> ${romDiff.toFixed(1)}° difference in peak flexion. Subject heavily favors the ${favored} limb while guarding the ${guarded} limb.`);
    suspectedConditions.push(`Unilateral Antalgic (Pain-Avoidant) Gait favoring ${favored} side`);
    severity = "ALERT";
  }

  // 3. Robinson's Symmetry Index (SI) Evaluation
  if (meanSi > 15.0) {
    findings.push(`<b>Pathological Asymmetry (Mean SI: ${meanSi}%):</b> Exceeds the 15% clinical deviation ceiling. Locomotion shows pronounced unilateral compensation.`);
    suspectedConditions.push("Hemiparetic Gait Pattern / Asymmetric Muscular Weakness");
    severity = "CRITICAL";
  } else if (meanSi > 8.0) {
    findings.push(`<b>Mild Symmetry Drift (Mean SI: ${meanSi}%):</b> Exceeds the 8% nominal threshold. Indicates minor limp or uncoordinated inter-limb phase transitions.`);
    if (severity !== "CRITICAL") severity = "WARNING";
  } else {
    findings.push(`<b>Bilateral Symmetry:</b> Mean SI (${meanSi}%) remains within the nominal <8% therapeutic window.`);
  }

  // 4. Cadence & Locomotion Rhythm Evaluation
  if (meanCadence < 70.0) {
    findings.push(`<b>Bradykinetic Locomotion (${meanCadence} SPM):</b> Cadence is significantly depressed compared to standard adult baseline (90–115 SPM).`);
    suspectedConditions.push("Cautious / Hesitant Locomotion / Motor Control Inhibition");
    if (severity === "NOMINAL") severity = "WARNING";
  } else if (meanCadence > 135.0) {
    findings.push(`<b>Accelerated Cadence (${meanCadence} SPM):</b> Unusually high cadence; likely compensative short-striding.`);
  }

  findings.forEach(f => {
    const li = document.createElement('li');
    li.innerHTML = f;
    listEl.appendChild(li);
  });

  if (severity === "CRITICAL") {
    badgeEl.innerText = "CRITICAL ASYMMETRY / PATHOLOGY DETECTED";
    badgeEl.style.color = "#ff3366";
    badgeEl.style.borderColor = "#ff3366";
    impEl.innerText = suspectedConditions.join(" OR ") || "Marked kinematic impairment observed.";
  } else if (severity === "ALERT") {
    badgeEl.innerText = "KINEMATIC DEFICIT DETECTED";
    badgeEl.style.color = "#ffb300";
    badgeEl.style.borderColor = "#ffb300";
    impEl.innerText = suspectedConditions.join(" WITH ") || "Unilateral joint restriction or guarding detected.";
  } else if (severity === "WARNING") {
    badgeEl.innerText = "MILD DEVIATION FROM NOMINAL BASELINE";
    badgeEl.style.color = "#ffb300";
    badgeEl.style.borderColor = "#ffb300";
    impEl.innerText = "Minor inter-limb asymmetry or pacing hesitation. No severe contracture detected.";
  } else {
    badgeEl.innerText = "NOMINAL PHYSIOLOGICAL PROFILE";
    badgeEl.style.color = "#00e5ff";
    badgeEl.style.borderColor = "#00e5ff";
    impEl.innerText = "No significant gait impairment or joint asymmetry detected across the captured duration.";
  }
}

// -------------------------------------------------------------
// Chart Render Function
// -------------------------------------------------------------
function renderDossierCharts(labels, lKnee, rKnee, si, cadence) {
  if (flexionChart) flexionChart.destroy();
  const ctxFlex = document.getElementById('dossierFlexionChart').getContext('2d');
  flexionChart = new Chart(ctxFlex, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'Left Knee Flexion (°)', data: lKnee, borderColor: '#00e5ff', borderWidth: 2, tension: 0.2, pointRadius: 0 },
        { label: 'Right Knee Flexion (°)', data: rKnee, borderColor: '#ffb300', borderWidth: 2, tension: 0.2, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 80, max: 185, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', maxTicksLimit: 12 } }
      },
      plugins: { legend: { labels: { color: '#94a3b8' } } }
    }
  });

  if (siChart) siChart.destroy();
  const ctxSi = document.getElementById('dossierSiChart').getContext('2d');
  siChart = new Chart(ctxSi, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'Symmetry Index (%)', data: si, borderColor: '#00ffcc', borderWidth: 2, tension: 0.2, pointRadius: 0 },
        { label: 'Nominal Threshold (8%)', data: Array(labels.length).fill(8.0), borderColor: '#ffb300', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 },
        { label: 'Asymmetry Boundary (15%)', data: Array(labels.length).fill(15.0), borderColor: '#ff3366', borderDash: [4, 4], borderWidth: 1.5, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 30, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', maxTicksLimit: 12 } }
      },
      plugins: { legend: { labels: { color: '#94a3b8' } } }
    }
  });

  if (cadenceChart) cadenceChart.destroy();
  const ctxCad = document.getElementById('dossierCadenceChart').getContext('2d');
  cadenceChart = new Chart(ctxCad, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'Cadence (SPM)', data: cadence, borderColor: '#9d4edd', borderWidth: 2, tension: 0.2, pointRadius: 0 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 140, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8' } },
        x: { grid: { color: 'rgba(255,255,255,0.03)' }, ticks: { color: '#64748b', maxTicksLimit: 12 } }
      },
      plugins: { legend: { labels: { color: '#94a3b8' } } }
    }
  });
}