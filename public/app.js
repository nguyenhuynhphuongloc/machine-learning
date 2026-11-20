/* IELTS AI Learning Analytics Frontend */

const charts = {};
let sampleCache = null;

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

function badgeClass(type) {
  const map = {
    success: "badge-success",
    info: "badge-info",
    warning: "badge-warning",
    danger: "badge-danger",
  };
  return map[type] || "badge-info";
}

async function apiFetch(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Lỗi kết nối máy chủ API");
  }
  return res.json();
}

// ------------------------------------------------------------------
// Tabs Navigation
// ------------------------------------------------------------------
function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });
}

// ------------------------------------------------------------------
// TAB 1: Predict Actual IELTS Band, Quality, Strength/Weakness
// ------------------------------------------------------------------
function getStudentFormData() {
  const form = document.getElementById("studentForm");
  const fd = new FormData(form);
  const data = {};
  for (const [k, v] of fd.entries()) {
    data[k] = Number(v);
  }
  return data;
}

function fillStudentForm(presetData) {
  const form = document.getElementById("studentForm");
  Object.entries(presetData).forEach(([k, v]) => {
    const el = form.elements[k];
    if (el) el.value = v;
  });
}

function renderStudentResults(res, inputData) {
  const { learning_quality: q, ielts_prediction: p, student_classification: c } = res;

  // Hero Predicted Band Banner
  document.getElementById("kpiPredictedOverall").textContent = p.predicted_overall.toFixed(1);
  const gainPrefix = p.expected_gain >= 0 ? "+" : "";
  document.getElementById("kpiGainInterval").textContent =
    `Đầu vào ${p.entry_score.toFixed(1)} → Tăng trưởng ${gainPrefix}${p.expected_gain.toFixed(1)} band · KTC 90%: ${p.confidence_interval_90.low.toFixed(1)} – ${p.confidence_interval_90.high.toFixed(1)}`;
  document.getElementById("kpiTargetProbLine").textContent =
    `Điểm trung bình thô 4 kỹ năng: ${p.raw_overall_mean.toFixed(2)} · Xác suất đạt Band ${p.target_band.toFixed(1)}: ${p.probability_reach_target}%`;

  document.getElementById("bandL").textContent = p.predicted_skills.listening.toFixed(1);
  document.getElementById("bandR").textContent = p.predicted_skills.reading.toFixed(1);
  document.getElementById("bandW").textContent = p.predicted_skills.writing.toFixed(1);
  document.getElementById("bandS").textContent = p.predicted_skills.speaking.toFixed(1);

  // KPIs
  document.getElementById("kpiQualityScore").textContent = `${q.learning_quality_score}/100`;
  document.getElementById("kpiQualityGrade").innerHTML =
    `Xếp loại: <span class="badge ${badgeClass(q.grade_color)}">${escapeHtml(q.grade)}</span>`;

  document.getElementById("kpiStudentTier").innerHTML =
    `<span class="text-${c.badge}">${escapeHtml(c.predicted_tier)}</span>`;
  document.getElementById("kpiTierConfidence").textContent =
    `Độ tin cậy: ${c.confidence}% · Yếu nhất: ${c.weakest_skill.label_vi.split(" ")[0]} (${c.weakest_skill.hw_pct}%)`;

  document.getElementById("kpiTargetProb").textContent = `${p.probability_reach_target}%`;
  document.getElementById("kpiTargetLabel").textContent = `Mục tiêu Band ${p.target_band.toFixed(1)}`;

  // Chart 1: Entry Score vs Predicted 4 Skills Band
  destroyChart("skillsChart");
  const sCtx = document.getElementById("skillsChart").getContext("2d");
  charts["skillsChart"] = new Chart(sCtx, {
    type: "bar",
    data: {
      labels: ["Listening", "Reading", "Writing", "Speaking"],
      datasets: [
        {
          label: `Đầu vào (${p.entry_score.toFixed(1)})`,
          data: [p.entry_score, p.entry_score, p.entry_score, p.entry_score],
          backgroundColor: "rgba(148, 163, 184, 0.35)",
          borderRadius: 6,
        },
        {
          label: "Band thực tế dự đoán",
          data: [
            p.predicted_skills.listening,
            p.predicted_skills.reading,
            p.predicted_skills.writing,
            p.predicted_skills.speaking,
          ],
          backgroundColor: ["#6366f1", "#38bdf8", "#f59e0b", "#10b981"],
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { min: 0, max: 9, ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.1)" } },
        x: { ticks: { color: "#e2e8f0" }, grid: { display: false } },
      },
      plugins: { legend: { labels: { color: "#cbd5e1" } } },
    },
  });

  // What-If Banner
  const whatIf = p.what_if_simulation;
  const whatIfEl = document.getElementById("whatIfBanner");
  whatIfEl.classList.remove("hidden");
  whatIfEl.innerHTML =
    `<strong>Mô phỏng What-If:</strong> ${escapeHtml(whatIf.description)} → Band thực tế dự kiến đạt ` +
    `<strong>Band ${whatIf.projected_overall.toFixed(1)}</strong>` +
    (whatIf.potential_bonus > 0
      ? ` (bứt phá thêm +${whatIf.potential_bonus.toFixed(1)} band).`
      : ` (đang ở vùng tối ưu).`);

  // Chart 2: Learning Quality Radar (7 dimensions)
  destroyChart("qualityRadarChart");
  const qCtx = document.getElementById("qualityRadarChart").getContext("2d");
  charts["qualityRadarChart"] = new Chart(qCtx, {
    type: "radar",
    data: {
      labels: Object.keys(q.dimensions),
      datasets: [
        {
          label: "Chỉ số học tập (%)",
          data: Object.values(q.dimensions),
          backgroundColor: "rgba(16, 185, 129, 0.22)",
          borderColor: "#10b981",
          pointBackgroundColor: "#34d399",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: { stepSize: 25, color: "#94a3b8", backdropColor: "transparent" },
          grid: { color: "rgba(148,163,184,0.15)" },
          angleLines: { color: "rgba(148,163,184,0.15)" },
          pointLabels: { color: "#e2e8f0", font: { size: 11 } },
        },
      },
      plugins: { legend: { display: false } },
    },
  });

  // Persona Box
  const personaBox = document.getElementById("personaBox");
  personaBox.classList.remove("hidden");
  personaBox.innerHTML =
    `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">` +
    `<strong>Nhóm hành vi (K-Means): ${escapeHtml(q.learning_persona.name)}</strong>` +
    `<span class="badge ${badgeClass(q.learning_persona.badge)}">Cụm #${q.learning_persona.rank + 1}/4</span></div>` +
    `<div style="color:#cbd5e1;">${escapeHtml(q.learning_persona.summary)}</div>`;

  // Tier Probability Bars
  document.getElementById("tierProbsContainer").innerHTML = c.tier_probabilities
    .map(
      (item) => `
      <div class="prob-row">
        <span>${escapeHtml(item.tier)}</span>
        <div class="prob-track"><div class="prob-fill" style="width:${item.probability}%"></div></div>
        <span style="text-align:right;font-weight:600;">${item.probability}%</span>
      </div>`
    )
    .join("");

  // Skill Diagnosis
  document.getElementById("skillDiagnosisList").innerHTML =
    c.skill_diagnosis
      .map(
        (d) => `
      <div class="diag-item">
        <span><strong>${escapeHtml(d.label_vi)}</strong> · BTVN ${d.hw_pct}% → Band ${p.predicted_skills[d.skill].toFixed(1)}</span>
        <span class="badge ${badgeClass(d.badge)}">${escapeHtml(d.status)}</span>
      </div>`
      )
      .join("") +
    `<div class="insight-item advice" style="margin-top:6px;">
      <strong>Chiến lược nhóm ${escapeHtml(c.predicted_tier)}:</strong> ${escapeHtml(c.action_plan)}
    </div>`;

  // Recommendations
  const blocks = [];
  q.strengths.forEach((s) =>
    blocks.push(`<div class="insight-item strength"><strong>Điểm mạnh:</strong> ${escapeHtml(s)}</div>`)
  );
  q.risks.forEach((r) =>
    blocks.push(`<div class="insight-item risk"><strong>Cảnh báo:</strong> ${escapeHtml(r)}</div>`)
  );
  q.recommendations.forEach((rec) =>
    blocks.push(`<div class="insight-item advice"><strong>Khuyến nghị:</strong> ${escapeHtml(rec)}</div>`)
  );
  document.getElementById("recommendationsContainer").innerHTML = blocks.join("");
}

async function runStudentAnalysis() {
  const btn = document.getElementById("analyzeStudentBtn");
  const inputData = getStudentFormData();
  btn.disabled = true;
  btn.textContent = "Đang dự đoán Band thực tế...";
  try {
    const res = await apiFetch("/api/full-assessment", {
      method: "POST",
      body: JSON.stringify(inputData),
    });
    renderStudentResults(res, inputData);
  } catch (err) {
    alert("Lỗi dự đoán: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Dự đoán Band IELTS thực tế";
  }
}

// ------------------------------------------------------------------
// TAB 2: Common Error Classification (NLP)
// ------------------------------------------------------------------
function renderErrorResults(res) {
  document.getElementById("errorCountChip").textContent = `${res.total_segments} câu phân tích`;

  destroyChart("errorPieChart");
  const palette = ["#6366f1", "#ef4444", "#f59e0b", "#10b981", "#38bdf8", "#ec4899", "#a855f7", "#14b8a6"];
  charts["errorPieChart"] = new Chart(document.getElementById("errorPieChart").getContext("2d"), {
    type: "doughnut",
    data: {
      labels: res.category_distribution.map((d) => d.label_vi),
      datasets: [
        {
          data: res.category_distribution.map((d) => d.count),
          backgroundColor: palette.slice(0, res.category_distribution.length),
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "right", labels: { color: "#cbd5e1", font: { size: 11 } } },
      },
    },
  });

  document.getElementById("errorRemediationList").innerHTML = res.remediation_plan
    .map(
      (item) => `
      <div class="insight-item risk">
        <div><strong>Ưu tiên #${item.priority}: ${escapeHtml(item.error_type)}</strong> · <span class="badge badge-warning">${escapeHtml(item.share)}</span></div>
        <div style="margin-top:4px;color:#cbd5e1;">${escapeHtml(item.action)}</div>
        <div class="segment-fix">${escapeHtml(item.example)}</div>
      </div>`
    )
    .join("");

  document.getElementById("errorSegmentsList").innerHTML = res.segment_results
    .map(
      (s, idx) => `
      <div class="segment-card">
        <div class="segment-top">
          <span class="badge badge-danger">#${idx + 1} · ${escapeHtml(s.label_vi)}</span>
          <span class="chip">${escapeHtml(s.skill)} · Độ tin cậy ${s.confidence}%</span>
        </div>
        <div class="segment-quote">"${escapeHtml(s.text)}"</div>
        <div class="segment-meta">${escapeHtml(s.description_vi)}</div>
        ${
          s.detected_signals.length
            ? `<div class="segment-meta" style="color:#fbbf24;margin-top:4px;">Dấu hiệu phát hiện: ${escapeHtml(s.detected_signals.join("; "))}</div>`
            : ""
        }
        <div class="segment-fix">${escapeHtml(s.example_fix)}</div>
      </div>`
    )
    .join("");
}

async function runErrorClassification() {
  const text = document.getElementById("errorInputText").value.trim();
  if (!text) {
    alert("Vui lòng nhập nội dung câu hoặc nhấn 'Nạp bài mẫu 8 loại lỗi'.");
    return;
  }
  const btn = document.getElementById("analyzeErrorBtn");
  btn.disabled = true;
  btn.textContent = "Đang phân loại lỗi...";
  try {
    const res = await apiFetch("/api/classify-errors", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    renderErrorResults(res);
  } catch (err) {
    alert("Lỗi phân loại NLP: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Phân loại lỗi sai bằng NLP";
  }
}

// ------------------------------------------------------------------
// TAB 3: Batch Class Analysis
// ------------------------------------------------------------------
function renderBatchResults(res) {
  document.getElementById("batchResults").classList.remove("hidden");
  const s = res.summary;
  document.getElementById("batchTotal").textContent = res.total_students;
  document.getElementById("batchAvgIelts").textContent = s.avg_predicted_ielts.toFixed(2);
  document.getElementById("batchAvgQuality").textContent = `Chất lượng học TB: ${s.avg_learning_quality}/100`;
  document.getElementById("batchAtRisk").textContent = `${s.at_risk_count} HS`;
  document.getElementById("batchStrongCount").textContent = `Nhóm Mạnh: ${s.strong_count} HS`;
  document.getElementById("batchWeakestSkill").textContent = s.most_common_weak_skill;

  destroyChart("batchTierChart");
  charts["batchTierChart"] = new Chart(document.getElementById("batchTierChart").getContext("2d"), {
    type: "bar",
    data: {
      labels: Object.keys(s.tier_distribution),
      datasets: [
        {
          label: "Số lượng học sinh",
          data: Object.values(s.tier_distribution),
          backgroundColor: ["#ef4444", "#f59e0b", "#38bdf8", "#10b981"],
          borderRadius: 7,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, ticks: { stepSize: 1, color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.1)" } },
        x: { ticks: { color: "#e2e8f0" }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  destroyChart("batchSkillChart");
  charts["batchSkillChart"] = new Chart(document.getElementById("batchSkillChart").getContext("2d"), {
    type: "doughnut",
    data: {
      labels: Object.keys(s.weak_skill_distribution),
      datasets: [
        {
          data: Object.values(s.weak_skill_distribution),
          backgroundColor: ["#6366f1", "#38bdf8", "#ef4444", "#f59e0b"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "right", labels: { color: "#cbd5e1" } } },
    },
  });

  document.getElementById("batchTableBody").innerHTML = res.students
    .map(
      (st) => `
      <tr>
        <td><strong>${escapeHtml(st.name)}</strong></td>
        <td><span class="badge ${badgeClass(st.badge)}">${escapeHtml(st.tier)}</span></td>
        <td>${st.entry_score.toFixed(1)}</td>
        <td>${st.wow_sessions} buổi</td>
        <td>${st.attendance_rate}% / ${st.homework_completion_rate}%</td>
        <td>${st.learning_quality_score}/100</td>
        <td><strong>${st.predicted_overall.toFixed(1)}</strong> (${st.expected_gain >= 0 ? "+" : ""}${st.expected_gain.toFixed(1)})</td>
        <td>${escapeHtml(st.weakest_skill)}</td>
        <td>${st.probability_reach_target}%</td>
      </tr>`
    )
    .join("");
}

async function runBatchAnalysis() {
  const raw = document.getElementById("batchJsonInput").value.trim();
  const targetBand = Number(document.getElementById("batchTargetBand").value) || 6.5;
  let students;
  try {
    students = JSON.parse(raw);
    if (!Array.isArray(students)) throw new Error("Dữ liệu phải là mảng JSON các học sinh.");
  } catch (e) {
    alert("JSON không hợp lệ: " + e.message);
    return;
  }

  const btn = document.getElementById("runBatchBtn");
  btn.disabled = true;
  btn.textContent = "Đang phân tích lớp...";
  try {
    const res = await apiFetch("/api/batch-analyze", {
      method: "POST",
      body: JSON.stringify({ target_band: targetBand, students }),
    });
    renderBatchResults(res);
  } catch (err) {
    alert("Lỗi phân tích lớp: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Phân tích cả lớp";
  }
}

// ------------------------------------------------------------------
// TAB 4: Model Metrics & Algorithm Benchmarks
// ------------------------------------------------------------------
async function loadModelMetrics() {
  const m = await apiFetch("/api/model-metrics");
  document.getElementById("mIeltsR2").textContent = m.ielts_prediction.overall_r2.toFixed(3);
  document.getElementById("mIeltsMae").textContent =
    `MAE: ${m.ielts_prediction.overall_mae} band · Sai số ≤ 0.5 band: ${m.ielts_prediction.within_half_band_accuracy}%`;

  document.getElementById("mQualityR2").textContent = m.learning_quality.r2.toFixed(3);
  document.getElementById("mQualitySil").textContent =
    `MAE: ${m.learning_quality.mae}đ · K-Means Silhouette: ${m.learning_quality.silhouette_score}`;

  document.getElementById("mTierAcc").textContent = `${m.student_classification.accuracy}%`;
  document.getElementById("mTierF1").textContent = `Macro-F1: ${m.student_classification.f1_macro}%`;

  document.getElementById("mErrorAcc").textContent = `${m.error_classification.cv_accuracy}%`;
  document.getElementById("mErrorF1").textContent =
    `Macro-F1: ${m.error_classification.cv_f1_macro}% (${m.error_classification.num_categories} nhóm lỗi)`;

  destroyChart("benchmarkChart");
  const bench = m.ielts_prediction.algorithm_benchmark;
  charts["benchmarkChart"] = new Chart(document.getElementById("benchmarkChart").getContext("2d"), {
    type: "bar",
    data: {
      labels: bench.map((b) => b.name),
      datasets: [
        {
          label: "R² Score (cao hơn = tốt hơn)",
          data: bench.map((b) => b.r2),
          backgroundColor: "rgba(99, 102, 241, 0.85)",
          borderRadius: 6,
        },
        {
          label: "MAE (thấp hơn = tốt hơn)",
          data: bench.map((b) => b.mae),
          backgroundColor: "rgba(245, 158, 11, 0.85)",
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 1, ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.1)" } },
        x: { ticks: { color: "#e2e8f0" }, grid: { display: false } },
      },
      plugins: { legend: { labels: { color: "#cbd5e1" } } },
    },
  });

  destroyChart("featureImportanceChart");
  const fi = m.feature_importance.ielts_prediction;
  charts["featureImportanceChart"] = new Chart(
    document.getElementById("featureImportanceChart").getContext("2d"),
    {
      type: "bar",
      data: {
        labels: fi.map((f) => f.label_vi),
        datasets: [
          {
            label: "Tầm quan trọng (%)",
            data: fi.map((f) => f.importance),
            backgroundColor: "rgba(16, 185, 129, 0.82)",
            borderRadius: 6,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#94a3b8" }, grid: { color: "rgba(148,163,184,0.1)" } },
          y: { ticks: { color: "#e2e8f0", font: { size: 11 } }, grid: { display: false } },
        },
        plugins: { legend: { display: false } },
      },
    }
  );

  const cards = [
    {
      title: "1. Dự đoán Band IELTS thực tế",
      desc: `Thuật toán: ${m.ielts_prediction.selected_algorithm}. Từ 8 trường đầu vào (Điểm đầu vào, Số buổi WOW, % chuyên cần, % làm BTVN, % điểm 4 kỹ năng BTVN), dự đoán riêng 4 kỹ năng và Overall làm tròn chuẩn 0.5 IELTS (R²=${m.ielts_prediction.overall_r2}).`,
    },
    {
      title: "2. Đánh giá Chất lượng học tập",
      desc: `Thuật toán: ${m.learning_quality.algorithm}. Chấm điểm chất lượng học 0–100 (R²=${m.learning_quality.r2}) kết hợp phân cụm K-Means (4 nhóm hành vi học tập) giúp cảnh báo sớm học sinh thiếu chuyên cần hoặc ít đặt WOW.`,
    },
    {
      title: "3. Phân loại Học sinh Yếu / Mạnh",
      desc: `Thuật toán: ${m.student_classification.selected_algorithm} (Accuracy ${m.student_classification.accuracy}%). Phân loại 4 nhóm (Yếu, Trung bình, Khá, Mạnh) kèm xác suất từng lớp và phát hiện kỹ năng BTVN kéo điểm.`,
    },
    {
      title: "4. Phân loại Lỗi sai Phổ biến (NLP)",
      desc: `Thuật toán: ${m.error_classification.algorithm}. Xử lý song ngữ Anh/Việt trên ${m.error_classification.num_categories} nhóm lỗi IELTS phổ biến (CV Accuracy ${m.error_classification.cv_accuracy}%), tự động tạo kế hoạch khắc phục ưu tiên.`,
    },
  ];

  document.getElementById("algoCards").innerHTML = cards
    .map((c) => `<div class="algo-card"><h4>${escapeHtml(c.title)}</h4><p>${escapeHtml(c.desc)}</p></div>`)
    .join("");
}

// ------------------------------------------------------------------
// Bootstrapping
// ------------------------------------------------------------------
async function init() {
  initTabs();

  try {
    await apiFetch("/api/health");
    document.getElementById("apiStatus").textContent = "API ML Sẵn sàng";
  } catch (err) {
    const pill = document.getElementById("apiStatus");
    pill.textContent = "Lỗi kết nối API";
    pill.classList.add("error");
    return;
  }

  sampleCache = await apiFetch("/api/sample-data");

  const presetWrap = document.getElementById("presetButtons");
  sampleCache.student_presets.forEach((preset) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn btn-ghost btn-sm";
    btn.textContent = preset.label;
    btn.addEventListener("click", () => {
      fillStudentForm(preset.data);
      runStudentAnalysis();
    });
    presetWrap.appendChild(btn);
  });

  document.getElementById("studentForm").addEventListener("submit", (e) => {
    e.preventDefault();
    runStudentAnalysis();
  });

  document.getElementById("loadSampleErrorBtn").addEventListener("click", () => {
    document.getElementById("errorInputText").value = sampleCache.sample_error_essay;
    runErrorClassification();
  });
  document.getElementById("analyzeErrorBtn").addEventListener("click", runErrorClassification);

  document.getElementById("batchJsonInput").value = JSON.stringify(sampleCache.sample_class, null, 2);
  document.getElementById("loadSampleClassBtn").addEventListener("click", () => {
    document.getElementById("batchJsonInput").value = JSON.stringify(sampleCache.sample_class, null, 2);
    runBatchAnalysis();
  });
  document.getElementById("runBatchBtn").addEventListener("click", runBatchAnalysis);

  document.getElementById("errorInputText").value = sampleCache.sample_error_essay;
  await Promise.all([runStudentAnalysis(), runErrorClassification(), runBatchAnalysis(), loadModelMetrics()]);
}

document.addEventListener("DOMContentLoaded", init);
