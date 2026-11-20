"""
Unified ML Inference & Analytics Service for IELTS Learning Analytics.

Thread-safe lazy singleton that loads pre-trained models from `models/ielts_models.joblib`,
with automatic fallback retraining in memory (Vercel read-only filesystem safe).
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from ml.data_generator import compute_official_overall, round_ielts_band
from ml.error_analyzer import analyze_error_batch
from ml.trainer import (
    PREDICTION_FEATURES,
    QUALITY_FEATURES,
    TIER_ORDER,
    train_all_models,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MODEL_BUNDLE_PATH = MODELS_DIR / "ielts_models.joblib"

SKILL_LABELS_VI = {
    "listening": "Listening (Nghe)",
    "reading": "Reading (Đọc)",
    "writing": "Writing (Viết)",
    "speaking": "Speaking (Nói)",
}


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class IELTSAnalyticsService:
    """Core ML service providing all 4 analytical capabilities."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle: Optional[Dict[str, Any]] = None

    def _ensure_loaded(self) -> Dict[str, Any]:
        if self._bundle is not None:
            return self._bundle

        with self._lock:
            if self._bundle is not None:
                return self._bundle

            if MODEL_BUNDLE_PATH.exists():
                try:
                    loaded = joblib.load(MODEL_BUNDLE_PATH)
                    if (
                        loaded.get("schema_version") == 2
                        and loaded.get("prediction_features") == PREDICTION_FEATURES
                        and "skill_models" in loaded
                    ):
                        self._bundle = loaded
                        return self._bundle
                except Exception:
                    pass

            bundle, _ = train_all_models(DATA_DIR, models_dir=None, regenerate_data=False)
            self._bundle = bundle
            return self._bundle

    def get_metrics(self) -> Dict[str, Any]:
        return self._ensure_loaded()["metrics"]

    # ------------------------------------------------------------------
    # MODULE 1: Đánh giá chất lượng học tập
    # ------------------------------------------------------------------
    def evaluate_learning_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self._ensure_loaded()
        x_vec = np.array([[float(data[f]) for f in QUALITY_FEATURES]], dtype=np.float64)

        raw_score = float(bundle["quality_model"].predict(x_vec)[0])
        quality_score = round(float(np.clip(raw_score, 0.0, 100.0)), 1)

        if quality_score >= 85:
            grade, grade_color = "Xuất sắc", "success"
        elif quality_score >= 70:
            grade, grade_color = "Tốt", "info"
        elif quality_score >= 55:
            grade, grade_color = "Trung bình", "warning"
        else:
            grade, grade_color = "Cần cải thiện gấp", "danger"

        x_scaled = bundle["cluster_scaler"].transform(x_vec)
        cluster_id = int(bundle["kmeans"].predict(x_scaled)[0])
        persona = bundle["cluster_to_persona"][cluster_id]

        wow_pct = round(min(float(data["wow_sessions"]) / 18.0, 1.0) * 100.0, 1)
        dimensions = {
            "% Chuyên cần": round(float(data["attendance_rate"]), 1),
            "% Làm BTVN": round(float(data["homework_completion_rate"]), 1),
            "Tần suất WOW": wow_pct,
            "BTVN Listening": round(float(data["hw_listening_pct"]), 1),
            "BTVN Reading": round(float(data["hw_reading_pct"]), 1),
            "BTVN Writing": round(float(data["hw_writing_pct"]), 1),
            "BTVN Speaking": round(float(data["hw_speaking_pct"]), 1),
        }

        strengths: List[str] = []
        risks: List[str] = []
        recommendations: List[str] = []

        if float(data["attendance_rate"]) >= 85:
            strengths.append("Tỷ lệ chuyên cần cao (≥ 85%), theo sát tiến độ giảng dạy trên lớp.")
        elif float(data["attendance_rate"]) < 75:
            risks.append("Phần trăm chuyên cần thấp (< 75%), dễ bỏ lỡ kiến thức trọng tâm.")
            recommendations.append("Duy trì đi học tối thiểu 85% số buổi và bù bài ngay khi vắng.")

        if float(data["homework_completion_rate"]) >= 85:
            strengths.append("Hoàn thành bài tập về nhà rất đều đặn và kỷ luật.")
        elif float(data["homework_completion_rate"]) < 70:
            risks.append("Phần trăm làm BTVN chưa đạt yêu cầu (< 70%).")
            recommendations.append("Hoàn thành tối thiểu 85% BTVN mỗi tuần để giữ nhịp luyện đề.")

        if float(data["wow_sessions"]) >= 12:
            strengths.append(f"Tận dụng tốt buổi đặt WOW ({int(data['wow_sessions'])} buổi) để sửa lỗi trực tiếp.")
        elif float(data["wow_sessions"]) < 6:
            risks.append("Số buổi đặt WOW còn ít (< 6 buổi), thiếu kèm sửa lỗi chuyên sâu 1-1.")
            recommendations.append("Đặt thêm ít nhất 2 buổi WOW/tuần tập trung chữa bài Writing & Speaking.")

        hw_skills = {
            "Listening": float(data["hw_listening_pct"]),
            "Reading": float(data["hw_reading_pct"]),
            "Writing": float(data["hw_writing_pct"]),
            "Speaking": float(data["hw_speaking_pct"]),
        }
        lowest_hw_skill = min(hw_skills, key=hw_skills.get)
        if hw_skills[lowest_hw_skill] < 65:
            risks.append(
                f"Điểm % BTVN kỹ năng {lowest_hw_skill} đang thấp nhất ({hw_skills[lowest_hw_skill]:.0f}%)."
            )
            recommendations.append(
                f"Ưu tiên dành 40% thời lượng buổi WOW sắp tới để kéo điểm BTVN {lowest_hw_skill} lên trên 75%."
            )
        else:
            strengths.append("Điểm phần trăm BTVN ở cả 4 kỹ năng đều khá đồng đều (≥ 65%).")

        if not recommendations:
            recommendations.append("Giữ vững phong độ chuyên cần, BTVN và đặt thêm buổi WOW luyện đề áp lực thời gian.")

        return {
            "learning_quality_score": quality_score,
            "grade": grade,
            "grade_color": grade_color,
            "learning_persona": persona,
            "dimensions": dimensions,
            "strengths": strengths[:4],
            "risks": risks[:4],
            "recommendations": recommendations[:4],
        }

    # ------------------------------------------------------------------
    # MODULE 2: Dự đoán Band IELTS thực tế
    # ------------------------------------------------------------------
    def predict_ielts_score(self, data: Dict[str, Any], target_band: float = 6.5) -> Dict[str, Any]:
        bundle = self._ensure_loaded()
        x_vec = np.array([[float(data[f]) for f in PREDICTION_FEATURES]], dtype=np.float64)

        raw_skills: Dict[str, float] = {}
        rounded_skills: Dict[str, float] = {}
        skill_intervals: Dict[str, Dict[str, float]] = {}

        for skill in ("listening", "reading", "writing", "speaking"):
            pred_val = float(np.clip(bundle["skill_models"][skill].predict(x_vec)[0], 1.0, 9.0))
            std_val = float(bundle["skill_residual_std"][skill])
            raw_skills[skill] = round(pred_val, 2)
            rounded_skills[skill] = round_ielts_band(pred_val)
            skill_intervals[skill] = {
                "low": round_ielts_band(pred_val - 1.64 * std_val),
                "high": round_ielts_band(pred_val + 1.64 * std_val),
            }

        predicted_overall = compute_official_overall(
            rounded_skills["listening"],
            rounded_skills["reading"],
            rounded_skills["writing"],
            rounded_skills["speaking"],
        )
        raw_overall_mean = sum(raw_skills.values()) / 4.0
        overall_std = max(0.18, float(bundle["overall_residual_std"]))

        threshold = target_band - 0.25
        z_score = (raw_overall_mean - threshold) / overall_std
        prob_reach_target = round(float(np.clip(_normal_cdf(z_score) * 100.0, 1.0, 99.5)), 1)

        entry_score = round_ielts_band(float(data["entry_score"]))
        expected_gain = round(predicted_overall - entry_score, 1)

        # What-if simulation: +6 buổi WOW, +10% chuyên cần & +15% BTVN
        what_if_data = dict(data)
        what_if_data["wow_sessions"] = min(35, int(data["wow_sessions"]) + 6)
        what_if_data["attendance_rate"] = min(100.0, float(data["attendance_rate"]) + 10.0)
        what_if_data["homework_completion_rate"] = min(100.0, float(data["homework_completion_rate"]) + 15.0)
        for k in ("hw_listening_pct", "hw_reading_pct", "hw_writing_pct", "hw_speaking_pct"):
            what_if_data[k] = min(100.0, float(data[k]) + 8.0)

        x_whatif = np.array([[float(what_if_data[f]) for f in PREDICTION_FEATURES]], dtype=np.float64)
        whatif_skills = {
            s: round_ielts_band(float(bundle["skill_models"][s].predict(x_whatif)[0]))
            for s in ("listening", "reading", "writing", "speaking")
        }
        whatif_overall = compute_official_overall(
            whatif_skills["listening"],
            whatif_skills["reading"],
            whatif_skills["writing"],
            whatif_skills["speaking"],
        )

        return {
            "entry_score": entry_score,
            "predicted_overall": predicted_overall,
            "raw_overall_mean": round(raw_overall_mean, 2),
            "expected_gain": expected_gain,
            "confidence_interval_90": {
                "low": round_ielts_band(raw_overall_mean - 1.64 * overall_std),
                "high": round_ielts_band(raw_overall_mean + 1.64 * overall_std),
            },
            "predicted_skills": rounded_skills,
            "raw_skills": raw_skills,
            "skill_intervals": skill_intervals,
            "target_band": target_band,
            "probability_reach_target": prob_reach_target,
            "what_if_simulation": {
                "description": "Nếu đặt thêm +6 buổi WOW, tăng +10% chuyên cần, +15% làm BTVN và cải thiện +8% điểm 4 kỹ năng BTVN",
                "projected_overall": whatif_overall,
                "potential_bonus": round(max(0.0, whatif_overall - predicted_overall), 1),
            },
        }

    # ------------------------------------------------------------------
    # MODULE 3: Phân loại học sinh Yếu / Mạnh
    # ------------------------------------------------------------------
    def classify_student(self, data: Dict[str, Any]) -> Dict[str, Any]:
        bundle = self._ensure_loaded()
        x_vec = np.array([[float(data[f]) for f in PREDICTION_FEATURES]], dtype=np.float64)

        clf = bundle["tier_clf"]
        probs = clf.predict_proba(x_vec)[0]
        prob_by_class = {cls: round(float(p) * 100.0, 1) for cls, p in zip(clf.classes_, probs)}
        ordered_probs = [
            {"tier": tier, "probability": prob_by_class.get(tier, 0.0)} for tier in TIER_ORDER
        ]
        predicted_tier = str(clf.predict(x_vec)[0])
        confidence = prob_by_class.get(predicted_tier, 0.0)

        # Diagnose strengths/weaknesses from 4 homework skill % + predicted bands
        hw_scores = {
            "listening": float(data["hw_listening_pct"]),
            "reading": float(data["hw_reading_pct"]),
            "writing": float(data["hw_writing_pct"]),
            "speaking": float(data["hw_speaking_pct"]),
        }
        avg_hw = sum(hw_scores.values()) / 4.0
        strongest_skill = max(hw_scores, key=hw_scores.get)
        weakest_skill = min(hw_scores, key=hw_scores.get)
        skill_gap = round(hw_scores[strongest_skill] - hw_scores[weakest_skill], 1)

        skill_diagnosis = []
        for s_key, pct in hw_scores.items():
            if pct >= avg_hw + 6.0 or pct >= 80.0:
                status, badge = "Điểm mạnh", "success"
            elif pct <= avg_hw - 6.0 or pct < 60.0:
                status, badge = "Điểm yếu cần ưu tiên", "danger"
            else:
                status, badge = "Cân bằng", "info"
            skill_diagnosis.append(
                {
                    "skill": s_key,
                    "label_vi": SKILL_LABELS_VI[s_key],
                    "hw_pct": round(pct, 1),
                    "status": status,
                    "badge": badge,
                }
            )

        tier_meta = {
            "Yếu": {
                "badge": "danger",
                "group": "Nhóm Yếu (< 5.0)",
                "action_plan": "Cần học vụ kèm sát lịch đặt WOW 1-1, siết tỷ lệ làm BTVN ≥ 80% và củng cố lại nền tảng ngữ pháp/từ vựng.",
            },
            "Trung bình": {
                "badge": "warning",
                "group": "Nhóm Trung bình (5.0 – 5.5)",
                "action_plan": "Tăng cường chữa bài BTVN chi tiết trong buổi WOW, tập trung nâng kỹ năng thấp nhất lên trên 70%.",
            },
            "Khá": {
                "badge": "info",
                "group": "Nhóm Khá (6.0 – 6.5)",
                "action_plan": "Duy trì chuyên cần cao, đẩy mạnh luyện đề chuẩn thời gian thực và nâng chất lượng bài Writing/Speaking.",
            },
            "Mạnh": {
                "badge": "success",
                "group": "Nhóm Mạnh (≥ 7.0)",
                "action_plan": "Sẵn sàng thi thực tế hướng tới 7.5–8.5+; ưu tiên tinh chỉnh collocation học thuật và xử lý bẫy câu khó.",
            },
        }
        meta = tier_meta.get(predicted_tier, tier_meta["Trung bình"])

        return {
            "predicted_tier": predicted_tier,
            "group_label": meta["group"],
            "badge": meta["badge"],
            "confidence": confidence,
            "tier_probabilities": ordered_probs,
            "strongest_skill": {
                "skill": strongest_skill,
                "label_vi": SKILL_LABELS_VI[strongest_skill],
                "hw_pct": hw_scores[strongest_skill],
            },
            "weakest_skill": {
                "skill": weakest_skill,
                "label_vi": SKILL_LABELS_VI[weakest_skill],
                "hw_pct": hw_scores[weakest_skill],
            },
            "skill_imbalance_gap": skill_gap,
            "skill_diagnosis": skill_diagnosis,
            "action_plan": meta["action_plan"],
        }

    # ------------------------------------------------------------------
    # MODULE 4: Phân loại lỗi sai phổ biến (NLP)
    # ------------------------------------------------------------------
    def classify_errors(self, raw_text: str) -> Dict[str, Any]:
        bundle = self._ensure_loaded()
        return analyze_error_batch(
            raw_text=raw_text,
            nlp_pipeline=bundle["error_pipeline"],
            classes=bundle["error_classes"],
        )

    # ------------------------------------------------------------------
    # ALL-IN-ONE: Full Student Report + Batch Class Analysis
    # ------------------------------------------------------------------
    def full_assessment(
        self,
        student_data: Dict[str, Any],
        target_band: float = 6.5,
        error_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        quality = self.evaluate_learning_quality(student_data)
        prediction = self.predict_ielts_score(student_data, target_band=target_band)
        classification = self.classify_student(student_data)
        errors = self.classify_errors(error_text) if error_text and error_text.strip() else None

        return {
            "learning_quality": quality,
            "ielts_prediction": prediction,
            "student_classification": classification,
            "error_analysis": errors,
        }

    def batch_analyze_class(self, students: List[Dict[str, Any]], target_band: float = 6.5) -> Dict[str, Any]:
        if not students:
            return {"total_students": 0, "summary": {}, "students": []}

        tier_counts = {t: 0 for t in TIER_ORDER}
        weak_skill_counts = {s: 0 for s in ("listening", "reading", "writing", "speaking")}
        overall_scores: List[float] = []
        quality_scores: List[float] = []
        detailed: List[Dict[str, Any]] = []

        for idx, stu in enumerate(students):
            name = str(stu.get("name") or f"Học sinh #{idx + 1}")
            qual = self.evaluate_learning_quality(stu)
            pred = self.predict_ielts_score(stu, target_band=target_band)
            cls = self.classify_student(stu)

            tier = cls["predicted_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            weak_skill_counts[cls["weakest_skill"]["skill"]] += 1
            overall_scores.append(pred["predicted_overall"])
            quality_scores.append(qual["learning_quality_score"])

            detailed.append(
                {
                    "name": name,
                    "entry_score": pred["entry_score"],
                    "wow_sessions": int(stu["wow_sessions"]),
                    "attendance_rate": float(stu["attendance_rate"]),
                    "homework_completion_rate": float(stu["homework_completion_rate"]),
                    "learning_quality_score": qual["learning_quality_score"],
                    "quality_grade": qual["grade"],
                    "persona": qual["learning_persona"]["name"],
                    "predicted_overall": pred["predicted_overall"],
                    "expected_gain": pred["expected_gain"],
                    "tier": tier,
                    "badge": cls["badge"],
                    "weakest_skill": cls["weakest_skill"]["label_vi"],
                    "probability_reach_target": pred["probability_reach_target"],
                }
            )

        detailed.sort(key=lambda r: (r["predicted_overall"], r["learning_quality_score"]))
        top_weak_skill = max(weak_skill_counts, key=weak_skill_counts.get)

        return {
            "total_students": len(students),
            "summary": {
                "avg_predicted_ielts": round(float(np.mean(overall_scores)), 2),
                "avg_learning_quality": round(float(np.mean(quality_scores)), 1),
                "tier_distribution": tier_counts,
                "at_risk_count": tier_counts["Yếu"],
                "strong_count": tier_counts["Mạnh"],
                "most_common_weak_skill": SKILL_LABELS_VI[top_weak_skill],
                "weak_skill_distribution": {
                    SKILL_LABELS_VI[k]: v for k, v in weak_skill_counts.items()
                },
            },
            "students": detailed,
        }


analytics_service = IELTSAnalyticsService()
