"""Automated verification tests for IELTS AI Learning Analytics ML & FastAPI API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.index import (  # noqa: E402
    BatchClassInput,
    ErrorTextInput,
    FullAssessmentInput,
    PUBLIC_DIR,
    batch_analyze,
    classify_errors,
    full_assessment,
    get_model_metrics,
    get_sample_data,
    health_check,
)


def test_health_and_metrics() -> None:
    health = health_check()
    assert health["status"] == "ok"

    m = get_model_metrics()
    assert m["ielts_prediction"]["overall_r2"] > 0.85
    assert m["learning_quality"]["r2"] > 0.90
    assert m["student_classification"]["accuracy"] > 78.0
    assert m["error_classification"]["cv_accuracy"] > 75.0


def test_full_assessment_weak_vs_strong() -> None:
    weak_res = full_assessment(
        FullAssessmentInput(
            entry_score=3.5,
            wow_sessions=2,
            attendance_rate=58.0,
            homework_completion_rate=45.0,
            hw_listening_pct=48.0,
            hw_reading_pct=46.0,
            hw_writing_pct=40.0,
            hw_speaking_pct=42.0,
            target_band=6.5,
        )
    )
    strong_res = full_assessment(
        FullAssessmentInput(
            entry_score=6.5,
            wow_sessions=20,
            attendance_rate=96.0,
            homework_completion_rate=95.0,
            hw_listening_pct=92.0,
            hw_reading_pct=90.0,
            hw_writing_pct=84.0,
            hw_speaking_pct=86.0,
            target_band=7.5,
        )
    )

    assert weak_res["learning_quality"]["learning_quality_score"] < 55.0
    assert strong_res["learning_quality"]["learning_quality_score"] > 85.0
    assert weak_res["ielts_prediction"]["predicted_overall"] < 5.0
    assert strong_res["ielts_prediction"]["predicted_overall"] >= 7.0
    assert weak_res["student_classification"]["predicted_tier"] == "Yếu"
    assert strong_res["student_classification"]["predicted_tier"] == "Mạnh"


def test_error_classification_accuracy() -> None:
    cases = [
        ("Although it was raining, but he went out.", "sentence_structure"),
        ("The number of students have increased sharply last year.", "grammar_tense"),
        ("We need to discuss about the problem despite of the cost.", "article_preposition"),
        ("Many students do mistakes and get useful informations.", "collocation_vocab"),
        ("The goverment must protect the enviroment.", "spelling_wordform"),
        ("Wrote three words when instruction said NO MORE THAN TWO WORDS in Listening.", "listening_reading_trap"),
    ]
    for text, expected_cat in cases:
        body = classify_errors(ErrorTextInput(text=text))
        assert body["total_segments"] == 1
        predicted = body["segment_results"][0]["category"]
        assert predicted == expected_cat, f"Expected {expected_cat} for '{text}', got {predicted}"


def test_batch_and_frontend_fields() -> None:
    sample = get_sample_data()
    b_data = batch_analyze(BatchClassInput(target_band=6.5, students=sample["sample_class"]))
    assert b_data["total_students"] == len(sample["sample_class"])
    assert "avg_predicted_ielts" in b_data["summary"]

    html = (PUBLIC_DIR / "index.html").read_text(encoding="utf-8")
    for required_label in (
        "Điểm đầu vào",
        "Số buổi đặt WOW",
        "Phần trăm chuyên cần",
        "Phần trăm làm BTVN",
        "Listening (BTVN)",
        "Reading (BTVN)",
        "Writing (BTVN)",
        "Speaking (BTVN)",
    ):
        assert required_label in html, f"Missing UI field: {required_label}"


if __name__ == "__main__":
    test_health_and_metrics()
    test_full_assessment_weak_vs_strong()
    test_error_classification_accuracy()
    test_batch_and_frontend_fields()
    print("ALL TESTS PASSED!")
