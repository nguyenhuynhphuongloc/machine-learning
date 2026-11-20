"""
FastAPI Backend for IELTS AI Learning Analytics.

Entrypoint for both Vercel Serverless Functions (`/api/index.py`)
and local development (`uvicorn api.index:app --reload`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.service import analytics_service  # noqa: E402

PUBLIC_DIR = ROOT_DIR / "public"

app = FastAPI(
    title="IELTS AI Learning Analytics API",
    description=(
        "Hệ thống Học máy dự đoán Band IELTS thực tế từ điểm đầu vào, số buổi đặt WOW, "
        "% chuyên cần, % làm BTVN và % điểm 4 kỹ năng BTVN; đánh giá chất lượng học tập, "
        "phân loại học sinh yếu/mạnh và phân loại lỗi sai phổ biến."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Pydantic Schemas
# ------------------------------------------------------------------
class LearningHabitInput(BaseModel):
    wow_sessions: int = Field(12, ge=0, le=60, description="Số buổi đặt WOW")
    attendance_rate: float = Field(88.0, ge=0.0, le=100.0, description="Phần trăm chuyên cần (%)")
    homework_completion_rate: float = Field(85.0, ge=0.0, le=100.0, description="Phần trăm làm bài tập về nhà (%)")
    hw_listening_pct: float = Field(78.0, ge=0.0, le=100.0, description="Điểm % BTVN Listening")
    hw_reading_pct: float = Field(76.0, ge=0.0, le=100.0, description="Điểm % BTVN Reading")
    hw_writing_pct: float = Field(66.0, ge=0.0, le=100.0, description="Điểm % BTVN Writing")
    hw_speaking_pct: float = Field(68.0, ge=0.0, le=100.0, description="Điểm % BTVN Speaking")


class StudentProfileInput(LearningHabitInput):
    name: Optional[str] = Field(None, description="Tên học sinh (tùy chọn)")
    entry_score: float = Field(5.0, ge=1.0, le=9.0, description="Điểm IELTS đầu vào")
    target_band: float = Field(6.5, ge=4.0, le=9.0, description="Band IELTS mục tiêu")


class ErrorTextInput(BaseModel):
    text: str = Field(..., min_length=3, max_length=12000, description="Đoạn văn bài viết hoặc danh sách lỗi sai")


class FullAssessmentInput(StudentProfileInput):
    error_text: Optional[str] = Field(None, max_length=12000, description="Đoạn văn/lỗi sai cần phân tích thêm")


class BatchClassInput(BaseModel):
    target_band: float = Field(6.5, ge=4.0, le=9.0)
    students: List[StudentProfileInput] = Field(..., min_length=1, max_length=200)


# ------------------------------------------------------------------
# REST Endpoints
# ------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    metrics = analytics_service.get_metrics()
    return {
        "status": "ok",
        "service": "IELTS AI Learning Analytics",
        "trained_at": metrics.get("trained_at"),
    }


@app.get("/api/model-metrics")
def get_model_metrics():
    return analytics_service.get_metrics()


@app.post("/api/evaluate-quality")
def evaluate_quality(payload: LearningHabitInput):
    return analytics_service.evaluate_learning_quality(payload.model_dump())


@app.post("/api/predict-ielts")
def predict_ielts(payload: StudentProfileInput):
    return analytics_service.predict_ielts_score(
        payload.model_dump(), target_band=payload.target_band
    )


@app.post("/api/classify-student")
def classify_student(payload: StudentProfileInput):
    return analytics_service.classify_student(payload.model_dump())


@app.post("/api/classify-errors")
def classify_errors(payload: ErrorTextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Vui lòng nhập nội dung câu hoặc danh sách lỗi sai.")
    return analytics_service.classify_errors(payload.text)


@app.post("/api/full-assessment")
def full_assessment(payload: FullAssessmentInput):
    return analytics_service.full_assessment(
        student_data=payload.model_dump(),
        target_band=payload.target_band,
        error_text=payload.error_text,
    )


@app.post("/api/batch-analyze")
def batch_analyze(payload: BatchClassInput):
    students_dicts = [s.model_dump() for s in payload.students]
    return analytics_service.batch_analyze_class(students_dicts, target_band=payload.target_band)


@app.get("/api/sample-data")
def get_sample_data():
    """Return realistic Vietnamese IELTS center presets matching the 8 exact fields."""
    return {
        "student_presets": [
            {
                "id": "weak",
                "label": "Học sinh cần kèm sát (Yếu)",
                "data": {
                    "name": "Nguyễn Minh Quân",
                    "entry_score": 3.5,
                    "wow_sessions": 2,
                    "attendance_rate": 58.0,
                    "homework_completion_rate": 45.0,
                    "hw_listening_pct": 48.0,
                    "hw_reading_pct": 46.0,
                    "hw_writing_pct": 40.0,
                    "hw_speaking_pct": 42.0,
                    "target_band": 5.5,
                },
            },
            {
                "id": "average",
                "label": "Học sinh tiến bộ ổn định (Khá)",
                "data": {
                    "name": "Trần Thảo Vy",
                    "entry_score": 5.0,
                    "wow_sessions": 12,
                    "attendance_rate": 88.0,
                    "homework_completion_rate": 85.0,
                    "hw_listening_pct": 78.0,
                    "hw_reading_pct": 76.0,
                    "hw_writing_pct": 66.0,
                    "hw_speaking_pct": 68.0,
                    "target_band": 6.5,
                },
            },
            {
                "id": "strong",
                "label": "Học sinh mục tiêu 7.5+ (Mạnh)",
                "data": {
                    "name": "Lê Hoàng Anh",
                    "entry_score": 6.5,
                    "wow_sessions": 20,
                    "attendance_rate": 96.0,
                    "homework_completion_rate": 95.0,
                    "hw_listening_pct": 92.0,
                    "hw_reading_pct": 90.0,
                    "hw_writing_pct": 84.0,
                    "hw_speaking_pct": 86.0,
                    "target_band": 7.5,
                },
            },
        ],
        "sample_error_essay": (
            "Although studying abroad is expensive, but many students still want to go.\n"
            "The number of university students have increased sharply last year.\n"
            "We need to discuss about the impact of social media in young people.\n"
            "Many students do a lot of mistakes and collect many useful informations.\n"
            "The goverment should protect the enviroment because it is very benefit for society.\n"
            "In the other hand, online learning is good. However, it is hard. However, it is fast.\n"
            "Fell for speaker self-correction in Listening Section 1 and wrote three words instead of NO MORE THAN TWO WORDS.\n"
            "Thiếu đoạn Overview tổng quan trong IELTS Writing Task 1 và bài viết chỉ dài 180 từ."
        ),
        "sample_class": [
            {
                "name": "Nguyễn Minh Quân",
                "entry_score": 3.5, "wow_sessions": 2, "attendance_rate": 58, "homework_completion_rate": 45,
                "hw_listening_pct": 48, "hw_reading_pct": 46, "hw_writing_pct": 40, "hw_speaking_pct": 42,
                "target_band": 6.5,
            },
            {
                "name": "Phạm Gia Huy",
                "entry_score": 4.5, "wow_sessions": 7, "attendance_rate": 74, "homework_completion_rate": 66,
                "hw_listening_pct": 64, "hw_reading_pct": 62, "hw_writing_pct": 54, "hw_speaking_pct": 58,
                "target_band": 6.5,
            },
            {
                "name": "Trần Thảo Vy",
                "entry_score": 5.0, "wow_sessions": 12, "attendance_rate": 88, "homework_completion_rate": 85,
                "hw_listening_pct": 78, "hw_reading_pct": 76, "hw_writing_pct": 66, "hw_speaking_pct": 68,
                "target_band": 6.5,
            },
            {
                "name": "Đỗ Khánh Linh",
                "entry_score": 5.5, "wow_sessions": 16, "attendance_rate": 92, "homework_completion_rate": 90,
                "hw_listening_pct": 82, "hw_reading_pct": 84, "hw_writing_pct": 74, "hw_speaking_pct": 76,
                "target_band": 6.5,
            },
            {
                "name": "Lê Hoàng Anh",
                "entry_score": 6.5, "wow_sessions": 22, "attendance_rate": 96, "homework_completion_rate": 95,
                "hw_listening_pct": 92, "hw_reading_pct": 90, "hw_writing_pct": 84, "hw_speaking_pct": 86,
                "target_band": 6.5,
            },
            {
                "name": "Vũ Bảo Ngọc",
                "entry_score": 4.0, "wow_sessions": 4, "attendance_rate": 68, "homework_completion_rate": 55,
                "hw_listening_pct": 56, "hw_reading_pct": 54, "hw_writing_pct": 46, "hw_speaking_pct": 50,
                "target_band": 6.5,
            },
        ],
    }


# Serve static frontend locally (on Vercel, `public/` is served directly by CDN)
if PUBLIC_DIR.exists():
    @app.get("/", include_in_schema=False)
    def serve_index():
        return FileResponse(PUBLIC_DIR / "index.html")

    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="static")
