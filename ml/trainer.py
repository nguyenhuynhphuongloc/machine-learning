"""
End-to-end Machine Learning Training Pipeline for IELTS Learning Analytics.

Uses the exact 8 center-ready features:
- entry_score (Điểm đầu vào)
- wow_sessions (Số buổi đặt WOW)
- attendance_rate (% chuyên cần)
- homework_completion_rate (% làm BTVN)
- hw_listening_pct, hw_reading_pct, hw_writing_pct, hw_speaking_pct (% điểm 4 kỹ năng BTVN)

Trains and evaluates 4 core ML modules:
1. Learning Quality Assessment: GradientBoostingRegressor + KMeans(4) Persona Clustering
2. Actual IELTS Band Prediction: 4x GradientBoostingRegressor (L/R/W/S) benchmarked vs Ridge & RandomForest
3. Student Strength/Weakness Classification: RandomForestClassifier benchmarked vs LogisticRegression
4. Common IELTS Error Classification: FeatureUnion(Word TF-IDF + Char TF-IDF) + LogisticRegression
"""

from __future__ import annotations

import os

# Prevent joblib/loky from spawning wmic subprocesses on Windows/serverless environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data_generator import (
    compute_official_overall,
    generate_error_dataset,
    generate_student_dataset,
    round_ielts_band,
    save_datasets,
)

QUALITY_FEATURES = [
    "wow_sessions",
    "attendance_rate",
    "homework_completion_rate",
    "hw_listening_pct",
    "hw_reading_pct",
    "hw_writing_pct",
    "hw_speaking_pct",
]

PREDICTION_FEATURES = [
    "entry_score",
    "wow_sessions",
    "attendance_rate",
    "homework_completion_rate",
    "hw_listening_pct",
    "hw_reading_pct",
    "hw_writing_pct",
    "hw_speaking_pct",
]

FEATURE_LABELS_VI = {
    "entry_score": "Điểm đầu vào (Band)",
    "wow_sessions": "Số buổi đặt WOW",
    "attendance_rate": "% Chuyên cần",
    "homework_completion_rate": "% Làm BTVN",
    "hw_listening_pct": "% Điểm BTVN Listening",
    "hw_reading_pct": "% Điểm BTVN Reading",
    "hw_writing_pct": "% Điểm BTVN Writing",
    "hw_speaking_pct": "% Điểm BTVN Speaking",
}

PERSONA_PROFILES = [
    {
        "rank": 0,
        "name": "Cần hỗ trợ sát sao",
        "badge": "danger",
        "summary": "Chuyên cần, % BTVN và số buổi WOW còn thấp — cần học vụ nhắc lịch và kèm 1-1 ngay để tránh tụt band.",
    },
    {
        "rank": 1,
        "name": "Tiềm năng — Thiếu đều đặn",
        "badge": "warning",
        "summary": "Có nền tảng ở mức trung bình nhưng chưa đều tay; chỉ cần tăng % BTVN và đặt thêm buổi WOW sẽ bứt phá rõ rệt.",
    },
    {
        "rank": 2,
        "name": "Tiến bộ Ổn định",
        "badge": "info",
        "summary": "Duy trì chuyên cần tốt, làm BTVN đầy đủ và tận dụng buổi WOW hiệu quả; sẵn sàng đẩy mạnh kỹ năng còn thấp nhất.",
    },
    {
        "rank": 3,
        "name": "Xuất sắc & Bền vững",
        "badge": "success",
        "summary": "Chất lượng học tập rất cao ở cả 4 kỹ năng BTVN và chuyên cần; tiềm năng đạt Band thực tế 7.0–8.5+.",
    },
]

TIER_ORDER = ["Yếu", "Trung bình", "Khá", "Mạnh"]


def _load_or_generate_rows(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    students_csv = data_dir / "ielts_students.csv"
    errors_csv = data_dir / "ielts_errors.csv"

    if students_csv.exists() and errors_csv.exists():
        with students_csv.open("r", encoding="utf-8") as f:
            student_rows = list(csv.DictReader(f))
        with errors_csv.open("r", encoding="utf-8") as f:
            error_rows = list(csv.DictReader(f))
        # Verify schema matches new columns
        if student_rows and "entry_score" in student_rows[0] and "wow_sessions" in student_rows[0]:
            return student_rows, error_rows

    return generate_student_dataset(n_samples=1800, seed=42), generate_error_dataset(seed=42)


def _extract_matrix(rows: List[Dict[str, Any]], columns: List[str]) -> np.ndarray:
    return np.array([[float(r[col]) for col in columns] for r in rows], dtype=np.float64)


def train_all_models(
    data_dir: Path,
    models_dir: Path | None = None,
    regenerate_data: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train all 4 ML modules, compute evaluation metrics, and optionally persist artifacts."""
    if regenerate_data:
        save_datasets(data_dir, n_students=1800, seed=42)

    student_rows, error_rows = _load_or_generate_rows(data_dir)

    # ------------------------------------------------------------------
    # 1. LEARNING QUALITY ASSESSMENT (Regression + KMeans Clustering)
    # ------------------------------------------------------------------
    X_qual = _extract_matrix(student_rows, QUALITY_FEATURES)
    y_qual = np.array([float(r["learning_quality_score"]) for r in student_rows], dtype=np.float64)

    Xq_tr, Xq_te, yq_tr, yq_te = train_test_split(X_qual, y_qual, test_size=0.2, random_state=42)

    quality_model = GradientBoostingRegressor(
        n_estimators=95,
        max_depth=4,
        learning_rate=0.07,
        random_state=42,
    )
    quality_model.fit(Xq_tr, yq_tr)
    yq_pred = quality_model.predict(Xq_te)

    quality_metrics = {
        "algorithm": "GradientBoostingRegressor + KMeans(4)",
        "r2": round(float(r2_score(yq_te, yq_pred)), 4),
        "mae": round(float(mean_absolute_error(yq_te, yq_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(yq_te, yq_pred))), 3),
    }

    cluster_scaler = StandardScaler()
    X_qual_scaled = cluster_scaler.fit_transform(X_qual)
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_qual_scaled)
    sil_score = float(silhouette_score(X_qual_scaled, cluster_labels))
    quality_metrics["silhouette_score"] = round(sil_score, 4)

    cluster_means = [
        (cid, float(np.mean(y_qual[cluster_labels == cid]))) for cid in range(4)
    ]
    cluster_means.sort(key=lambda item: item[1])
    cluster_to_persona = {
        int(cid): PERSONA_PROFILES[rank] for rank, (cid, _) in enumerate(cluster_means)
    }

    quality_feature_importance = sorted(
        [
            {
                "feature": col,
                "label_vi": FEATURE_LABELS_VI[col],
                "importance": round(float(imp) * 100.0, 2),
            }
            for col, imp in zip(QUALITY_FEATURES, quality_model.feature_importances_)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # ------------------------------------------------------------------
    # 2. ACTUAL IELTS BAND PREDICTION (Multi-skill Regression + Benchmark)
    # ------------------------------------------------------------------
    X_pred = _extract_matrix(student_rows, PREDICTION_FEATURES)
    y_skills = {
        "listening": np.array([float(r["actual_listening"]) for r in student_rows]),
        "reading": np.array([float(r["actual_reading"]) for r in student_rows]),
        "writing": np.array([float(r["actual_writing"]) for r in student_rows]),
        "speaking": np.array([float(r["actual_speaking"]) for r in student_rows]),
    }
    y_overall = np.array([float(r["actual_overall"]) for r in student_rows])

    indices = np.arange(len(student_rows))
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=42)

    skill_models: Dict[str, GradientBoostingRegressor] = {}
    skill_residual_std: Dict[str, float] = {}
    skill_metrics: Dict[str, Dict[str, float]] = {}
    te_skill_preds: Dict[str, np.ndarray] = {}

    for skill, y_vec in y_skills.items():
        gbr = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.06,
            random_state=42,
        )
        gbr.fit(X_pred[tr_idx], y_vec[tr_idx])
        preds = gbr.predict(X_pred[te_idx])
        residuals = y_vec[te_idx] - preds

        skill_models[skill] = gbr
        skill_residual_std[skill] = round(float(np.std(residuals)), 4)
        te_skill_preds[skill] = preds
        skill_metrics[skill] = {
            "r2": round(float(r2_score(y_vec[te_idx], preds)), 4),
            "mae": round(float(mean_absolute_error(y_vec[te_idx], preds)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y_vec[te_idx], preds))), 3),
        }

    overall_preds_gbr = np.array(
        [
            compute_official_overall(
                round_ielts_band(te_skill_preds["listening"][i]),
                round_ielts_band(te_skill_preds["reading"][i]),
                round_ielts_band(te_skill_preds["writing"][i]),
                round_ielts_band(te_skill_preds["speaking"][i]),
            )
            for i in range(len(te_idx))
        ]
    )

    ridge_bench = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=2.0))])
    ridge_bench.fit(X_pred[tr_idx], y_overall[tr_idx])
    ridge_preds = ridge_bench.predict(X_pred[te_idx])

    rf_bench = RandomForestRegressor(n_estimators=70, max_depth=10, random_state=42, n_jobs=1)
    rf_bench.fit(X_pred[tr_idx], y_overall[tr_idx])
    rf_preds = rf_bench.predict(X_pred[te_idx])

    overall_residual_std = float(np.std(y_overall[te_idx] - overall_preds_gbr))

    ielts_prediction_metrics = {
        "selected_algorithm": "GradientBoostingRegressor (4-Skill Ensemble)",
        "overall_r2": round(float(r2_score(y_overall[te_idx], overall_preds_gbr)), 4),
        "overall_mae": round(float(mean_absolute_error(y_overall[te_idx], overall_preds_gbr)), 3),
        "overall_rmse": round(float(np.sqrt(mean_squared_error(y_overall[te_idx], overall_preds_gbr))), 3),
        "within_half_band_accuracy": round(
            float(np.mean(np.abs(y_overall[te_idx] - overall_preds_gbr) <= 0.5)) * 100.0, 1
        ),
        "per_skill": skill_metrics,
        "algorithm_benchmark": [
            {
                "name": "Gradient Boosting (4-Skill)",
                "r2": round(float(r2_score(y_overall[te_idx], overall_preds_gbr)), 4),
                "mae": round(float(mean_absolute_error(y_overall[te_idx], overall_preds_gbr)), 3),
            },
            {
                "name": "Random Forest Regressor",
                "r2": round(float(r2_score(y_overall[te_idx], rf_preds)), 4),
                "mae": round(float(mean_absolute_error(y_overall[te_idx], rf_preds)), 3),
            },
            {
                "name": "Ridge Regression",
                "r2": round(float(r2_score(y_overall[te_idx], ridge_preds)), 4),
                "mae": round(float(mean_absolute_error(y_overall[te_idx], ridge_preds)), 3),
            },
        ],
    }

    avg_imp = np.mean([m.feature_importances_ for m in skill_models.values()], axis=0)
    prediction_feature_importance = sorted(
        [
            {
                "feature": col,
                "label_vi": FEATURE_LABELS_VI[col],
                "importance": round(float(imp) * 100.0, 2),
            }
            for col, imp in zip(PREDICTION_FEATURES, avg_imp)
        ],
        key=lambda x: x["importance"],
        reverse=True,
    )

    # ------------------------------------------------------------------
    # 3. STUDENT STRENGTH / WEAKNESS CLASSIFICATION (RF vs Logistic)
    # ------------------------------------------------------------------
    y_tier = np.array([str(r["student_tier"]) for r in student_rows])
    Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
        X_pred, y_tier, test_size=0.2, random_state=42, stratify=y_tier
    )

    tier_clf = RandomForestClassifier(
        n_estimators=90,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    tier_clf.fit(Xt_tr, yt_tr)
    yt_pred_rf = tier_clf.predict(Xt_te)

    lr_tier_bench = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    lr_tier_bench.fit(Xt_tr, yt_tr)
    yt_pred_lr = lr_tier_bench.predict(Xt_te)

    tier_metrics = {
        "selected_algorithm": "RandomForestClassifier",
        "accuracy": round(float(accuracy_score(yt_te, yt_pred_rf)) * 100.0, 2),
        "f1_macro": round(float(f1_score(yt_te, yt_pred_rf, average="macro")) * 100.0, 2),
        "classes": TIER_ORDER,
        "algorithm_benchmark": [
            {
                "name": "Random Forest Classifier",
                "accuracy": round(float(accuracy_score(yt_te, yt_pred_rf)) * 100.0, 2),
                "f1_macro": round(float(f1_score(yt_te, yt_pred_rf, average="macro")) * 100.0, 2),
            },
            {
                "name": "Logistic Regression",
                "accuracy": round(float(accuracy_score(yt_te, yt_pred_lr)) * 100.0, 2),
                "f1_macro": round(float(f1_score(yt_te, yt_pred_lr, average="macro")) * 100.0, 2),
            },
        ],
    }

    # ------------------------------------------------------------------
    # 4. COMMON IELTS ERROR CLASSIFICATION (Hybrid TF-IDF + LogReg)
    # ------------------------------------------------------------------
    error_texts = [r["text"] for r in error_rows]
    error_labels = [r["category"] for r in error_rows]

    error_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=1,
                                max_features=3500,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=4500,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=3.5,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(error_pipeline, error_texts, error_labels, cv=cv, scoring="accuracy", n_jobs=1)
    cv_f1 = cross_val_score(error_pipeline, error_texts, error_labels, cv=cv, scoring="f1_macro", n_jobs=1)

    error_pipeline.fit(error_texts, error_labels)

    error_metrics = {
        "algorithm": "FeatureUnion(Word+Char TF-IDF) + LogisticRegression + Linguistic Rules",
        "cv_accuracy": round(float(np.mean(cv_acc)) * 100.0, 2),
        "cv_f1_macro": round(float(np.mean(cv_f1)) * 100.0, 2),
        "num_categories": len(set(error_labels)),
        "training_samples": len(error_texts),
    }

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_summary": {
            "student_samples": len(student_rows),
            "error_samples": len(error_rows),
            "error_categories": len(set(error_labels)),
        },
        "learning_quality": quality_metrics,
        "ielts_prediction": ielts_prediction_metrics,
        "student_classification": tier_metrics,
        "error_classification": error_metrics,
        "feature_importance": {
            "learning_quality": quality_feature_importance,
            "ielts_prediction": prediction_feature_importance,
        },
    }

    bundle = {
        "schema_version": 2,
        "prediction_features": PREDICTION_FEATURES,
        "quality_features": QUALITY_FEATURES,
        "quality_model": quality_model,
        "cluster_scaler": cluster_scaler,
        "kmeans": kmeans,
        "cluster_to_persona": cluster_to_persona,
        "skill_models": skill_models,
        "skill_residual_std": skill_residual_std,
        "overall_residual_std": round(overall_residual_std, 4),
        "tier_clf": tier_clf,
        "error_pipeline": error_pipeline,
        "error_classes": list(error_pipeline.classes_),
        "metrics": metrics,
    }

    if models_dir is not None:
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, models_dir / "ielts_models.joblib", compress=3)
            with (models_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    return bundle, metrics
