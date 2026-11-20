"""CLI script to generate datasets, train all ML models, and save artifacts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.trainer import train_all_models  # noqa: E402


def main() -> None:
    data_dir = ROOT_DIR / "data"
    models_dir = ROOT_DIR / "models"
    print("Đang tạo dữ liệu mẫu và huấn luyện 4 mô hình Học máy IELTS...")
    _, metrics = train_all_models(data_dir=data_dir, models_dir=models_dir, regenerate_data=True)
    print("\nHuấn luyện hoàn tất! Chỉ số đánh giá mô hình:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nĐã lưu mô hình tại: {models_dir / 'ielts_models.joblib'}")


if __name__ == "__main__":
    main()
