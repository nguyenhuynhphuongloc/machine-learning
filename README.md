# IELTS AI Learning Analytics (Python Machine Learning + FastAPI + Vercel)

Hệ thống Học máy toàn diện bằng Python dự đoán **Band IELTS thực tế** dựa trên các chỉ số học tập cốt lõi của học viên, đánh giá chất lượng học tập, phân loại học sinh yếu/mạnh và phân loại lỗi sai phổ biến — sẵn sàng triển khai (deploy) lên **Vercel**.

## 8 Trường dữ liệu đầu vào cốt lõi

| Trường | Key API | Ý nghĩa |
|---|---|---|
| **Điểm đầu vào** | `entry_score` | Điểm Band IELTS kiểm tra đầu vào (`1.0 – 9.0`) |
| **Số buổi đặt WOW** | `wow_sessions` | Số buổi học kèm / đặt lịch WOW (`0 – 60` buổi) |
| **Phần trăm chuyên cần** | `attendance_rate` | Tỷ lệ đi học chuyên cần (`0 – 100%`) |
| **Phần trăm làm BTVN** | `homework_completion_rate` | Tỷ lệ hoàn thành bài tập về nhà (`0 – 100%`) |
| **Điểm % BTVN Listening** | `hw_listening_pct` | Phần trăm điểm BTVN kỹ năng Nghe (`0 – 100%`) |
| **Điểm % BTVN Reading** | `hw_reading_pct` | Phần trăm điểm BTVN kỹ năng Đọc (`0 – 100%`) |
| **Điểm % BTVN Writing** | `hw_writing_pct` | Phần trăm điểm BTVN kỹ năng Viết (`0 – 100%`) |
| **Điểm % BTVN Speaking** | `hw_speaking_pct` | Phần trăm điểm BTVN kỹ năng Nói (`0 – 100%`) |

## 4 Tính năng Học máy cốt lõi

1. **Dự đoán Band IELTS thực tế (`/api/predict-ielts`)**
   - **Thuật toán:** 4 mô hình `GradientBoostingRegressor` riêng biệt cho **Listening, Reading, Writing, Speaking** (so sánh benchmark với `RandomForestRegressor` và `Ridge`), làm tròn chuẩn 0.5 chính thức của IELTS.
   - Trả về Band Overall thực tế, Band 4 kỹ năng, khoảng tin cậy 90%, xác suất đạt band mục tiêu và mô phỏng **What-If** khi tăng số buổi đặt WOW & % BTVN.

2. **Đánh giá chất lượng học tập (`/api/evaluate-quality`)**
   - **Thuật toán:** `GradientBoostingRegressor` (chấm điểm 0–100) + `KMeans (4 cụm)` phân nhóm hành vi học tập.
   - Hiển thị biểu đồ Radar 7 chiều, điểm mạnh, cảnh báo sớm và khuyến nghị hành động.

3. **Phân loại học sinh Yếu / Mạnh (`/api/classify-student` & `/api/batch-analyze`)**
   - **Thuật toán:** `RandomForestClassifier` (benchmark với `LogisticRegression`) phân loại 4 nhóm: **Yếu**, **Trung bình**, **Khá**, **Mạnh** + chẩn đoán kỹ năng yếu nhất và phân tích hàng loạt cả lớp.

4. **Phân loại lỗi sai phổ biến NLP (`/api/classify-errors`)**
   - **Thuật toán:** `FeatureUnion(Word TF-IDF + Character TF-IDF)` + `LogisticRegression` kết hợp hệ luật ngôn ngữ (Linguistic Rules) cho 8 nhóm lỗi IELTS phổ biến của người học Việt Nam.

---

## Chạy thử trên máy cá nhân (Local)

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt
pip install uvicorn

# 2. (Tùy chọn) Huấn luyện lại mô hình từ dữ liệu mới
python scripts/train.py

# 3. Khởi động server FastAPI + Giao diện Web
uvicorn api.index:app --reload --port 8000
```

Mở trình duyệt tại:
- Giao diện Dashboard: `http://localhost:8000`
- Tài liệu Swagger API: `http://localhost:8000/docs`

---

## Hướng dẫn Deploy lên Vercel

Dự án đã được cấu hình sẵn `vercel.json`, `.vercelignore` và tối ưu `requirements.txt` (không dùng `pandas`/`torch` nặng để đảm bảo gọn nhẹ dưới giới hạn 250MB của Vercel Serverless Functions).

### Cách 1: Deploy qua GitHub (Khuyên dùng)
1. Push toàn bộ thư mục dự án lên GitHub.
2. Đăng nhập [vercel.com](https://vercel.com) → chọn **Add New Project** → Import repository.
3. Giữ nguyên cấu hình mặc định (Vercel tự nhận diện `vercel.json`, `api/index.py`, `requirements.txt`, `public/`).
4. Nhấn **Deploy**.

### Cách 2: Deploy trực tiếp bằng Vercel CLI
```bash
npm i -g vercel
vercel login
vercel --prod
```
