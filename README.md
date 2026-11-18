# AI Business Analyst Agent (localhost)

Agent phân tích kinh doanh 8 bước có trạm kiểm soát Human-in-the-loop. Giao diện Streamlit nền trắng. **Gemini chỉ dùng để suy luận và diễn giải; mọi con số đều do Pandas, Scipy và Scikit-learn tính.**

## Chạy nhanh (Windows)
1. Cài Python 3.10 trở lên.
2. Nhấp đúp `run.bat`. Lần đầu script sẽ tạo `.venv`, cài thư viện và tạo tệp `.env`.
3. Mở `.env`, điền `GEMINI_API_KEY=...`, rồi chạy lại `run.bat`.
4. Trình duyệt mở `http://localhost:8501`.

Linux/macOS: `./run.sh`. Muốn chạy thủ công: `pip install -r requirements.txt && streamlit run app.py`.

Thử ngay với `sample_data/context_example.txt` (dán vào ô bối cảnh) và `sample_data/sample_sales.csv`. Tệp CSV mẫu được cài sẵn lỗi: trùng lặp, số âm, ngoại lệ, tiền lưu dạng chữ "1.234.000 ₫", ngày giao trước ngày đặt, ô trống.

## Kiến trúc
```
app.py                 UI + state machine (dừng ở trạm kiểm soát, PASS thì tự chạy tiếp)
core/pipeline.py       Điều phối 8 bước, JSON logs (short-term memory), run_until_checkpoint
core/llm.py            Gemini client, retry/backoff, JSON parser, vòng lặp Generator ⇄ Critic
core/context.py        B1  Trích PDF/DOCX/PPTX → Hồ sơ bối cảnh (không suy diễn)
core/profiling.py      B2  Suy luận kiểu, missing, duplicates, IQR, giá trị âm, ngày tương lai, logic → điểm 1–10
core/cleaning.py       B3  Kế hoạch làm sạch (LLM đề xuất, code kiểm tra) → thực thi → loss% / logic / shift
core/preprocess.py     B4  EDA + FE + split → Imputer/Scaler/One-Hot fit trên Train → SMOTE → leakage/dim check
core/modeling.py       B5  Kruskal/Spearman/Chi² + hiệu chỉnh BH, RandomForest vs baseline, permutation importance,
                           Metric Tree (KPI = Số giao dịch × AOV), drill-down theo phân khúc (MECE)
core/narrative.py      B6  Executive summary theo Kim tự tháp, kiểm tra thuật ngữ kỹ thuật · B7 SMART/5W1H + Impact/Effort
core/report.py         B8  DOCX + PDF (font DejaVu hỗ trợ tiếng Việt), mở lại tệp để kiểm định
core/charts.py         Biểu đồ nền trắng: trục bar bắt đầu từ 0, ≤10 danh mục, không dùng pie
```

## Ngưỡng kiểm định (sửa ở đầu từng module)
| Trạm | Điều kiện REQUIRE_HUMAN_REVIEW |
|---|---|
| B3 | Loại > 10% số dòng · có vi phạm logic nghiệp vụ · mean/median lệch > 15% sau điền khuyết |
| B4 | Biến có \|r\| ≥ 0.98 với mục tiêu · scaler không khớp Train · số cột/số dòng > 0.10 · SMOTE lệch > 0.5 SD |
| B5 | Không yếu tố nào đạt p_adj < 0.05 · chênh Train−Test F1 > 0.10 / R² > 0.15 · F1 < 0.60 / R² < 0.30 |
| B6 | Bar không bắt đầu từ 0 / pie / > 10 danh mục · headline không có số · còn thuật ngữ kỹ thuật |
| B7 | Hành động thiếu Who/When/KPI có số · thiếu một horizon · < 2 rủi ro có phương án dự phòng |
| B8 | Hình thiếu đoạn dẫn hoặc caption · thiếu 1 trong 5 phần · DOCX/PDF không mở lại được |

## Bảo mật
- API key chỉ đọc từ `.env` hoặc ô nhập ở sidebar, không ghi vào mã nguồn. `.env` đã có trong `.gitignore`.
- Không có key thì app chạy **chế độ offline** (luật xác định), tiện để kiểm thử.
