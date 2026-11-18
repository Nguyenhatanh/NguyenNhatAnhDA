"""BƯỚC 1 — Tiếp nhận & thấu hiểu bối cảnh doanh nghiệp."""
from __future__ import annotations

import io

from .llm import LLMClient

NA = "Chưa cung cấp"


def extract_text(name: str, data: bytes, max_chars: int = 30000) -> str:
    """Trích văn bản từ PDF / DOCX / PPTX / TXT / MD."""
    ext = name.lower().rsplit(".", 1)[-1]
    text = ""
    try:
        if ext == "pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        elif ext == "docx":
            import docx

            d = docx.Document(io.BytesIO(data))
            parts = [p.text for p in d.paragraphs if p.text.strip()]
            for t in d.tables:  # từ điển dữ liệu thường nằm trong bảng
                for row in t.rows:
                    parts.append(" | ".join(c.text.strip() for c in row.cells))
            text = "\n".join(parts)
        elif ext == "pptx":
            from pptx import Presentation

            prs = Presentation(io.BytesIO(data))
            parts = []
            for i, slide in enumerate(prs.slides, 1):
                parts.append(f"--- Slide {i} ---")
                for shp in slide.shapes:
                    if shp.has_text_frame:
                        parts.append(shp.text_frame.text)
                    if getattr(shp, "has_table", False) and shp.has_table:
                        for row in shp.table.rows:
                            parts.append(" | ".join(c.text for c in row.cells))
            text = "\n".join(parts)
        else:
            text = data.decode("utf-8", errors="ignore")
    except Exception as exc:
        text = f"[Lỗi đọc tệp {name}: {exc}]"
    return text[:max_chars]


def _fallback_profile(user_text: str) -> dict:
    """Hồ sơ tối thiểu khi không có LLM: không suy diễn, chỉ ghi lại nguyên văn."""
    return {
        "company_overview": {"industry": NA, "business_model": NA, "scale_operations": NA},
        "objectives": {"main_problem": user_text.strip()[:500] or NA, "key_questions": []},
        "data_constraints": {"assumptions": [], "data_dictionary": [], "time_scope": NA, "region_scope": NA},
        "kpi_hint": NA,
        "missing_info": ["Chế độ offline: chưa phân tích ngữ nghĩa bằng LLM — vui lòng tự kiểm tra hồ sơ."],
    }


def build_profile(llm: LLMClient, user_text: str, docs_text: str, previous: dict | None = None, feedback: str | None = None) -> dict:
    prompt = (
        "NHIỆM VỤ (Bước 1 - Business Profiling): Trích xuất Hồ sơ Bối cảnh Doanh nghiệp từ thông tin người dùng.\n"
        "QUY TẮC: Không tự suy diễn. Trường nào không có trong nguồn => ghi 'Chưa cung cấp' và liệt kê vào missing_info.\n\n"
        f"MÔ TẢ CỦA NGƯỜI DÙNG:\n{user_text or '(trống)'}\n\n"
        f"TÀI LIỆU NỘI BỘ ĐÍNH KÈM:\n{docs_text or '(không có)'}\n\n"
    )
    if previous and feedback:
        prompt += f"HỒ SƠ PHIÊN BẢN TRƯỚC:\n{previous}\n\nYÊU CẦU CHỈNH SỬA CỦA NGƯỜI DÙNG (ưu tiên cao nhất):\n{feedback}\n\n"
    prompt += (
        "Trả về JSON đúng cấu trúc:\n"
        '{"company_overview": {"industry": "", "business_model": "", "scale_operations": ""},\n'
        ' "objectives": {"main_problem": "", "key_questions": [""]},\n'
        ' "data_constraints": {"assumptions": [""], "data_dictionary": [{"column": "", "meaning": ""}], "time_scope": "", "region_scope": ""},\n'
        ' "kpi_hint": "tên chỉ số/biến mục tiêu chính nếu người dùng nêu, ví dụ doanh thu, churn",\n'
        ' "missing_info": ["thông tin quan trọng còn thiếu"]}'
    )
    fb = previous if previous else _fallback_profile(user_text + ("\n" + feedback if feedback else ""))
    prof = llm.json(prompt, fallback=fb)
    # Chuẩn hoá khung để UI không vỡ khi LLM trả thiếu trường
    base = _fallback_profile(user_text)
    for k, v in base.items():
        if k not in prof or prof[k] in (None, ""):
            prof[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                prof[k].setdefault(kk, vv)
    return prof


def profile_markdown(p: dict) -> str:
    co, ob, dc = p["company_overview"], p["objectives"], p["data_constraints"]
    lines = [
        "#### 🏢 Tổng quan Doanh nghiệp",
        f"- **Ngành nghề:** {co.get('industry', NA)}",
        f"- **Mô hình kinh doanh:** {co.get('business_model', NA)}",
        f"- **Quy mô / đặc thù vận hành:** {co.get('scale_operations', NA)}",
        "#### 🎯 Mục tiêu Phân tích & Câu hỏi Trọng tâm",
        f"- **Vấn đề chính:** {ob.get('main_problem', NA)}",
        f"- **Chỉ số mục tiêu (KPI):** {p.get('kpi_hint', NA)}",
    ]
    for q in ob.get("key_questions") or []:
        if q:
            lines.append(f"  - ❓ {q}")
    lines += [
        "#### 📊 Dữ liệu & Ràng buộc Đi kèm",
        f"- **Phạm vi thời gian:** {dc.get('time_scope', NA)}",
        f"- **Phạm vi khu vực:** {dc.get('region_scope', NA)}",
    ]
    for a in dc.get("assumptions") or []:
        if a:
            lines.append(f"- Giả định: {a}")
    dd = [d for d in (dc.get("data_dictionary") or []) if isinstance(d, dict) and d.get("column")]
    if dd:
        lines.append("- **Từ điển dữ liệu:**")
        lines += [f"  - `{d['column']}`: {d.get('meaning', '')}" for d in dd]
    miss = [m for m in (p.get("missing_info") or []) if m]
    if miss:
        lines.append("#### ⚠️ Thông tin còn thiếu")
        lines += [f"- {m}" for m in miss]
    return "\n".join(lines)
