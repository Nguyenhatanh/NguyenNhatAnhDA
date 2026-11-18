"""BƯỚC 8 — Đóng gói báo cáo chuẩn BA (5 phần) -> Word (.docx) + PDF, kèm trạm kiểm định xuất tệp."""
from __future__ import annotations

import datetime as dt
import os

import matplotlib
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from .narrative import HORIZONS

SECTIONS = ["1. Tóm tắt điều hành", "2. Hiện trạng dữ liệu và quá trình làm sạch",
            "3. Khám phá dữ liệu", "4. Phân tích chuyên sâu và nguyên nhân gốc rễ",
            "5. Đề xuất giải pháp và lộ trình thực hiện"]
TITLE = "Báo cáo Phân tích Chiến lược & Tối ưu Kinh doanh"
NAVY = (0x1F, 0x4E, 0x79)


def _certainty(p: float) -> str:
    return "Rất chắc chắn" if p < 0.001 else "Chắc chắn" if p < 0.01 else "Khá chắc chắn" if p < 0.05 else "Chưa đủ bằng chứng"


# ------------------------------------------------------ mô hình nội dung chung
def build_blocks(S: dict) -> list[tuple]:
    """S = toàn bộ trạng thái phiên. Trả về danh sách block trung lập định dạng."""
    B: list[tuple] = []
    biz, story, plan = S["biz"], S["story"], S["plan"]
    prof, clean, pp, an = S["prof"], S["clean"], S.get("pp_summary", {}), S["analysis"]
    charts_by_sec: dict[int, list] = {}
    for c in S["charts"]:
        charts_by_sec.setdefault(c["section"], []).append(c)
    fig_no = [0]

    def figure(c):
        fig_no[0] += 1
        B.append(("p", story["chart_leads"].get(c["id"], "")))
        B.append(("figure", c["path"], f"Hình {fig_no[0]}. {c['caption'].replace('Hình: ', '')}"))

    # ---- 1
    es = story["executive_summary"]
    B += [("h1", SECTIONS[0]), ("lead", es.get("headline", ""))]
    if es.get("key_points"):
        B.append(("bullets", es["key_points"]))
    if es.get("top_recommendation"):
        B.append(("p_bold", "Khuyến nghị ưu tiên số 1: " + es["top_recommendation"]))
    B.append(("h2", "Bối cảnh & câu hỏi phân tích"))
    B.append(("p", f"Ngành: {biz['company_overview'].get('industry')}. Mô hình: {biz['company_overview'].get('business_model')}. "
                   f"Vấn đề cần giải quyết: {biz['objectives'].get('main_problem')}"))
    if biz["objectives"].get("key_questions"):
        B.append(("bullets", [q for q in biz["objectives"]["key_questions"] if q]))
    # ---- 2
    ov = prof["overview"]
    B += [("h1", SECTIONS[1]), ("p", S.get("data_narrative") or story.get("data_narrative") or
                                     f"Tập dữ liệu gốc gồm {ov['rows']:,} dòng và {ov['columns']} cột, điểm chất lượng sơ bộ {prof['quality_score']}/10 ({prof['risk_comment']})."),
          ("table", ["Chỉ số", "Giá trị"], [["Số dòng ban đầu", f"{ov['rows']:,}"], ["Số cột", str(ov["columns"])],
                                            ["Dòng trùng lặp", str(ov["duplicates"])], ["Tỷ lệ thiếu trung bình", f"{ov['avg_missing_pct']}%"],
                                            ["Điểm chất lượng sơ bộ", f"{prof['quality_score']}/10"], ["Số dòng sau làm sạch", f"{clean['rows_after']:,}"],
                                            ["Tỷ lệ dữ liệu bị loại", f"{clean['data_loss_percentage']}%"], ["Số ô được điền khuyết", f"{clean['imputed_cells']:,}"]])]
    if prof["issues"]:
        B += [("h2", "Các vấn đề chất lượng chính"),
              ("table", ["Cột", "Vấn đề", "Mức độ", "Chi tiết"], [[i["column"], i["type"], i["severity"], i["detail"]] for i in prof["issues"][:12]])]
    for c in charts_by_sec.get(2, []):
        figure(c)
    B += [("h2", "Nhật ký làm sạch"), ("bullets", clean["cleaning_action_log"] or ["Không cần thao tác làm sạch."])]
    if clean["logic_checks"]:
        B.append(("table", ["Luật nghiệp vụ", "Số vi phạm"], [[l["rule"], str(l["violations"]) + (f" (lỗi: {l['error']})" if l["error"] else "")] for l in clean["logic_checks"]]))
    # ---- 3
    B += [("h1", SECTIONS[2])]
    if story.get("eda_narrative"):
        B.append(("p", story["eda_narrative"]))
    for c in charts_by_sec.get(3, []):
        figure(c)
    # ---- 4
    B += [("h1", SECTIONS[3])]
    if story.get("deep_dive_narrative"):
        B.append(("p", story["deep_dive_narrative"]))
    mt = an.get("metric_tree")
    if mt:
        B += [("h2", f"Phân rã chỉ số {an.get('kpi')}"),
              ("table", ["Thành phần", f"Kỳ trước ({mt['period_a']})", f"Kỳ sau ({mt['period_b']})", "Thay đổi"],
               [[f"Tổng {an.get('kpi')}", f"{mt['kpi_a']:,.0f}", f"{mt['kpi_b']:,.0f}", f"{mt['total_change_pct']:+.1f}%"],
                ["Số giao dịch", f"{mt['transactions_a']:,}", f"{mt['transactions_b']:,}", f"{mt['components']['Số giao dịch']:+.1f}%"],
                ["Giá trị TB/giao dịch", f"{mt['aov_a']:,.0f}", f"{mt['aov_b']:,.0f}", f"{mt['components']['Giá trị TB/giao dịch']:+.1f}%"]])]
    for c in charts_by_sec.get(4, []):
        figure(c)
    for sd in an.get("segment_drill", [])[:2]:
        B += [("h2", f"Nhóm đóng góp nhiều nhất vào thay đổi theo {sd['segment']}"),
              ("table", ["Nhóm", "Kỳ trước", "Kỳ sau", "Chênh lệch", "% tổng chênh lệch"],
               [[t["group"], f"{t['A']:,.0f}", f"{t['B']:,.0f}", f"{t['delta']:+,.0f}", f"{t['share_of_total_delta_pct']}%" if t["share_of_total_delta_pct"] is not None else "—"]
                for t in sd["top_contributors"]])]
    sig = [t for t in an.get("tests", []) if t["significant"]][:10]
    if sig:
        B += [("h2", "Các yếu tố có ảnh hưởng được kiểm chứng"),
              ("table", ["Yếu tố", "Mức độ chắc chắn", "Ghi chú"],
               [[t["feature"], _certainty(t["p_adj"]), (f"Nhóm tốt nhất cao hơn nhóm thấp nhất {t['best_vs_worst_pct']}%" if t.get("best_vs_worst_pct") else
                                                         f"Tương quan {'thuận' if (t.get('effect') or 0) > 0 else 'nghịch'}" if t["test"] == "Spearman" else "Khác biệt rõ giữa các nhóm")] for t in sig])]
    if an.get("insights"):
        B += [("h2", "Nhận định then chốt"), ("bullets", [f"{i['insight']} — Bằng chứng: {i.get('evidence', '')}" for i in an["insights"]])]
    # ---- 5
    B += [("h1", SECTIONS[4])]
    if plan.get("key_findings"):
        B += [("h2", "Kết luận trọng tâm"), ("bullets", plan["key_findings"])]
    for h, label in HORIZONS.items():
        acts = [a for a in plan.get("actions", []) if a.get("horizon") == h]
        if acts:
            B += [("h2", label), ("table", ["Hành động", "Lý do", "Người phụ trách", "Thời hạn", "Cách làm / Phạm vi", "Chỉ tiêu", "Tác động / Nguồn lực"],
                                  [[a.get("what", ""), a.get("why", ""), a.get("who", ""), a.get("when", ""), f"{a.get('how', '')} — {a.get('where', '')}",
                                    a.get("kpi", ""), f"{a.get('impact', '')}/{a.get('effort', '')}"] for a in acts])]
    if plan.get("risks"):
        B += [("h2", "Rủi ro & phương án dự phòng"), ("table", ["Rủi ro", "Phương án dự phòng"], [[r.get("risk", ""), r.get("mitigation", "")] for r in plan["risks"]])]
    if plan.get("limitations"):
        B += [("h2", "Hạn chế của phân tích"), ("bullets", plan["limitations"])]
    # ---- Phụ lục kỹ thuật (được phép dùng thuật ngữ)
    B.append(("h1", "Phụ lục: chi tiết phương pháp"))
    m = an.get("metrics")
    if m:
        MV = {"r2": "Mức giải thích biến động", "rmse": "Sai số bình phương trung bình", "mae": "Sai số tuyệt đối trung bình", "accuracy": "Tỷ lệ dự đoán đúng", "f1_macro": "Điểm cân bằng chính xác", "roc_auc": "Khả năng phân biệt nhãn"}
        rows = [[MV.get(k, k), str(m["train"].get(k, "")), str(m["test"].get(k, "")), str(m["baseline_test"].get(k, ""))] for k in m["test"]]
        B += [("p", f"Mô hình rừng ngẫu nhiên — biến mục tiêu: {an.get('outcome')}"), ("table", ["Chỉ số", "Tập học", "Tập kiểm tra", "Mô hình đối chứng"], rows)]
    if pp.get("preprocessing_log"):
        B += [("h2", "Tiền xử lý"), ("bullets", pp["preprocessing_log"])]
    if an.get("tests"):
        B += [("h2", "Kiểm định thống kê (đã hiệu chỉnh cho nhiều phép thử)"),
              ("table", ["Biến", "Phép kiểm định", "Giá trị thống kê", "Xác suất ngẫu nhiên", "Mức ảnh hưởng"],
               [[t["feature"], {"Kruskal-Wallis": "So sánh nhiều nhóm", "Spearman": "Tương quan thứ hạng", "Chi-square": "Kiểm định độc lập"}.get(t["test"], t["test"]), str(t["stat"]), f"{t['p_adj']:.4g}", f"{t.get('effect') if t.get('effect') is not None else '—'}"] for t in an["tests"][:15]])]
    return B


# ---------------------------------------------------------------------- DOCX
def _docx_font(doc):
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Arial", Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
    for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Title", 22)):
        s = doc.styles[name]
        s.font.name, s.font.size, s.font.color.rgb = "Arial", Pt(size), RGBColor(*NAVY)
        s.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")


def _shade(cell, hex_fill):
    from docx.oxml import OxmlElement

    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def to_docx(blocks: list[tuple], path: str, meta: dict) -> str:
    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.9)
    _docx_font(doc)
    doc.add_paragraph(TITLE, style="Title")
    p = doc.add_paragraph(f"{meta.get('industry', '')}  |  Ngày lập: {dt.date.today():%d/%m/%Y}  |  Lập bởi: Tác tử phân tích dữ liệu kinh doanh")
    p.runs[0].font.color.rgb = RGBColor(0x59, 0x59, 0x59)
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            doc.add_heading(b[1], level=1)
        elif kind == "h2":
            doc.add_heading(b[1], level=2)
        elif kind == "lead":
            r = doc.add_paragraph().add_run(b[1])
            r.bold, r.font.size, r.font.color.rgb = True, Pt(12), RGBColor(*NAVY)
        elif kind == "p" and b[1]:
            doc.add_paragraph(b[1])
        elif kind == "p_bold":
            doc.add_paragraph().add_run(b[1]).bold = True
        elif kind == "bullets":
            for item in b[1]:
                doc.add_paragraph(str(item), style="List Bullet")
        elif kind == "table":
            headers, rows = b[1], b[2]
            t = doc.add_table(rows=1, cols=len(headers))
            t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
            for i, h in enumerate(headers):
                cell = t.rows[0].cells[i]
                cell.text = ""
                run = cell.paragraphs[0].add_run(h)
                run.bold, run.font.color.rgb, run.font.size = True, RGBColor(0xFF, 0xFF, 0xFF), Pt(9)
                _shade(cell, "1F4E79")
            for row in rows:
                cells = t.add_row().cells
                for i, v in enumerate(row):
                    cells[i].text = str(v)
                    for par in cells[i].paragraphs:
                        for r in par.runs:
                            r.font.size = Pt(9)
            doc.add_paragraph()
        elif kind == "figure":
            doc.add_picture(b[1], width=Inches(6.0))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(b[2])
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic, cap.runs[0].font.size = True, Pt(9)
    doc.save(path)
    return path


# ----------------------------------------------------------------------- PDF
def _register_fonts():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    if "DejaVu" in pdfmetrics.getRegisteredFontNames():
        return
    fdir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")  # DejaVu đi kèm matplotlib => hỗ trợ tiếng Việt
    pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(fdir, "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(fdir, "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVu-Oblique", os.path.join(fdir, "DejaVuSans-Oblique.ttf")))
    from reportlab.lib.fonts import addMapping

    addMapping("DejaVu", 0, 0, "DejaVu")
    addMapping("DejaVu", 1, 0, "DejaVu-Bold")
    addMapping("DejaVu", 0, 1, "DejaVu-Oblique")


def to_pdf(blocks: list[tuple], path: str, meta: dict) -> str:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import Image, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    _register_fonts()
    navy = colors.HexColor("#1F4E79")
    base = ParagraphStyle("b", fontName="DejaVu", fontSize=9.5, leading=13.5, spaceAfter=5)
    st = {
        "title": ParagraphStyle("t", parent=base, fontName="DejaVu-Bold", fontSize=19, leading=24, textColor=navy, spaceAfter=6),
        "h1": ParagraphStyle("h1", parent=base, fontName="DejaVu-Bold", fontSize=13.5, leading=18, textColor=navy, spaceBefore=12, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base, fontName="DejaVu-Bold", fontSize=11, leading=15, textColor=navy, spaceBefore=8, spaceAfter=4),
        "lead": ParagraphStyle("lead", parent=base, fontName="DejaVu-Bold", fontSize=11, leading=15, textColor=navy),
        "cap": ParagraphStyle("cap", parent=base, fontName="DejaVu-Oblique", fontSize=8, alignment=TA_CENTER, textColor=colors.HexColor("#555555")),
        "cell": ParagraphStyle("cell", parent=base, fontSize=7.5, leading=10, spaceAfter=0),
        "cellh": ParagraphStyle("cellh", parent=base, fontName="DejaVu-Bold", fontSize=7.5, leading=10, textColor=colors.white, spaceAfter=0),
    }
    E = lambda s: escape(str(s))  # noqa: E731
    W = A4[0] - 3.6 * cm
    story = [Paragraph(E(TITLE), st["title"]),
             Paragraph(E(f"{meta.get('industry', '')} | Ngày lập: {dt.date.today():%d/%m/%Y} | Lập bởi: Tác tử phân tích dữ liệu kinh doanh"), base), Spacer(1, 8)]
    for b in blocks:
        k = b[0]
        if k in ("h1", "h2", "lead"):
            story.append(Paragraph(E(b[1]), st[k]))
        elif k == "p" and b[1]:
            story.append(Paragraph(E(b[1]), base))
        elif k == "p_bold":
            story.append(Paragraph(f"<b>{E(b[1])}</b>", base))
        elif k == "bullets":
            story.append(ListFlowable([ListItem(Paragraph(E(i), base), leftIndent=10) for i in b[1]], bulletType="bullet", start="•", leftIndent=12))
        elif k == "table":
            headers, rows = b[1], b[2]
            data = [[Paragraph(E(h), st["cellh"]) for h in headers]] + [[Paragraph(E(v), st["cell"]) for v in r] for r in rows]
            t = Table(data, colWidths=[W / len(headers)] * len(headers), repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), navy), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BFBFBF")),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FA")]),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
            story += [t, Spacer(1, 8)]
        elif k == "figure":
            iw, ih = ImageReader(b[1]).getSize()
            w = min(W, 16 * cm)
            h = w * ih / iw
            if h > 15 * cm:
                h, w = 15 * cm, 15 * cm * iw / ih
            story += [Image(b[1], width=w, height=h), Paragraph(E(b[2]), st["cap"])]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.8 * cm, 1.1 * cm, TITLE)
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Trang {doc_.page}")
        canvas.restoreState()

    SimpleDocTemplate(path, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.8 * cm,
                      title=TITLE, author="Tác tử phân tích dữ liệu kinh doanh").build(story, onFirstPage=footer, onLaterPages=footer)
    return path


# -------------------------------------------------------- trạm kiểm soát xuất tệp
def build_and_validate(S: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    blocks = build_blocks(S)
    reasons, log = [], []
    # 1) Visual-Text flow: mọi hình đều có đoạn dẫn ngay phía trước + caption
    for i, b in enumerate(blocks):
        if b[0] == "figure":
            lead = blocks[i - 1] if i else None
            if not (lead and lead[0] == "p" and str(lead[1]).strip()):
                reasons.append(f"Hình '{b[2][:40]}' thiếu đoạn dẫn")
            if not b[2].strip():
                reasons.append("Có hình thiếu chú thích")
    n_fig = sum(b[0] == "figure" for b in blocks)
    log.append(f"Gắn kết hình và lời: {n_fig} hình, đều có đoạn dẫn và chú thích" if not reasons else "Gắn kết hình và lời: có hình thiếu diễn giải")
    # 2) Structural completeness
    h1 = [b[1] for b in blocks if b[0] == "h1"]
    miss = [s for s in SECTIONS if s not in h1]
    if miss:
        reasons.append(f"Thiếu phần: {miss}")
    log.append(f"Cấu trúc: đủ {5 - len(miss)}/5 phần trọng tâm")
    # 3) Export readability: render thật rồi mở lại để kiểm tra
    meta = {"industry": S["biz"]["company_overview"].get("industry", "")}
    docx_path = to_docx(blocks, os.path.join(out_dir, f"{TITLE}.docx"), meta)
    pdf_path = to_pdf(blocks, os.path.join(out_dir, f"{TITLE}.pdf"), meta)
    try:
        d = Document(docx_path)
        n_img = len(d.inline_shapes)
        if n_img != n_fig:
            reasons.append(f"DOCX có {n_img}/{n_fig} hình")
        from pypdf import PdfReader

        pr = PdfReader(pdf_path)
        txt = pr.pages[0].extract_text() or ""
        if "Tóm tắt" not in txt and "Báo cáo" not in txt:
            reasons.append("PDF không trích xuất được tiếng Việt (lỗi font)")
        log.append(f"Xuất tệp: Word {os.path.getsize(docx_path)//1024} KB ({n_img} hình); PDF {len(pr.pages)} trang, phông tiếng Việt hiển thị đúng")
    except Exception as exc:
        reasons.append(f"Không mở lại được tệp xuất: {exc}")
    return {"status": "REQUIRE_HUMAN_REVIEW" if reasons else "PASS", "report_readiness_summary": log, "reasons": reasons,
            "actionable_question_for_human": "Bạn có muốn bổ sung nội dung trước khi xuất bản không?" if reasons else "",
            "docx": docx_path, "pdf": pdf_path}
