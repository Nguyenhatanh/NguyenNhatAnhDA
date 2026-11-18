"""
AI Business Analyst Agent — giao diện Streamlit (localhost, nền trắng).
Chạy:  streamlit run app.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from core import pipeline as P
from core.context import extract_text
from core.llm import LLMClient
from core.narrative import HORIZONS
from core.dashboard import interactive
from core import dashboard as D

st.set_page_config(page_title="Tác tử Phân tích Dữ liệu Kinh doanh", page_icon="📊", layout="wide")
st.markdown("""
<style>
  h1, h2, h3, h4, h5, h6, .stButton button p, .stDownloadButton button p, [data-testid="stMetricValue"] div {font-weight:700 !important;}
  [data-testid="stMetricValue"] div {color:#1F4E79; font-weight:800 !important;}
  .stApp, [data-testid="stHeader"] {background:#FFFFFF;}
  [data-testid="stSidebar"] {background:#F7F9FC; border-right:1px solid #E5E9F0;}
  h1, h2, h3, h4 {color:#1F4E79;}
  .cp {border:1px solid #F0B429; background:#FFFBEA; border-radius:8px; padding:14px 18px; margin:10px 0;}
  .ok {border:1px solid #2E7D32; background:#F1F8F1; border-radius:8px; padding:10px 16px; margin:8px 0;}
  .export {border:2px solid #1F4E79; border-radius:10px; padding:18px 22px; background:#FFFFFF;}
  .stepbox {font-size:0.92rem; line-height:1.9;}
  .kpi {border:1px solid #E5E9F0; border-radius:12px; padding:16px 18px; background:#FFFFFF;}
  .kpi .lb {font-size:0.82rem; color:#5A6878; font-weight:600; text-transform:uppercase; letter-spacing:.5px;}
  .kpi .v {font-size:1.7rem; font-weight:800; color:#16263A; margin:4px 0;}
  .kpi .d {font-size:0.9rem; font-weight:700;}
  .up {color:#2E7D32;} .down {color:#C0392B;} .flat {color:#5A6878;}
  .al {border-radius:10px; padding:10px 14px; margin:6px 0; font-weight:500;}
  .al.do {background:#FDECEA; border-left:6px solid #C0392B;}
  .al.vang {background:#FFF6E0; border-left:6px solid #B7791F;}
  .al.xanh {background:#EAF5EA; border-left:6px solid #2E7D32;}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("S", P.new_state())
ss.setdefault("stage", "s1_input")   # s1_input | s1_review | s2_upload | s2_review | review | done
ss.setdefault("stop", None)
S = ss.S
CP_MSG = {
    3: "⚠️ <b>TRẠM KIỂM SOÁT BƯỚC 3 · LÀM SẠCH DỮ LIỆU</b>: quá trình làm sạch đã hoàn tất nhưng có điểm cần anh/chị duyệt.",
    4: "⚠️ <b>TRẠM KIỂM SOÁT BƯỚC 4 · TIỀN XỬ LÝ</b>: phát hiện rủi ro kỹ thuật cần xem xét.",
    5: "⚠️ <b>TRẠM KIỂM SOÁT BƯỚC 5 · KIỂM ĐỊNH KẾT QUẢ</b>: kết quả thống kê hoặc mô hình chưa đạt độ tin cậy yêu cầu.",
    6: "⚠️ <b>TRẠM KIỂM SOÁT BƯỚC 6 · BÁO CÁO ĐIỀU HÀNH</b>: bản báo cáo còn điểm chưa đạt chuẩn trình bày.",
    7: "⚠️ <b>TRẠM KIỂM SOÁT BƯỚC 7 · KẾT LUẬN VÀ ĐỀ XUẤT</b>: kế hoạch hành động còn thiếu thông tin bắt buộc.",
    8: "⚠️ <b>TRẠM KIỂM SOÁT BƯỚC 8 · XUẤT TỆP</b>: bản thảo chưa đạt chuẩn xuất bản.",
}
CP_ASK = {
    3: "👉 Phê duyệt để chuyển sang bước khám phá dữ liệu, hoặc ghi chỉ dẫn điều chỉnh phương pháp làm sạch (ví dụ: 'xoá các dòng vi phạm logic', 'giữ nguyên cột X')?",
    4: "👉 Phê duyệt để tiếp tục, hoặc ghi chỉ dẫn: rút gọn biến, đổi biến mục tiêu, loại biến nghi rò rỉ.",
    5: "👉 Chấp nhận kết quả kèm cảnh báo, hoặc ghi chỉ dẫn đổi biến mục tiêu, loại bớt biến.",
    6: "👉 Phê duyệt để sang bước đề xuất, hoặc ghi chỉ dẫn chỉnh biểu đồ, văn phong báo cáo.",
    7: "👉 Phê duyệt kế hoạch, hoặc ghi chỉ dẫn bổ sung người phụ trách, thời hạn, chỉ tiêu và nhóm việc làm ngay.",
    8: "👉 Bạn muốn bổ sung nội dung, hay xuất báo cáo với hiện trạng?",
}


def friendly_error(raw: str) -> str:
    """Ẩn chuỗi lỗi kỹ thuật, chỉ hiển thị hướng xử lý."""
    r = raw.lower()
    if "api key" in r or "api_key" in r or "unauthenticated" in r or "access_token" in r or "permission" in r:
        return "🔑 Khóa truy cập Gemini chưa hợp lệ. Vui lòng tạo khóa mới tại aistudio.google.com/apikey rồi dán lại vào ô bên trên. Trong lúc chờ, hệ thống vẫn chạy bằng quy tắc có sẵn."
    if "quota" in r or "429" in r or "resource_exhausted" in r:
        return "⏳ Khóa truy cập đã dùng hết hạn mức tạm thời. Vui lòng thử lại sau ít phút."
    if "timeout" in r or "connect" in r or "network" in r or "unavailable" in r or "503" in r:
        return "🌐 Không kết nối được tới máy chủ Gemini. Vui lòng kiểm tra mạng Internet."
    if "not found" in r or "404" in r:
        return "🧩 Phiên bản mô hình không tồn tại. Vui lòng kiểm tra lại tên mô hình, ví dụ gemini-2.5-flash."
    return "⚠️ Dịch vụ Gemini tạm thời chưa phản hồi. Hệ thống vẫn tiếp tục bằng quy tắc có sẵn."


# ================================================================== SIDEBAR
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")
    api_key = st.text_input("Khóa truy cập Gemini", value=os.getenv("GEMINI_API_KEY", ""), type="password",
                            help="Lưu trong tệp .env (GEMINI_API_KEY=...) để không phải nhập lại.")
    model = st.text_input("Phiên bản mô hình", value=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    key = (api_key, model)
    if ss.get("llm_key") != key:
        ss.llm, ss.llm_key = LLMClient(api_key, model), key
    llm: LLMClient = ss.llm
    if llm.mock:
        st.warning("Đang chạy chế độ ngoại tuyến: chưa có khóa truy cập, hệ thống dùng quy tắc có sẵn thay cho mô hình ngôn ngữ.")
    else:
        st.success(f"Đã cấu hình Gemini · {llm.model}")
    if not llm.mock and st.button("🔌 Kiểm tra kết nối", width="stretch"):
        with st.spinner("Đang gọi thử Gemini…"):
            ok = llm._call("Trả lời đúng 1 từ: OK", None, json_mode=False, temperature=0, retries=0)
        st.success("Kết nối thành công ✔") if ok else None
    if llm.last_error:
        st.error(friendly_error(llm.last_error))

    st.markdown("### 🧭 Tiến trình 8 bước")
    icon = {"PASS": "✅", "REQUIRE_HUMAN_REVIEW": "⚠️"}
    rows = []
    for n, name in P.STEP_NAMES.items():
        stt = S["statuses"].get(n)
        confirmed = (n == 1 and ss.stage not in ("s1_input", "s1_review")) or (n == 2 and ss.stage not in ("s1_input", "s1_review", "s2_upload", "s2_review"))
        ic = "✅" if confirmed else icon.get(stt, "⬜")
        if ss.stage == "review" and ss.stop == n:
            ic = "🛑"
        rows.append(f"{ic} <b>{n}.</b> {name}")
    st.markdown("<div class='stepbox'>" + "<br>".join(rows) + "</div>", unsafe_allow_html=True)
    if S["logs"]:
        st.download_button("⬇️ Tải nhật ký xử lý (JSON)", json.dumps(S["logs"], ensure_ascii=False, default=str, indent=2),
                           "ba_agent_logs.json", "application/json", width="stretch")
    if st.button("🔄 Phiên làm việc mới", width="stretch"):
        for k in ("S", "stage", "stop"):
            ss.pop(k, None)
        st.rerun()


# ================================================================== HELPERS
def run_from(start: int):
    """Chạy ĐÚNG MỘT bước rồi dừng; người dùng phải bấm xác nhận mới sang bước kế tiếp."""
    with st.status(f"Bước {start}: {P.STEP_NAMES[start]}…", expanded=True) as box:
        try:
            P.STEPS[start](S, llm, S["feedback"].pop(f"retry_{start}", None))
        except Exception as exc:
            box.update(label="Quá trình xử lý bị gián đoạn", state="error")
            st.error("Hệ thống chưa hoàn tất bước này. Vui lòng kiểm tra lại tệp dữ liệu hoặc ghi chỉ dẫn điều chỉnh rồi chạy lại.")
            with st.expander("Chi tiết kỹ thuật (dành cho quản trị viên)"):
                st.code(f"{type(exc).__name__}: {str(exc)[:500]}")
            return
        box.update(label=f"Hoàn tất Bước {start} · chờ xác nhận", state="complete")
    ss.stop, ss.stage = start, "review"
    st.rerun()


def status_badge(n: int):
    stt = S["statuses"].get(n)
    if stt == "PASS":
        nxt = f"sang Bước {n + 1}" if n < 8 else "xem kết quả cuối cùng"
        st.markdown(f"<div class='ok'>✅ <b>Kết quả kiểm soát: Đạt</b> — bấm xác nhận để {nxt}.</div>", unsafe_allow_html=True)
    elif stt:
        st.markdown("**Kết quả kiểm soát:** Cần người duyệt")


_KEY = [0]


def _next_key():
    _KEY[0] += 1
    return _KEY[0]


def show_charts(section_filter=None, cols=2):
    ch = [c for c in S["charts"] if section_filter is None or c["section"] in section_filter]
    grid = st.columns(cols)
    for i, c in enumerate(ch):
        with grid[i % cols]:
            fig = interactive(c, S)
            if fig is not None:
                st.plotly_chart(fig, width="stretch", config={"displaylogo": False}, key=f"pl_{c['id']}_{_next_key()}")
                st.caption(c["caption"].replace("Hình: ", ""))
            else:
                st.image(c["path"], caption=c["caption"], width="stretch")
            lead = S.get("story", {}).get("chart_leads", {}).get(c["id"])
            if lead:
                st.caption("💬 " + lead)


# ------------------------------------------------------------ render từng bước
def render1():
    st.markdown(S["biz_md"])


ISSUE_TYPE = {"missing_critical": "Thiếu dữ liệu nghiêm trọng", "wrong_dtype": "Sai định dạng", "negative_values": "Giá trị âm bất thường",
              "outliers": "Giá trị ngoại lệ", "future_dates": "Ngày ở tương lai", "duplicates": "Dòng trùng lặp", "logic": "Vi phạm logic nghiệp vụ"}
SEVERITY = {"high": "Cao", "medium": "Trung bình", "low": "Thấp"}
TYPE_VI = {"numeric": "Số", "numeric_as_text": "Số lưu dạng chữ", "datetime": "Ngày tháng", "datetime_as_text": "Ngày lưu dạng chữ",
           "categorical": "Phân loại", "text": "Văn bản", "id": "Mã định danh", "boolean": "Đúng/Sai", "empty": "Trống"}
MODEL_VI = {"RandomForestRegressor": "Rừng ngẫu nhiên (dự báo giá trị)", "RandomForestClassifier": "Rừng ngẫu nhiên (phân loại)"}
METRIC_VI = {"r2": "Mức giải thích biến động (0–1)", "rmse": "Sai số trung bình bình phương", "mae": "Sai số tuyệt đối trung bình",
             "accuracy": "Tỷ lệ dự đoán đúng", "f1_macro": "Điểm cân bằng chính xác", "roc_auc": "Khả năng phân biệt nhãn"}
TEST_VI = {"Kruskal-Wallis": "So sánh nhiều nhóm", "Spearman": "Tương quan thứ hạng", "Chi-square": "Kiểm định độc lập"}


def render2():
    pr = S["prof"]
    ov = pr["overview"]
    c = st.columns(5)
    c[0].metric("Số dòng", f"{ov['rows']:,}".replace(",", "."))
    c[1].metric("Số cột", ov["columns"])
    c[2].metric("Dung lượng", f"{ov['memory_mb']:.2f} MB".replace(".", ","))
    c[3].metric("Dòng trùng lặp", ov["duplicates"])
    c[4].metric("Điểm chất lượng", f"{pr['quality_score']}/10".replace(".", ","))
    st.markdown(f"**Đánh giá rủi ro:** {pr['risk_comment']}")
    st.markdown("##### Báo cáo kiểm định chất lượng dữ liệu: các điểm bất thường về định dạng và logic")
    if pr["issues"]:
        t = pd.DataFrame(pr["issues"]).rename(columns={"column": "Cột", "type": "Loại bất thường", "severity": "Mức độ", "detail": "Chi tiết"})
        t["Loại bất thường"] = t["Loại bất thường"].map(ISSUE_TYPE).fillna(t["Loại bất thường"])
        t["Mức độ"] = t["Mức độ"].map(SEVERITY)
        st.dataframe(t, width="stretch", hide_index=True)
    else:
        st.success("Không phát hiện điểm bất thường đáng kể.")
    with st.expander("📋 Cấu trúc và thống kê từng cột"):
        t = pd.DataFrame(pr["columns"]).drop(columns=["sample", "pandas_dtype"], errors="ignore")
        t["inferred_type"] = t["inferred_type"].map(TYPE_VI)
        t = t.rename(columns={"column": "Cột", "inferred_type": "Kiểu dữ liệu", "missing": "Số ô trống", "missing_pct": "Tỷ lệ trống (%)",
                              "unique": "Số giá trị khác nhau", "mean": "Trung bình", "median": "Trung vị", "min": "Nhỏ nhất", "max": "Lớn nhất",
                              "outliers_iqr": "Số ngoại lệ", "negatives": "Số giá trị âm", "unparseable": "Không chuyển đổi được",
                              "min_date": "Ngày sớm nhất", "max_date": "Ngày muộn nhất", "future_dates": "Ngày tương lai"})
        st.dataframe(t, width="stretch", hide_index=True)
    st.markdown("##### Nhận định rủi ro nghiệp vụ")
    st.markdown(S["prof_comment"])


def render3():
    cl = S["clean"]
    c = st.columns(4)
    c[0].metric("Số dòng trước làm sạch", f"{cl['rows_before']:,}".replace(",", "."))
    c[1].metric("Số dòng sau làm sạch", f"{cl['rows_after']:,}".replace(",", "."))
    c[2].metric("Tỷ lệ dòng bị loại", f"{cl['data_loss_percentage']}%".replace(".", ","))
    c[3].metric("Số ô được điền bổ sung", f"{cl['imputed_cells']:,}".replace(",", "."))
    st.markdown("**Nhật ký làm sạch**")
    st.markdown("\n".join(f"- {x}" for x in cl["cleaning_action_log"]) or "- Không cần thao tác làm sạch.")
    if cl["logic_checks"]:
        st.markdown("**Kiểm tra luật nghiệp vụ**")
        t = pd.DataFrame(cl["logic_checks"])[["rule", "violations"]].rename(columns={"rule": "Luật", "violations": "Số dòng vi phạm"})
        st.dataframe(t, width="stretch", hide_index=True)
    if cl["distribution_shift"]:
        st.markdown("**Kiểm tra sai lệch phân phối trước và sau khi điền bổ sung**")
        t = pd.DataFrame(cl["distribution_shift"]).rename(columns={"column": "Cột", "mean_before": "Trung bình trước", "mean_after": "Trung bình sau",
                                                                    "median_before": "Trung vị trước", "median_after": "Trung vị sau",
                                                                    "mean_change_pct": "Lệch trung bình (%)", "median_change_pct": "Lệch trung vị (%)", "flag": "Vượt ngưỡng 15%"})
        st.dataframe(t, width="stretch", hide_index=True)
    with st.expander("Xem trước dữ liệu sạch"):
        st.dataframe(cl["df"].head(50), width="stretch")
        st.download_button("⬇️ Tải dữ liệu sạch (CSV)", cl["df"].to_csv(index=False).encode("utf-8-sig"), "du_lieu_sach.csv", "text/csv")


def render4():
    mp, pp = S["mplan"], S["pp_summary"]
    st.markdown(f"**Thiết lập phân tích:** biến mục tiêu `{mp['target']}` · chỉ số chính `{mp['kpi']}` · cột thời gian `{mp['date_col']}` · "
                f"phân khúc so sánh: {', '.join(mp['segment_cols']) or 'không có'}")
    st.markdown("##### Khám phá dữ liệu")
    show_charts({2, 3})
    st.markdown("**Nhật ký tiền xử lý**")
    st.markdown("\n".join(f"- {x}" for x in pp["preprocessing_log"]))
    for w in pp.get("warnings", []):
        st.info(w)
    lk, ib, dm = pp.get("leakage_check"), pp.get("imbalance_check"), pp.get("dimensionality_check")
    c = st.columns(3)
    if lk:
        sus = lk.get("suspicious_features") or []
        c[0].markdown("**Rò rỉ dữ liệu**  \n" + ("Chuẩn hóa chỉ dựa trên tập học: " + ("đạt" if lk.get("scaler_fit_on_train_only") else "chưa đạt")) +
                      "  \n" + (f"Biến đáng ngờ: {', '.join(f['feature'] for f in sus)}" if sus else "Không có biến đáng ngờ"))
    if ib:
        c[1].markdown(f"**Cân bằng nhãn**  \nTrước: {ib['before']}  \nSau: {ib['after']}")
    if dm:
        c[2].markdown(f"**Số chiều dữ liệu**  \n{dm['cols_before']} biến gốc → {dm['cols_after']} cột sau mã hóa  \nTỷ lệ cột/dòng: {dm['ratio']}")


def render5():
    an = S["analysis"]
    if an["metrics"]:
        m = an["metrics"]
        st.markdown(f"**Mô hình:** {MODEL_VI.get(m['model'], m['model'])} · biến mục tiêu `{an['outcome']}`")
        t = pd.DataFrame({"Tập học": m["train"], "Tập kiểm tra": m["test"], "Mô hình đối chứng": m["baseline_test"]})
        t.index = [METRIC_VI.get(i, i) for i in t.index]
        st.dataframe(t, width="stretch")
    if an["metric_tree"]:
        mt = an["metric_tree"]
        comp = " · ".join(f"{k}: {v:+.1f}%" for k, v in mt["components"].items())
        st.markdown(f"**Phân rã chỉ số:** {an['kpi']} = Số giao dịch × Giá trị trung bình mỗi giao dịch → tổng thay đổi "
                    f"**{mt['total_change_pct']:+.1f}%** ({comp}). Yếu tố chính: **{mt['main_driver']}**.")
    show_charts({4})
    if an["tests"]:
        with st.expander(f"Kiểm định thống kê: {sum(t['significant'] for t in an['tests'])}/{len(an['tests'])} yếu tố có ý nghĩa"):
            t = pd.DataFrame(an["tests"])[["feature", "test", "p_adj", "significant"]]
            t["test"] = t["test"].map(TEST_VI).fillna(t["test"])
            t = t.rename(columns={"feature": "Yếu tố", "test": "Phép kiểm định", "p_adj": "Xác suất ngẫu nhiên (đã hiệu chỉnh)", "significant": "Có ý nghĩa (độ tin cậy 95%)"})
            st.dataframe(t, width="stretch", hide_index=True)
    st.markdown("##### Nhận định chính")
    for i in an["insights"]:
        st.markdown(f"- **{i.get('insight', '')}**  \n  _Bằng chứng:_ {i.get('evidence', '')}  \n  _Hành động gợi ý:_ {i.get('action_hint', '')}")
    if an.get("validation_report"):
        st.markdown(f"**Đánh giá độ tin cậy:** {an['validation_report']}")
    for f in an["validation"]["flags"]:
        st.warning(f)


def render6():
    es = S["story"]["executive_summary"]
    st.markdown(f"#### 🎯 {es.get('headline', '')}")
    st.markdown("\n".join(f"- {k}" for k in es.get("key_points", [])))
    if es.get("top_recommendation"):
        st.markdown(f"**Khuyến nghị ưu tiên:** {es['top_recommendation']}")
    with st.expander("Biểu đồ kèm diễn giải"):
        show_charts()
    st.markdown("**Kết quả rà soát trình bày**")
    st.markdown("\n".join(f"- {x}" for x in S["viz_audit"]["visualization_audit_log"]))


def render7():
    pl = S["plan"]
    cols = {"what": "Hành động", "why": "Lý do", "who": "Người phụ trách", "when": "Thời hạn", "where": "Phạm vi", "how": "Cách thực hiện",
            "kpi": "Chỉ tiêu", "impact": "Tác động", "effort": "Nguồn lực"}
    lv = {"H": "Cao", "M": "Trung bình", "L": "Thấp"}
    for h, label in HORIZONS.items():
        acts = [a for a in pl.get("actions", []) if a.get("horizon") == h]
        if acts:
            st.markdown(f"##### {label}")
            t = pd.DataFrame(acts)[[c for c in cols if c in acts[0]]]
            for c in ("impact", "effort"):
                if c in t:
                    t[c] = t[c].astype(str).str[:1].str.upper().map(lv).fillna(t[c])
            st.dataframe(t.rename(columns=cols), width="stretch", hide_index=True)
    if pl.get("risks"):
        st.markdown("##### Rủi ro và phương án dự phòng")
        st.dataframe(pd.DataFrame(pl["risks"]).rename(columns={"risk": "Rủi ro", "mitigation": "Phương án dự phòng"}), width="stretch", hide_index=True)
    if pl.get("limitations"):
        st.markdown("**Hạn chế của phân tích:** " + "; ".join(pl["limitations"]))
    st.markdown("**Rà soát kế hoạch:** " + " · ".join(S["plan_audit"]["strategic_audit_report"]))


def render8():
    st.markdown("**Kết quả rà soát trước khi xuất bản**")
    st.markdown("\n".join(f"- {x}" for x in S["export"]["report_readiness_summary"]))


RENDER = {1: render1, 2: render2, 3: render3, 4: render4, 5: render5, 6: render6, 7: render7, 8: render8}


def _fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    a = abs(v)
    s = f"{v/1e9:,.2f} tỷ" if a >= 1e9 else f"{v/1e6:,.1f} triệu" if a >= 1e6 else f"{v:,.0f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def vn_table(t: pd.DataFrame) -> pd.DataFrame:
    """Định dạng số kiểu Việt Nam: dấu chấm phân cách nghìn, dấu phẩy thập phân."""
    t = t.copy()
    for c in t.columns:
        if pd.api.types.is_numeric_dtype(t[c]):
            dec = 2 if "%" in str(c) else 0
            t[c] = t[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", "."))
    return t


def render_dashboard():
    """Bảng điều khiển 2 tầng: Tầng I tính toán, Tầng II trực quan."""
    df, types, mp = S["clean"]["df"], S["clean"]["types"], S["mplan"]
    p = D.build_params(df, mp, types)
    if not p["kpi"]:
        st.info("Dữ liệu không có cột số phù hợp làm chỉ số chính, nên chưa dựng được bảng điều khiển.")
        return
    # ---- Khu vực tiêu đề & bộ lọc
    st.markdown(f"### Bảng điều khiển {p['kpi']}")
    f = st.columns([2] + [1] * min(2, len(p["segments"])))
    rng = None
    if p.get("date_col"):
        a, b = p["ngay_dau"].date(), p["ngay_cuoi"].date()
        rng = f[0].date_input("Khoảng thời gian", (a, b), min_value=a, max_value=b, format="DD/MM/YYYY")
        if not isinstance(rng, (tuple, list)) or len(rng) != 2:
            rng = (a, b)
    segf = {}
    for i, s in enumerate(p["segments"][:2], 1):
        segf[s] = f[i].multiselect(s, sorted(df[s].dropna().astype(str).unique()), placeholder="Tất cả")
    d = D.apply_filters(df, p, rng, segf)
    if d.empty:
        st.warning("Không có dữ liệu trong phạm vi lọc đã chọn.")
        return
    tables = D.summary_tables(d, p)
    kt = D.kpi_table(d, p, tables)
    # ---- Khu vực thẻ chỉ số
    cards = kt[kt["Giá trị"].notna()].head(5)
    cs = st.columns(len(cards))
    for col, (_, r) in zip(cs, cards.iterrows()):
        chg = r["Thay đổi (%)"]
        cls, arrow = ("flat", "") if chg is None or pd.isna(chg) else (("up", "▲") if chg >= 0 else ("down", "▼"))
        dtxt = f"{arrow} {chg:+.1f}%".replace(".", ",") + f" · {r['Ghi chú'].lower()}" if cls != "flat" else r["Ghi chú"]
        col.markdown(f"<div class='kpi'><div class='lb'>{r['Chỉ số']}</div><div class='v'>{_fmt(r['Giá trị'])}</div><div class='d {cls}'>{dtxt}</div></div>", unsafe_allow_html=True)
    st.write("")
    # ---- Khu vực biểu đồ chính
    seg = p["segments"][0] if p["segments"] else None
    c1, c2 = st.columns([3, 2])
    if "theo_thang" in tables:
        c1.plotly_chart(D.fig_trend(tables["theo_thang"], p["kpi"]), width="stretch", config={"displaylogo": False}, key="db_trend")
    if seg:
        c2.plotly_chart(D.fig_donut(tables[f"theo_{seg}"], seg), width="stretch", config={"displaylogo": False}, key="db_donut")
    c3, c4 = st.columns([3, 2])
    seg2 = p["segments"][1] if len(p["segments"]) > 1 else seg
    if seg2:
        c3.plotly_chart(D.fig_bar(tables[f"theo_{seg2}"], seg2), width="stretch", config={"displaylogo": False}, key="db_bar")
    # ---- Khu vực cảnh báo
    with c4:
        st.markdown("**Vùng cảnh báo**")
        for a in D.alerts(tables, p):
            st.markdown(f"<div class='al {a['muc']}'>{a['noi_dung']}</div>", unsafe_allow_html=True)
    # ---- Chi tiết
    if seg2:
        t = tables[f"theo_{seg2}"]
        st.markdown(f"**Nhóm dẫn đầu và nhóm thấp nhất theo {seg2}**")
        st.dataframe(vn_table(pd.concat([t.head(3), t.tail(3)]).drop_duplicates()), width="stretch")
    # ---- Tầng I: phần tính toán
    with st.expander("Tầng tính toán: bảng chỉ số, bảng tổng hợp, tham số và dữ liệu nguồn"):
        st.markdown("**Bảng chỉ số cốt lõi**")
        st.dataframe(vn_table(kt), width="stretch", hide_index=True)
        for k, t in tables.items():
            st.markdown(f"**Bảng tổng hợp {k.replace('theo_', 'theo ').replace('_', ' ')}**")
            st.dataframe(vn_table(t), width="stretch")
        st.markdown("**Bảng tham số**")
        st.dataframe(pd.DataFrame([
            ["Cột thời gian", p["date_col"]], ["Chỉ số chính", p["kpi"]], ["Phân khúc", ", ".join(p["segments"])],
            ["Kỳ so sánh", "Tháng gần nhất so với tháng liền trước"], ["Ngưỡng cảnh báo đỏ", f"Giảm từ {abs(p['nguong_giam_do']):.0f}% trở lên"],
            ["Ngưỡng cảnh báo vàng", "Giảm dưới 10%"]], columns=["Tham số", "Giá trị"]), width="stretch", hide_index=True)
        st.markdown(f"**Dữ liệu nguồn đã lọc** ({len(d):,} dòng)".replace(",", "."))
        st.dataframe(d.head(200), width="stretch")


def export_widget():
    ex = S["export"]
    st.markdown("<div class='export'>", unsafe_allow_html=True)
    st.markdown("#### ✅ Báo cáo phân tích đã sẵn sàng\n"
                "- 5 phần chuẩn: tóm tắt điều hành, hiện trạng dữ liệu, khám phá dữ liệu, nguyên nhân gốc rễ, đề xuất và lộ trình\n"
                "- Biểu đồ đã qua rà soát tính trung thực, mỗi hình có diễn giải và chú thích\n"
                "- Kế hoạch hành động có người phụ trách, thời hạn và chỉ tiêu đo được")
    c1, c2 = st.columns(2)
    with open(ex["docx"], "rb") as f:
        c1.download_button("📄 Tải báo cáo Word", f.read(), os.path.basename(ex["docx"]),
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", width="stretch", type="primary")
    with open(ex["pdf"], "rb") as f:
        c2.download_button("📑 Tải báo cáo PDF", f.read(), os.path.basename(ex["pdf"]), "application/pdf", width="stretch", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)


# ================================================================== MAIN
st.title("📊 Tác tử Phân tích Dữ liệu Kinh doanh")
st.caption("Quy trình 8 bước khép kín · Con người phê duyệt tại các điểm rủi ro · Mọi con số do mã nguồn tính, mô hình ngôn ngữ chỉ diễn giải")

# ---------------- BƯỚC 1: nhập bối cảnh
if ss.stage == "s1_input":
    st.subheader("Bước 1 · Tiếp nhận bối cảnh doanh nghiệp")
    txt = st.text_area("Mô tả doanh nghiệp, vấn đề cần giải quyết, chỉ số mục tiêu, phạm vi thời gian/khu vực…", height=180,
                       placeholder="Ví dụ: Chuỗi bán lẻ điện máy B2C, doanh thu 6 tháng gần đây giảm 13%. Cần tìm nguyên nhân gốc rễ…")
    docs = st.file_uploader("Tài liệu nội bộ (PDF / Word / PowerPoint / TXT) — tuỳ chọn", type=["pdf", "docx", "pptx", "txt", "md"], accept_multiple_files=True)
    if st.button("🔍 Phân tích bối cảnh", type="primary", disabled=not (txt.strip() or docs)):
        with st.spinner("Đang trích xuất Hồ sơ Bối cảnh Doanh nghiệp…"):
            docs_text = "\n\n".join(f"=== {d.name} ===\n{extract_text(d.name, d.getvalue())}" for d in (docs or []))
            P.step1(S, llm, txt, docs_text)
        ss.stage = "s1_review"
        st.rerun()

# ---------------- Lịch sử các bước đã chốt
if ss.stage not in ("s1_input",):
    done_steps = sorted({l["step"] for l in S["logs"]})
    current = 1 if ss.stage == "s1_review" else 2 if ss.stage == "s2_review" else ss.stop if ss.stage == "review" else None
    for n in done_steps:
        if n == current:
            continue
        with st.expander(f"{'✅' if S['statuses'].get(n) == 'PASS' or n in (1, 2) else '⚠️→✅'} Bước {n} · {P.STEP_NAMES[n]}", expanded=False):
            RENDER[n]()
            if n > 2:
                status_badge(n)

if ss.stage == "s1_review":
    st.subheader("Bước 1 · Hồ sơ Bối cảnh Doanh nghiệp")
    render1()
    st.markdown("<div class='cp'>⚠️ <b>[TRẠM KIỂM SOÁT BƯỚC 1]</b>: Tôi đã tiếp nhận và tổng hợp hồ sơ bối cảnh doanh nghiệp như trên.<br>"
                "👉 Bạn hãy kiểm tra thông tin đã chính xác và đủ ý chưa?<br>"
                "- Nếu <b>ĐÃ ĐÚNG</b>: bấm <b>Xác nhận</b> để chuyển sang Bước 2 (Đánh giá chất lượng dữ liệu thô).<br>"
                "- Nếu <b>CẦN CHỈNH SỬA / BỔ SUNG</b>: ghi rõ phần cần sửa rồi bấm <b>Cập nhật hồ sơ</b>.</div>", unsafe_allow_html=True)
    fb = st.text_area("Yêu cầu chỉnh sửa / bổ sung", key="fb1")
    c1, c2 = st.columns([1, 1])
    if c1.button("✅ Xác nhận — Tiếp tục", type="primary", width="stretch"):
        ss.stage = "s2_upload"
        st.rerun()
    if c2.button("✏️ Cập nhật hồ sơ", width="stretch", disabled=not fb.strip()):
        with st.spinner("Đang cập nhật hồ sơ…"):
            P.step1(S, llm, S["user_text"], S["docs_text"], feedback=fb)
        st.rerun()

elif ss.stage == "s2_upload":
    st.subheader("Bước 2 · Tải lên dữ liệu thô")
    st.info("Bối cảnh đã được lưu làm Kim chỉ nam (System Context) cho các bước tiếp theo. Mời bạn tải lên tệp dữ liệu thô.")
    up = st.file_uploader("Dữ liệu thô (CSV / Excel)", type=["csv", "xlsx", "xls", "xlsm"])
    if up and st.button("🧪 Quét chất lượng dữ liệu", type="primary"):
        with st.spinner("Đang quét schema, giá trị thiếu, trùng lặp, ngoại lệ, logic…"):
            try:
                P.step2(S, llm, up.name, up.getvalue())
            except Exception as exc:
                st.error(f"Không đọc được tệp: {exc}")
                st.stop()
        ss.stage = "s2_review"
        st.rerun()

elif ss.stage == "s2_review":
    st.subheader("Bước 2 · Báo cáo kiểm định chất lượng dữ liệu")
    render2()
    st.markdown("<div class='cp'>⚠️ <b>[TRẠM KIỂM SOÁT BƯỚC 2]</b>: Tôi đã hoàn tất quét và đánh giá chất lượng tập dữ liệu thô.<br>"
                "👉 Nếu <b>ĐỒNG Ý</b>: bấm <b>Tiếp tục</b> để chuyển sang Bước 3 (Làm sạch thông minh + kiểm định tổn thất).<br>"
                "- Nếu <b>CẦN LƯU Ý ĐẶC BIỆT</b>: ghi chú bên dưới (ví dụ: 'Cột discount_code được phép trống, đừng xử lý nó').</div>", unsafe_allow_html=True)
    notes = st.text_area("Lưu ý đặc biệt cho bước làm sạch (tuỳ chọn)", key="fb2")
    if st.button("▶️ Tiếp tục", type="primary"):
        S["feedback"][2] = notes
        run_from(3)

elif ss.stage == "review":
    n = ss.stop
    st.subheader(f"Bước {n} · {P.STEP_NAMES[n]}")
    RENDER[n]()
    status_badge(n)
    reasons = S["logs"][-1]["summary"].get("reasons") or S["logs"][-1]["summary"].get("validation", {}).get("flags", [])
    extra = ""
    if n == 5 and S["analysis"].get("actionable_question_for_human"):
        extra = f"<br>❓ {S['analysis']['actionable_question_for_human']}"
    if S["statuses"].get(n) == "PASS":
        nxt = f"Bước {n + 1} · {P.STEP_NAMES[n + 1]}" if n < 8 else "màn hình kết quả cuối cùng"
        st.markdown(f"<div class='cp'>🛑 <b>TRẠM XÁC NHẬN BƯỚC {n}</b>: bước này đã đạt các tiêu chí kiểm soát.<br>"
                    f"👉 Vui lòng xem kết quả phía trên, rồi bấm <b>Xác nhận</b> để chuyển sang {nxt}, hoặc ghi chỉ dẫn để chạy lại bước này.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='cp'>{CP_MSG[n]}<br>" + "".join(f"- {r}<br>" for r in reasons) + f"{CP_ASK[n]}{extra}</div>", unsafe_allow_html=True)
    fb = st.text_area("Chỉ dẫn điều chỉnh cho tác tử", key=f"fb_{n}_{len(S['logs'])}")
    c1, c2 = st.columns(2)
    if c1.button("✅ Xác nhận & sang bước tiếp theo" if n < 8 else "✅ Xác nhận & xem kết quả", type="primary", width="stretch"):
        if n == 8:
            ss.stage = "done"
            st.rerun()
        run_from(n + 1)
    if c2.button(f"🔁 Chạy lại Bước {n} với điều chỉnh", width="stretch", disabled=not fb.strip()):
        S["feedback"][f"retry_{n}"] = fb
        run_from(n)

elif ss.stage == "done":
    st.subheader("Kết quả phân tích")
    tab1, tab2 = st.tabs(["📊 Bảng điều khiển", "📄 Báo cáo điều hành"])
    with tab1:
        render_dashboard()
    with tab2:
        if S.get("story"):
            render6()
        render8()
        export_widget()
