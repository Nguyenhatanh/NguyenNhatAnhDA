"""BƯỚC 6 (Trực quan hoá & biên soạn điều hành) và BƯỚC 7 (Kết luận & đề xuất chiến lược)."""
from __future__ import annotations

import re

from .llm import LLMClient, to_facts

JARGON = [r"p[-_ ]?value", r"p_adj", r"\bAUC\b", r"\bROC\b", r"random ?forest", r"\bVIF\b", r"kruskal", r"spearman", r"chi[- ]?square",
          r"\bSMOTE\b", r"\bF1\b", r"R²|\bR2\b", r"\bRMSE\b", r"\bMAE\b", r"\bIQR\b", r"permutation", r"epsilon", r"cram[ée]r",
          r"standardscaler", r"one[- ]hot", r"overfit", r"\bBH\b", r"benjamini", r"log[- ]?decomposition", r"\bKPI\b", r"\binsights?\b", r"\bAOV\b", r"dashboard", r"baseline"]
HORIZONS = {"quick_win": "Việc làm ngay (tuần đầu tiên)", "strategic": "Sáng kiến chiến lược (3–6 tháng)", "long_term": "Tầm nhìn dài hạn (6–12 tháng)"}
LEVEL = {"H": 3, "M": 2, "L": 1}


def find_jargon(text: str) -> list[str]:
    return sorted({m.group(0) for pat in JARGON for m in re.finditer(pat, text, flags=re.IGNORECASE)})


# ================================================================ BƯỚC 6
def _fallback_story(facts: dict, chart_list: list[dict]) -> dict:
    ins = facts.get("insights", [])
    head = ins[0]["insight"] if ins else "Phân tích hoàn tất; xem chi tiết các phát hiện bên dưới."
    leads = {}
    for c in chart_list:
        kv = [f"{k} = {v:,}" if isinstance(v, (int, float)) else f"{k} = {v}" for k, v in c["facts"].items()
              if isinstance(v, (int, float, str)) and k != "n_categories"][:3]
        leads[c["id"]] = f"{c['caption'].replace('Hình: ', '')} " + (f"Điểm chính: {'; '.join(kv)}." if kv else "")
    return {"executive_summary": {"headline": head, "key_points": [i["insight"] for i in ins[1:5]],
                                  "top_recommendation": ins[0].get("action_hint", "") if ins else ""},
            "chart_leads": leads, "data_narrative": "", "eda_narrative": "", "deep_dive_narrative": ""}


def build_story(llm: LLMClient, biz: dict, facts: dict, chart_list: list[dict], feedback: str | None = None) -> dict:
    charts_brief = [{"id": c["id"], "title": c["title"], "section": c["section"], "facts": c["facts"]} for c in chart_list]
    fb = _fallback_story(facts, chart_list)
    prompt = (
        "VAI TRÒ: Chuyên gia trình bày báo cáo điều hành.\n"
        "YÊU CẦU:\n1) Nguyên lý Kết luận đặt đầu báo cáo: kết luận quan trọng nhất + con số then chốt, đặt đầu tiên.\n"
        "2) Dịch toàn bộ thuật ngữ kỹ thuật sang ngôn ngữ quản trị (doanh thu, chi phí, rủi ro, thời gian). CẤM dùng các từ: "
        "p-value, AUC, Random Forest, F1, R², RMSE, SMOTE, IQR, Spearman, Kruskal, Chi-square, overfitting, One-Hot.\n"
        "3) Mỗi biểu đồ có 1 đoạn dẫn 2-3 câu trả lời 'Biểu đồ này chứng minh điều gì? So what?'.\n"
        "4) Chỉ dùng số liệu có trong dữ kiện.\n"
        f"BỐI CẢNH: {to_facts(biz, 2500)}\nDỮ KIỆN PHÂN TÍCH: {to_facts(facts, 9000)}\nBIỂU ĐỒ: {to_facts(charts_brief, 6000)}\n"
        f"FEEDBACK: {feedback or '(không có)'}\n"
        'JSON: {"executive_summary":{"headline":"","key_points":["",""],"top_recommendation":""},'
        '"chart_leads":{"<chart_id>":""},"data_narrative":"","eda_narrative":"","deep_dive_narrative":""}'
    )
    story = llm.json(prompt, fallback=fb)
    if not isinstance(story, dict) or "executive_summary" not in story:
        story = fb
    story.setdefault("chart_leads", {})
    for c in chart_list:  # bảo đảm mọi biểu đồ đều có đoạn dẫn
        story["chart_leads"].setdefault(c["id"], fb["chart_leads"][c["id"]])
    jar = find_jargon(_business_text(story))
    if jar:  # Critic: buộc viết lại khi còn thuật ngữ kỹ thuật
        story, _ = llm.critique_revise(story, "Báo cáo điều hành bằng ngôn ngữ kinh doanh",
                                       [f"Loại bỏ/dịch các thuật ngữ kỹ thuật còn sót: {', '.join(jar)}", "Headline phải có con số then chốt"],
                                       to_facts(facts, 6000))
    return story


def _business_text(story: dict) -> str:
    es = story.get("executive_summary", {})
    return " ".join([es.get("headline", ""), " ".join(es.get("key_points", [])), es.get("top_recommendation", ""),
                     " ".join(story.get("chart_leads", {}).values()), story.get("eda_narrative", ""), story.get("deep_dive_narrative", "")])


def audit_story(story: dict, chart_list: list[dict]) -> dict:
    log, reasons = [], []
    # 1) Trung thực biểu đồ
    for c in chart_list:
        notes = []
        if c["kind"] == "pie":
            notes.append("dùng biểu đồ tròn")
        if c["kind"] in ("bar", "barh", "hist") and not c["y_zero"]:
            notes.append("trục giá trị không bắt đầu từ 0")
        if c.get("n_categories", 0) > 10:
            notes.append(f"{c['n_categories']} danh mục (quá nhiều)")
        log.append(f"{c['title']}: " + ("; ".join(notes) if notes else ("đạt — trục giá trị bắt đầu từ 0, không quá 10 nhóm" if c["kind"] in ("bar", "barh", "hist") else "đạt — loại biểu đồ phù hợp, không chi tiết thừa")))
        if notes:
            reasons.append(f"Biểu đồ {c['id']}: {', '.join(notes)}")
    # 2) Kim tự tháp
    head = story.get("executive_summary", {}).get("headline", "")
    if not head or not re.search(r"\d", head):
        reasons.append("Headline chưa nêu kết luận cốt lõi kèm con số")
    log.append(f"Kết luận đặt đầu báo cáo: “{head[:160]}”")
    # 3) Ngôn ngữ kinh doanh
    jar = find_jargon(_business_text(story))
    if jar:
        reasons.append(f"Còn thuật ngữ kỹ thuật chưa dịch: {', '.join(jar)}")
    log.append("Ngôn ngữ kinh doanh: " + ("đạt" if not jar else f"còn thuật ngữ {jar}"))
    missing = [c["id"] for c in chart_list if not story.get("chart_leads", {}).get(c["id"])]
    if missing:
        reasons.append(f"Biểu đồ thiếu đoạn dẫn: {missing}")
    return {"visualization_audit_log": log, "status": "REQUIRE_HUMAN_REVIEW" if reasons else "PASS", "reasons": reasons}


# ================================================================ BƯỚC 7
def _fallback_plan(facts: dict) -> dict:
    ins = facts.get("insights", [])
    acts = []
    for i, h in enumerate(["quick_win", "strategic", "long_term"]):
        src = ins[i] if i < len(ins) else {"insight": "Thiết lập hệ thống theo dõi KPI", "action_hint": "Xây dashboard KPI"}
        acts.append({"horizon": h, "what": src.get("action_hint", ""), "why": src.get("insight", ""), "who": "Trưởng phòng Kinh doanh",
                     "when": ["Tuần 1", "Tháng 1–6", "Tháng 6–12"][i], "where": "Toàn hệ thống", "how": "Thí điểm A/B có nhóm đối chứng",
                     "kpi": "Cải thiện KPI mục tiêu ≥ 5% so với kỳ cơ sở", "impact": "M", "effort": ["L", "M", "H"][i]})
    return {"key_findings": [i["insight"] for i in ins[:4]], "actions": acts,
            "risks": [{"risk": "Dữ liệu lịch sử có thể không phản ánh hành vi tương lai", "mitigation": "Thí điểm quy mô nhỏ, đo lường 4 tuần trước khi nhân rộng"},
                      {"risk": "Kết quả là tương quan, chưa chứng minh nhân quả", "mitigation": "Thiết kế thử nghiệm A/B có nhóm đối chứng"}],
            "limitations": ["Phân tích dựa trên dữ liệu đã cung cấp; chưa có dữ liệu chi phí/marketing để tính ROI."]}


def build_plan(llm: LLMClient, biz: dict, facts: dict, history: dict, feedback: str | None = None) -> dict:
    fb = _fallback_plan(facts)
    prompt = (
        "VAI TRÒ: Cố vấn chiến lược cấp cao.\n"
        "Chuyển các nhận định thành kế hoạch hành động cụ thể, đo được, có thời hạn, trả lời đủ: ai, làm gì, ở đâu, khi nào, vì sao, thế nào. Mỗi hành động BẮT BUỘC có: what, why (dẫn insight + số liệu), who (chức danh cụ thể), "
        "when (mốc thời gian cụ thể), where, how, kpi (chỉ tiêu định lượng có con số), impact (H/M/L), effort (H/M/L), horizon (quick_win|strategic|long_term).\n"
        "Ít nhất 2 hành động mỗi horizon. Quick wins = impact cao, effort thấp. Nêu ≥3 rủi ro kèm mitigation và các hạn chế của dữ liệu/mô hình.\n"
        "Giải pháp KHÔNG được mâu thuẫn với lịch sử xử lý dữ liệu (ví dụ cột đã bị loại thì không dựa vào nó).\n"
        f"BỐI CẢNH: {to_facts(biz, 2500)}\nINSIGHTS & SỐ LIỆU: {to_facts(facts, 7000)}\nLỊCH SỬ XỬ LÝ (JSON logs): {to_facts(history, 4000)}\n"
        f"FEEDBACK: {feedback or '(không có)'}\n"
        'JSON: {"key_findings":[""],"actions":[{...}],"risks":[{"risk":"","mitigation":""}],"limitations":[""]}'
    )
    plan = llm.json(prompt, fallback=fb)
    if not isinstance(plan, dict) or not plan.get("actions"):
        plan = fb
    problems = audit_plan(plan)["reasons"]
    if problems:  # Critic -> Revision
        plan, _ = llm.critique_revise(plan, "Action plan SMART/5W1H", problems + ["Mỗi KPI phải có con số", "Mỗi horizon ≥ 2 hành động"], to_facts(facts, 5000))
    plan["actions"] = sorted(plan.get("actions", []), key=lambda a: (list(HORIZONS).index(a.get("horizon", "long_term")) if a.get("horizon") in HORIZONS else 3,
                                                                    -LEVEL.get(str(a.get("impact", "M"))[:1].upper(), 2) + LEVEL.get(str(a.get("effort", "M"))[:1].upper(), 2)))
    return plan


def audit_plan(plan: dict) -> dict:
    reasons, log = [], []
    acts = plan.get("actions", [])
    for i, a in enumerate(acts, 1):
        miss = [k for k in ("what", "who", "when", "kpi", "how") if not str(a.get(k, "")).strip()]
        if miss:
            reasons.append(f"Hành động #{i} thiếu {', '.join(miss)}")
        if a.get("kpi") and not re.search(r"\d", str(a["kpi"])):
            reasons.append(f"Hành động #{i}: KPI chưa định lượng (“{a['kpi']}”)")
    for h, label in HORIZONS.items():
        n = sum(a.get("horizon") == h for a in acts)
        log.append(f"{label}: {n} hành động")
        if n == 0:
            reasons.append(f"Thiếu nhóm {label}")
    if len(plan.get("risks", [])) < 2 or any(not r.get("mitigation") for r in plan.get("risks", [])):
        reasons.append("Thiếu rủi ro/phương án dự phòng (cần ≥ 2, đủ mitigation)")
    if not plan.get("limitations"):
        reasons.append("Chưa nêu hạn chế của phân tích")
    log.append(f"Rủi ro: {len(plan.get('risks', []))}; Hạn chế: {len(plan.get('limitations', []))}")
    return {"strategic_audit_report": log, "status": "REQUIRE_HUMAN_REVIEW" if reasons else "PASS", "reasons": reasons}
