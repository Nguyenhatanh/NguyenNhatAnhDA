"""Bộ điều phối 8 bước (độc lập UI) — giữ trạng thái phiên, JSON logs (short-term memory) và quy tắc chuyển trạng thái."""
from __future__ import annotations

import datetime as dt
import os
import tempfile

from . import charts, cleaning, context, modeling, narrative, preprocess, profiling, report
from .llm import LLMClient

PASS, REVIEW = "PASS", "REQUIRE_HUMAN_REVIEW"
STEP_NAMES = {1: "Tiếp nhận bối cảnh", 2: "Đánh giá chất lượng dữ liệu", 3: "Làm sạch dữ liệu", 4: "Khám phá & Tiền xử lý",
              5: "Phân tích & Mô hình hoá", 6: "Trực quan hoá & Báo cáo", 7: "Kết luận & Đề xuất", 8: "Đóng gói & Xuất tệp"}


def new_state() -> dict:
    return {"statuses": {}, "logs": [], "charts": [], "out_dir": tempfile.mkdtemp(prefix="ba_agent_"), "feedback": {}}


def _log(S: dict, step: int, status: str, summary: dict | list | str):
    S["statuses"][step] = status
    S["logs"].append({"step": step, "name": STEP_NAMES[step], "time": dt.datetime.now().isoformat(timespec="seconds"),
                      "status": status, "summary": summary})


def history(S: dict) -> list:
    """Short-term memory: bản tóm tắt logs để đưa vào Bước 7-8, tránh mâu thuẫn với dữ liệu đã làm sạch."""
    return [{"step": l["step"], "status": l["status"], "summary": str(l["summary"])[:600]} for l in S["logs"]]


# -------------------------------------------------------------------- BƯỚC 1
def step1(S, llm: LLMClient, user_text: str, docs_text: str, feedback: str | None = None):
    S["user_text"], S["docs_text"] = user_text, docs_text
    S["biz"] = context.build_profile(llm, user_text, docs_text, previous=S.get("biz") if feedback else None, feedback=feedback)
    S["biz_md"] = context.profile_markdown(S["biz"])
    _log(S, 1, REVIEW, S["biz"])  # luôn dừng chờ xác nhận


# -------------------------------------------------------------------- BƯỚC 2
def step2(S, llm: LLMClient, name: str, data: bytes):
    S["raw_name"] = name
    S["df_raw"] = profiling.load_data(name, data)
    S["prof"] = profiling.profile(S["df_raw"])
    S["prof_comment"] = profiling.commentary(llm, S["prof"], S["biz"])
    _log(S, 2, REVIEW, {k: S["prof"][k] for k in ("overview", "issues", "quality_score", "risk_comment")})


# -------------------------------------------------------------------- BƯỚC 3
def step3(S, llm: LLMClient, feedback: str | None = None):
    S["plan_clean"] = cleaning.llm_plan(llm, S["prof"], S["biz"], S["feedback"].get(2, ""), feedback, S.get("plan_clean") if feedback else None)
    S["clean"] = cleaning.execute(S["df_raw"], S["prof"], S["plan_clean"])
    _log(S, 3, S["clean"]["status"], cleaning.summary_for_log(S["clean"]))
    return S["clean"]["status"]


# -------------------------------------------------------------------- BƯỚC 4
def step4(S, llm: LLMClient, feedback: str | None = None, use_smote: bool = True):
    df, types = S["clean"]["df"], S["clean"]["types"]
    S["mplan"] = preprocess.modeling_plan(llm, df, types, S["biz"], feedback)
    S["charts"] = preprocess.run_eda(df, types, S["mplan"], S["prof"], S["out_dir"])
    S["pp"] = preprocess.preprocess(df, types, S["mplan"], use_smote=use_smote)
    S["pp_summary"] = preprocess.summary_for_log(S["pp"])
    _log(S, 4, S["pp"]["status"], {"plan": S["mplan"], **S["pp_summary"]})
    return S["pp"]["status"]


# -------------------------------------------------------------------- BƯỚC 5
def step5(S, llm: LLMClient, feedback: str | None = None):
    df, types, mp, pp = S["clean"]["df"], S["clean"]["types"], S["mplan"], S["pp"]
    S["charts"] = [c for c in S["charts"] if c["section"] != 4]  # chạy lại không nhân đôi biểu đồ
    outcome = mp["target"] or mp["kpi"]
    is_clf = pp.get("mode") == "classification"
    an = {"outcome": outcome, "kpi": mp["kpi"], "mode": pp.get("mode")}
    an["tests"] = modeling.statistical_tests(df, types, outcome, is_clf, mp["exclude"]) if outcome else []
    an["metric_tree"] = modeling.metric_tree(df, mp["date_col"], mp["kpi"])
    an["segment_drill"] = modeling.segment_drill(df, mp["date_col"], mp["kpi"], mp["segment_cols"]) if an["metric_tree"] else []
    if an["metric_tree"]:
        S["charts"].append(charts.metric_tree_bar(an["metric_tree"], S["out_dir"], mp["kpi"]))
    an["metrics"] = None
    if pp.get("mode") in ("classification", "regression"):
        tm = modeling.train_model(pp, S["out_dir"])
        an["metrics"] = tm["metrics"]
        S["charts"].append(tm["chart"])
    val = modeling.validate(an["metrics"], an["tests"], pp.get("mode", "descriptive"))
    facts = {k: an[k] for k in ("outcome", "kpi", "metric_tree", "segment_drill", "metrics")}
    facts["tests"] = an["tests"][:15]
    ins = modeling.insights(llm, S["biz"], facts, val, feedback)
    an.update({"insights": ins.get("insights", []), "validation": val, "validation_report": ins.get("validation_report", ""),
               "actionable_question_for_human": ins.get("actionable_question_for_human", ""), "critic_issues": ins.get("critic_issues", [])})
    S["analysis"] = an
    _log(S, 5, val["status"], {"validation": val, "metrics": an["metrics"], "n_tests": len(an["tests"]), "insights": an["insights"]})
    return val["status"]


def _analysis_facts(S) -> dict:
    an = S["analysis"]
    return {"outcome": an["outcome"], "kpi": an["kpi"], "insights": an["insights"], "metric_tree": an["metric_tree"],
            "segment_drill": an["segment_drill"], "significant_tests": [t for t in an["tests"] if t["significant"]][:10],
            "model_metrics": an["metrics"], "data_quality": {"score": S["prof"]["quality_score"], "loss_pct": S["clean"]["data_loss_percentage"]}}


# -------------------------------------------------------------------- BƯỚC 6
def step6(S, llm: LLMClient, feedback: str | None = None):
    S["story"] = narrative.build_story(llm, S["biz"], _analysis_facts(S), S["charts"], feedback)
    S["viz_audit"] = narrative.audit_story(S["story"], S["charts"])
    _log(S, 6, S["viz_audit"]["status"], S["viz_audit"])
    return S["viz_audit"]["status"]


# -------------------------------------------------------------------- BƯỚC 7
def step7(S, llm: LLMClient, feedback: str | None = None):
    S["plan"] = narrative.build_plan(llm, S["biz"], _analysis_facts(S), history(S), feedback)
    S["plan_audit"] = narrative.audit_plan(S["plan"])
    _log(S, 7, S["plan_audit"]["status"], S["plan_audit"])
    return S["plan_audit"]["status"]


# -------------------------------------------------------------------- BƯỚC 8
def step8(S, llm: LLMClient | None = None, feedback: str | None = None):
    S["export"] = report.build_and_validate(S, os.path.join(S["out_dir"], "export"))
    _log(S, 8, S["export"]["status"], {k: v for k, v in S["export"].items() if k not in ("docx", "pdf")})
    return S["export"]["status"]


STEPS = {3: step3, 4: step4, 5: step5, 6: step6, 7: step7, 8: step8}


def run_until_checkpoint(S, llm: LLMClient, start: int, progress=None) -> int:
    """Chạy tuần tự từ `start`; dừng ở bước đầu tiên có trạng thái REQUIRE_HUMAN_REVIEW. Trả về số bước dừng (9 = hoàn tất)."""
    for n in range(start, 9):
        if progress:
            progress(n)
        status = STEPS[n](S, llm, S["feedback"].pop(f"retry_{n}", None))
        if status != PASS:
            return n
    return 9
