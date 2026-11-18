"""BƯỚC 3 — Làm sạch dữ liệu thông minh + Trạm kiểm soát tổn thất / logic / dịch chuyển phân phối."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .llm import LLMClient, to_facts
from .profiling import coerce_datetime, coerce_numeric, detect_date_pairs, iqr_bounds, is_money_like, is_texty

LOSS_THRESHOLD = 10.0     # % dòng bị loại tối đa
SHIFT_THRESHOLD = 15.0    # % thay đổi mean/median tối đa sau điền khuyết
M_VI = {"median": "trung vị", "mean": "trung bình", "mode": "giá trị phổ biến nhất", "constant": "giá trị mặc định", "ffill": "giá trị liền trước"}
MISSING_ACTIONS = {"median", "mean", "mode", "drop_rows", "keep", "drop_column", "constant", "ffill"}


# ------------------------------------------------------------------- kế hoạch
def default_plan(prof: dict, user_notes: str = "") -> dict:
    """Kế hoạch luật xác định — dùng làm nền và fallback khi không có LLM."""
    types = prof["types"]
    cols = {c["column"]: c for c in prof["columns"]}
    actions, outliers = {}, {}
    for c, t in types.items():
        pct = cols[c]["missing_pct"]
        if t == "empty" or pct > 60:
            m = "drop_column"
        elif pct == 0:
            m = "keep"
        elif t in ("numeric", "numeric_as_text"):
            m = "median"  # median bền với outlier hơn mean
        elif t in ("datetime", "datetime_as_text"):
            m = "drop_rows" if pct < 5 else "keep"
        elif t in ("categorical", "boolean"):
            m = "mode" if pct < 5 else "constant"
        else:
            m = "keep"
        actions[c] = {"type_fix": "numeric" if t == "numeric_as_text" else "datetime" if t == "datetime_as_text" else "none",
                      "missing": m, "constant_value": "Không rõ"}
        if t in ("numeric", "numeric_as_text") and cols[c].get("outliers_iqr", 0):
            outliers[c] = "cap"  # winsorize ở ngưỡng IQR×3 => giữ dòng, giảm ảnh hưởng
    dts = [c for c, t in types.items() if t in ("datetime", "datetime_as_text")]
    rules = [{"name": f"{b} không trước {a}", "expr": f"`{b}` >= `{a}`"} for a, b in detect_date_pairs(dts)]
    for c, t in types.items():
        if t in ("numeric", "numeric_as_text") and is_money_like(c):
            rules.append({"name": f"{c} không âm", "expr": f"`{c}` >= 0"})
    return {"column_actions": actions, "drop_duplicates": True, "drop_negative_money_rows": True,
            "outlier_policy": outliers, "outlier_k": 3.0, "logic_rules": rules,
            "logic_violation_action": "flag", "drop_future_dates": True, "user_notes": user_notes}


def llm_plan(llm: LLMClient, prof: dict, biz: dict, user_notes: str, feedback: str | None, prev_plan: dict | None) -> dict:
    base = default_plan(prof, user_notes)
    prompt = (
        "NHIỆM VỤ (Bước 3): Đề xuất kế hoạch làm sạch dữ liệu theo nghiệp vụ. Bắt đầu từ KẾ HOẠCH MẶC ĐỊNH, chỉ sửa chỗ cần thiết.\n"
        "Lựa chọn missing hợp lệ: median|mean|mode|drop_rows|keep|drop_column|constant|ffill. "
        "outlier_policy: cap|drop|keep. logic_violation_action: flag|drop.\n"
        "logic_rules.expr phải là biểu thức pandas.eval, tên cột bọc trong backtick, TRUE nghĩa là hợp lệ.\n"
        "Tôn trọng tuyệt đối ghi chú/feedback của người dùng (ví dụ cột được phép trống thì đặt 'keep').\n\n"
        f"BỐI CẢNH DOANH NGHIỆP: {to_facts(biz, 3000)}\n"
        f"BÁO CÁO CHẤT LƯỢNG: {to_facts({'columns': prof['columns'], 'issues': prof['issues']}, 7000)}\n"
        f"GHI CHÚ NGƯỜI DÙNG Ở BƯỚC 2: {user_notes or '(không có)'}\n"
        f"FEEDBACK ĐIỀU CHỈNH: {feedback or '(không có)'}\n"
        f"KẾ HOẠCH MẶC ĐỊNH: {to_facts(prev_plan or base, 7000)}\n"
        "Trả về JSON cùng cấu trúc với kế hoạch mặc định."
    )
    plan = llm.json(prompt, fallback=prev_plan or base)
    return sanitize_plan(plan, base, prof)


def sanitize_plan(plan: dict, base: dict, prof: dict) -> dict:
    """Critic bằng code: loại bỏ hành động trỏ tới cột không tồn tại / giá trị không hợp lệ."""
    valid_cols = set(prof["types"])
    out = {**base, **{k: v for k, v in (plan or {}).items() if k in base}}
    acts = {}
    for c, a in (out.get("column_actions") or {}).items():
        if c in valid_cols and isinstance(a, dict):
            merged = {**base["column_actions"][c], **a}
            if merged.get("missing") not in MISSING_ACTIONS:
                merged["missing"] = base["column_actions"][c]["missing"]
            acts[c] = merged
    for c in valid_cols - set(acts):
        acts[c] = base["column_actions"][c]
    out["column_actions"] = acts
    out["outlier_policy"] = {c: p for c, p in (out.get("outlier_policy") or {}).items() if c in valid_cols and p in ("cap", "drop", "keep")}
    out["logic_rules"] = [r for r in (out.get("logic_rules") or []) if isinstance(r, dict) and r.get("expr")]
    if out.get("logic_violation_action") not in ("flag", "drop"):
        out["logic_violation_action"] = "flag"
    try:
        out["outlier_k"] = float(out.get("outlier_k", 3.0))
    except (TypeError, ValueError):
        out["outlier_k"] = 3.0
    return out


# ------------------------------------------------------------------- thực thi
def _mode(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else None


def execute(df_raw: pd.DataFrame, prof: dict, plan: dict) -> dict:
    df = df_raw.copy()
    n0 = len(df)
    log: list[str] = []
    types = dict(prof["types"])

    # 1) Chuẩn hoá chuỗi rỗng -> NaN, cắt khoảng trắng thừa
    for c in df.columns:
        if is_texty(df[c]):
            s = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = s.replace("", np.nan)

    # 2) Sửa kiểu dữ liệu
    for c, a in plan["column_actions"].items():
        if a.get("type_fix") == "numeric":
            before_na = df[c].isna().sum()
            df[c] = coerce_numeric(df[c])
            types[c] = "numeric"
            log.append(f"Ép kiểu số cột `{c}` (xử lý dấu phân cách, ký hiệu tiền); {int(df[c].isna().sum() - before_na)} giá trị không hợp lệ -> NaN")
        elif a.get("type_fix") == "datetime":
            before_na = df[c].isna().sum()
            df[c] = coerce_datetime(df[c])
            types[c] = "datetime"
            log.append(f"Chuẩn hoá ngày tháng cột `{c}`; {int(df[c].isna().sum() - before_na)} giá trị lỗi định dạng -> NaT")
        elif types.get(c) == "datetime":
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 3) Bỏ cột
    drop_cols = [c for c, a in plan["column_actions"].items() if a.get("missing") == "drop_column" and c in df]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log.append(f"Loại bỏ {len(drop_cols)} cột thiếu quá nặng/không dùng được: {', '.join(drop_cols)}")

    # 4) Trùng lặp
    if plan.get("drop_duplicates", True):
        d = int(df.duplicated().sum())
        if d:
            df = df.drop_duplicates()
            log.append(f"Xoá {d} dòng trùng lặp hoàn toàn")

    # 5) Giá trị âm ở cột tiền/số lượng (lỗi nhập liệu)
    if plan.get("drop_negative_money_rows", True):
        for c in [c for c in df.columns if types.get(c) == "numeric" and is_money_like(c)]:
            mask = df[c] < 0
            if mask.sum():
                log.append(f"Xoá {int(mask.sum())} dòng có `{c}` âm (lỗi nhập liệu)")
                df = df[~mask]

    # 6) Ngày tương lai
    if plan.get("drop_future_dates", True):
        today = pd.Timestamp.today().normalize()
        for c in [c for c in df.columns if types.get(c) == "datetime"]:
            mask = df[c] > today
            if mask.sum():
                log.append(f"Xoá {int(mask.sum())} dòng có `{c}` ở tương lai")
                df = df[~mask]

    # 7) Giá trị thiếu (ghi lại thống kê trước/sau để kiểm tra dịch chuyển phân phối)
    shift = []
    imputed_cells = 0
    for c, a in plan["column_actions"].items():
        if c not in df:
            continue
        m, na = a.get("missing"), int(df[c].isna().sum())
        if na == 0 or m in ("keep", "drop_column"):
            continue
        is_num = types.get(c) == "numeric"
        before = df[c].dropna()
        if m == "drop_rows":
            df = df[df[c].notna()]
            log.append(f"Xoá {na} dòng thiếu `{c}` (biến trọng yếu, không nên điền giả)")
            continue
        if m in ("median", "mean") and is_num:
            val = before.median() if m == "median" else before.mean()
        elif m == "mode":
            val = _mode(before)
        elif m == "ffill":
            df[c] = df[c].ffill().bfill()
            val = None
        else:  # constant hoặc median/mean trên cột không phải số
            val = a.get("constant_value", "Không rõ") if not is_num else before.median()
        if val is not None:
            df[c] = df[c].fillna(val)
        imputed_cells += na
        log.append(f"Điền {na} ô thiếu ở `{c}` bằng {M_VI.get(m, m)}" + (f" = {round(val, 4) if isinstance(val, float) else val}" if val is not None else ""))
        if is_num and len(before):
            after = df[c].astype(float)
            dm = 100 * abs(after.mean() - before.mean()) / (abs(before.mean()) or 1)
            dmed = 100 * abs(after.median() - before.median()) / (abs(before.median()) or 1)
            shift.append({"column": c, "mean_before": round(before.mean(), 4), "mean_after": round(after.mean(), 4),
                          "median_before": round(before.median(), 4), "median_after": round(after.median(), 4),
                          "mean_change_pct": round(dm, 2), "median_change_pct": round(dmed, 2),
                          "flag": bool(max(dm, dmed) > SHIFT_THRESHOLD)})

    # 8) Ngoại lệ
    k = plan.get("outlier_k", 3.0)
    for c, pol in plan.get("outlier_policy", {}).items():
        if c not in df or types.get(c) != "numeric" or pol == "keep":
            continue
        x = df[c].astype(float)
        lo, hi = iqr_bounds(x.dropna(), k)
        if is_money_like(c):
            lo = max(lo, 0)
        mask = (x < lo) | (x > hi)
        if not mask.sum():
            continue
        if pol == "drop":
            df = df[~mask]
            log.append(f"Xoá {int(mask.sum())} dòng ngoại lệ `{c}` ngoài khoảng [{lo:,.0f}; {hi:,.0f}] (ngưỡng tứ phân vị ×{k:g})")
        else:
            df[c] = x.clip(lo, hi)
            log.append(f"Giới hạn {int(mask.sum())} giá trị ngoại lệ `{c}` về khoảng [{lo:,.0f}; {hi:,.0f}] (ngưỡng tứ phân vị ×{k:g})")

    # 9) Kiểm tra logic nghiệp vụ (assert bằng code)
    logic = []
    for r in plan.get("logic_rules", []):
        try:
            ok = df.eval(r["expr"], engine="python")
            ok = ok.fillna(True) if hasattr(ok, "fillna") else ok
            viol = int((~ok.astype(bool)).sum())
            logic.append({"rule": r.get("name", r["expr"]), "expr": r["expr"], "violations": viol, "error": None})
            if viol and plan.get("logic_violation_action") == "drop":
                df = df[ok.astype(bool)]
                log.append(f"Xoá {viol} dòng vi phạm luật '{r.get('name')}'")
        except Exception as exc:  # luật không chạy được -> ghi log, không làm sập pipeline
            logic.append({"rule": r.get("name", r["expr"]), "expr": r["expr"], "violations": 0, "error": str(exc)[:150]})

    df = df.reset_index(drop=True)
    n1 = len(df)
    loss = round(100 * (n0 - n1) / n0, 2) if n0 else 0.0
    logic_viol = sum(l["violations"] for l in logic if plan.get("logic_violation_action") != "drop")
    shift_flags = [s for s in shift if s["flag"]]
    reasons = []
    if loss > LOSS_THRESHOLD:
        reasons.append(f"Tỷ lệ dòng bị loại {loss}% vượt ngưỡng {LOSS_THRESHOLD}%")
    if logic_viol:
        reasons.append(f"Phát hiện {logic_viol} điểm vi phạm logic nghiệp vụ")
    if shift_flags:
        reasons.append("Phân phối bị dịch chuyển > 15% sau điền khuyết ở: " + ", ".join(s["column"] for s in shift_flags))
    return {
        "df": df, "types": {c: types[c] for c in df.columns},
        "cleaning_action_log": log, "rows_before": n0, "rows_after": n1, "data_loss_percentage": loss,
        "imputed_cells": imputed_cells, "logic_checks": logic, "logic_violations": logic_viol,
        "distribution_shift": shift, "status": "REQUIRE_HUMAN_REVIEW" if reasons else "PASS", "reasons": reasons,
    }


def summary_for_log(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("df",)}
