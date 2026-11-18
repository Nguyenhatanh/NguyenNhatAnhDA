"""BƯỚC 4 — EDA + Tiền xử lý (Feature Engineering, Encoding, Scaling, Imbalance) + kiểm định kỹ thuật."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import charts
from .llm import LLMClient, to_facts
from .profiling import is_money_like

TARGET_HINTS = ("churn", "target", "label", "converted", "conversion", "revenue", "doanh_thu", "doanh thu", "sales",
                "profit", "return", "fraud", "default", "is_", "y")
DIM_RATIO_MAX = 0.10       # số cột / số dòng train tối đa
LEAK_CORR = 0.98
IMBALANCE_MINORITY = 0.20


# ------------------------------------------------------------- kế hoạch mô hình
def _heuristic_plan(df: pd.DataFrame, types: dict, biz: dict) -> dict:
    hint = (str(biz.get("kpi_hint", "")) + " " + str(biz.get("objectives", {}).get("main_problem", ""))).lower()
    nums = [c for c, t in types.items() if t == "numeric"]
    cats = [c for c, t in types.items() if t in ("categorical", "boolean")]
    dts = [c for c, t in types.items() if t == "datetime"]
    target = None
    kpi_hint = str(biz.get("kpi_hint", "")).lower()
    usable = [c for c in df.columns if types.get(c) in ("numeric", "categorical", "boolean")]
    for source in (kpi_hint, hint):  # 1) ưu tiên KPI người dùng nêu, rồi mới tới mô tả vấn đề
        hits = [c for c in usable if re.search(rf"(?<![\w-]){re.escape(c.lower())}(?![\w-])", source)]
        if hits:
            target = sorted(hits, key=lambda c: types.get(c) != "numeric")[0]
            break
    if not target:
        for h in TARGET_HINTS:  # 2) tên cột gợi ý biến mục tiêu
            cand = [c for c in df.columns if h in c.lower() and types.get(c) in ("numeric", "categorical", "boolean")]
            if cand:
                target = cand[0]
                break
    kpi = target if target in nums else next((c for c in nums if is_money_like(c)), nums[0] if nums else None)
    return {"target": target, "kpi": kpi, "date_col": dts[0] if dts else None,
            "exclude": [c for c, t in types.items() if t in ("id", "text")],
            "segment_cols": cats[:4], "ratio_features": [], "reason": "Heuristic theo tên cột và mô tả mục tiêu"}


def modeling_plan(llm: LLMClient, df: pd.DataFrame, types: dict, biz: dict, feedback: str | None = None) -> dict:
    base = _heuristic_plan(df, types, biz)
    schema = [{"column": c, "type": types[c], "unique": int(df[c].nunique()), "sample": [str(v) for v in df[c].dropna().unique()[:3]]}
              for c in df.columns]
    prompt = (
        "NHIỆM VỤ (Bước 4): Chọn thiết lập phân tích phù hợp mục tiêu kinh doanh.\n"
        "- target: biến mục tiêu để mô hình hoá (null nếu bài toán thuần mô tả).\n"
        "- kpi: cột số chính dùng để đo hiệu quả kinh doanh (thường là doanh thu).\n"
        "- date_col: cột thời gian chính (null nếu không có).\n"
        "- exclude: cột phải loại khỏi mô hình (mã định danh, văn bản tự do, biến chỉ biết SAU kết quả => rò rỉ dữ liệu).\n"
        "- segment_cols: tối đa 4 cột phân loại quan trọng để so sánh nhóm.\n"
        "- ratio_features: tối đa 3 cặp [tử, mẫu] là cột số có ý nghĩa kinh doanh (ví dụ doanh thu/số lượng = giá bán).\n"
        f"BỐI CẢNH: {to_facts(biz, 3000)}\nSCHEMA SAU LÀM SẠCH: {to_facts(schema, 6000)}\n"
        f"FEEDBACK NGƯỜI DÙNG: {feedback or '(không có)'}\nĐỀ XUẤT MẶC ĐỊNH: {to_facts(base)}\n"
        'Trả về JSON: {"target":..., "kpi":..., "date_col":..., "exclude":[], "segment_cols":[], "ratio_features":[[a,b]], "reason":"..."}'
    )
    p = llm.json(prompt, fallback=base) or base
    cols = set(df.columns)
    t = p.get("target") if "target" in p else base["target"]  # LLM thiếu trường => dùng heuristic
    plan = {
        "target": t if t in cols else None,
        "kpi": p.get("kpi") if p.get("kpi") in cols and types.get(p.get("kpi")) == "numeric" else base["kpi"],
        "date_col": p.get("date_col") if p.get("date_col") in cols and types.get(p.get("date_col")) == "datetime" else base["date_col"],
        "exclude": [c for c in (p.get("exclude") or []) if c in cols] or base["exclude"],
        "segment_cols": [c for c in (p.get("segment_cols") or []) if c in cols and types.get(c) in ("categorical", "boolean")][:4] or base["segment_cols"],
        "ratio_features": [r for r in (p.get("ratio_features") or []) if isinstance(r, list) and len(r) == 2
                           and all(x in cols and types.get(x) == "numeric" for x in r)][:3],
        "reason": p.get("reason", base["reason"]),
    }
    if plan["target"] in plan["exclude"]:
        plan["exclude"].remove(plan["target"])
    return plan


# ---------------------------------------------------------------------- EDA
def run_eda(df: pd.DataFrame, types: dict, plan: dict, prof_raw: dict, out_dir: str) -> list[dict]:
    out = []
    m = charts.missing_bar(prof_raw, out_dir)
    if m:
        out.append(m)
    kpi, dc = plan.get("kpi"), plan.get("date_col")
    agg = "sum" if kpi and is_money_like(kpi) else "mean"
    if kpi:
        out.append(charts.distribution(df, kpi, out_dir))
        if dc:
            t = charts.trend(df, dc, kpi, out_dir, agg)
            if t:
                out.append(t)
        for c in plan.get("segment_cols", [])[:2]:
            out.append(charts.category_bar(df, c, kpi, out_dir, agg))
    nums = [c for c, t in types.items() if t == "numeric" and c not in plan.get("exclude", [])]
    h = charts.corr_heatmap(df, nums, out_dir)
    if h:
        out.append(h)
    tgt = plan.get("target")
    if tgt and _is_classification(df[tgt], types.get(tgt)):
        out.append(charts.class_balance(df[tgt].dropna(), out_dir))
    return out


def _is_classification(y: pd.Series, t: str | None) -> bool:
    if t in ("categorical", "boolean"):
        return True
    return pd.api.types.is_numeric_dtype(y) and y.nunique() <= 10 and np.allclose(y.dropna() % 1, 0)


# ------------------------------------------------------------- tiền xử lý
def preprocess(df: pd.DataFrame, types: dict, plan: dict, use_smote: bool = True) -> dict:
    log, reasons, warnings_ = [], [], []
    target = plan.get("target")
    if not target:
        return {"mode": "descriptive", "preprocessing_log": ["Không có biến mục tiêu: chuyển sang phân tích mô tả, bỏ qua bước xây dựng mô hình."],
                "status": "PASS", "reasons": [], "warnings": []}

    d = df.copy()
    # Feature Engineering
    new_feats = []
    for c in [c for c, t in types.items() if t == "datetime" and c in d and c not in plan["exclude"]]:
        d[f"{c}_month"] = d[c].dt.month
        d[f"{c}_dow"] = d[c].dt.dayofweek
        d[f"{c}_is_weekend"] = (d[c].dt.dayofweek >= 5).astype(int)
        new_feats += [f"{c}_month", f"{c}_dow", f"{c}_is_weekend"]
    dts = [c for c, t in types.items() if t == "datetime"]
    if len(dts) >= 2:
        a, b = dts[0], dts[1]
        d[f"{b}_minus_{a}_days"] = (d[b] - d[a]).dt.days
        new_feats.append(f"{b}_minus_{a}_days")
    for a, b in plan.get("ratio_features", []):
        name = f"{a}_per_{b}"
        d[name] = d[a] / d[b].replace(0, np.nan)
        new_feats.append(name)
    if new_feats:
        log.append(f"Tạo {len(new_feats)} biến đặc trưng mới: {', '.join(new_feats)}")

    d = d[d[target].notna()]
    y_raw = d[target]
    is_clf = _is_classification(y_raw, types.get(target))
    y = y_raw.astype(str) if is_clf else y_raw.astype(float)

    drop = set(plan["exclude"]) | set(dts) | {target}
    feats = [c for c in d.columns if c not in drop]
    # Loại biến chứa trực tiếp biến mục tiêu trong tên ratio (ví dụ revenue_per_qty khi target=revenue)
    leak_by_name = [c for c in feats if c in new_feats and re.search(rf"(^|_){re.escape(target)}(_|$)", c)]
    if leak_by_name:
        feats = [c for c in feats if c not in leak_by_name]
        log.append(f"Loại {len(leak_by_name)} biến được tính từ chính biến mục tiêu (chống rò rỉ): {', '.join(leak_by_name)}")
    num_cols = [c for c in feats if pd.api.types.is_numeric_dtype(d[c])]
    cat_cols = [c for c in feats if c not in num_cols]
    high_card = [c for c in cat_cols if d[c].nunique() > 50]
    cat_cols = [c for c in cat_cols if c not in high_card]
    if high_card:
        log.append(f"Bỏ {len(high_card)} biến phân loại có trên 50 giá trị khác nhau để tránh bùng nổ số chiều: {', '.join(high_card)}")
    X = d[num_cols + cat_cols]
    n_before = X.shape[1]

    strat = y if is_clf and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    log.append(f"Chia tập học/tập kiểm tra theo tỷ lệ 80/20{', giữ nguyên tỷ lệ nhãn' if strat is not None else ''}: {len(X_tr)} / {len(X_te)} dòng")

    prep = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=15, sparse_output=False))]), cat_cols),
    ], remainder="drop", verbose_feature_names_out=False)
    Xt_tr = prep.fit_transform(X_tr)  # FIT CHỈ TRÊN TRAIN
    Xt_te = prep.transform(X_te)
    feat_names = list(prep.get_feature_names_out())
    log.append("Điền giá trị thiếu, chuẩn hóa biến số, mã hóa biến phân loại (tối đa 15 nhóm mỗi biến); tham số chỉ tính trên tập học rồi áp sang tập kiểm tra")
    log.append(f"Số cột: {n_before} biến gốc -> {len(feat_names)} cột sau mã hoá")

    # --- Kiểm tra 1: rò rỉ dữ liệu
    leak = {"scaler_fit_on_train_only": None, "suspicious_features": []}
    if num_cols:
        sc = prep.named_transformers_["num"].named_steps["sc"]
        imp = prep.named_transformers_["num"].named_steps["imp"]
        tr_means = pd.DataFrame(imp.transform(X_tr[num_cols]), columns=num_cols).mean().values
        leak["scaler_fit_on_train_only"] = bool(np.allclose(sc.mean_, tr_means, rtol=1e-6, atol=1e-9))
        if not leak["scaler_fit_on_train_only"]:
            reasons.append("Tham số chuẩn hóa không khớp với tập học (nghi rò rỉ dữ liệu)")
    y_num = pd.Series(pd.factorize(y_tr)[0] if is_clf else y_tr.values, index=y_tr.index)
    for c in num_cols:
        x = X_tr[c]
        if x.nunique() > 1:
            r = abs(x.corr(y_num, method="spearman"))
            if r >= LEAK_CORR:
                leak["suspicious_features"].append({"feature": c, "abs_corr": round(float(r), 4)})
    if leak["suspicious_features"]:
        reasons.append("Biến có tương quan ≥ 0.98 với mục tiêu (nghi gian lận): " + ", ".join(f["feature"] for f in leak["suspicious_features"]))

    # --- Kiểm tra 2: mất cân bằng nhãn
    imb = None
    if is_clf:
        before = y_tr.value_counts(normalize=True).round(4).to_dict()
        minority = min(before.values())
        imb = {"before": before, "after": before, "method": "none", "distortion": []}
        if minority < IMBALANCE_MINORITY and use_smote:
            min_count = int(y_tr.value_counts().min())
            try:
                from imblearn.over_sampling import SMOTE

                if min_count < 2:
                    raise ValueError("lớp thiểu số < 2 mẫu")
                sm = SMOTE(random_state=42, k_neighbors=min(5, min_count - 1))
                Xs, ys = sm.fit_resample(Xt_tr, y_tr)
                # Kiểm tra biến dạng: dịch chuyển trung bình đặc trưng (đơn vị độ lệch chuẩn)
                shift = np.abs(Xs.mean(axis=0) - Xt_tr.mean(axis=0)) / (Xt_tr.std(axis=0) + 1e-9)
                top = np.argsort(-shift)[:3]
                imb.update({"after": pd.Series(ys).value_counts(normalize=True).round(4).to_dict(), "method": f"SMOTE(k={min(5, min_count-1)}) chỉ trên Train",
                            "distortion": [{"feature": feat_names[i], "mean_shift_sd": round(float(shift[i]), 3)} for i in top]})
                if shift.max() > 0.5:
                    reasons.append(f"Việc sinh mẫu tổng hợp làm lệch trung bình biến {feat_names[int(np.argmax(shift))]} {shift.max():.2f} độ lệch chuẩn")
                Xt_tr, y_tr = Xs, pd.Series(ys)
                log.append(f"Cân bằng nhãn bằng phương pháp sinh mẫu tổng hợp trên tập học: lớp thiểu số {minority:.1%} -> {min(imb['after'].values()):.1%}")
            except Exception as exc:
                imb["method"] = f"class_weight='balanced' (SMOTE không áp dụng: {exc})"
                log.append(f"Mất cân bằng nhãn ({minority:.1%}): dùng trọng số cân bằng theo lớp")

    # --- Kiểm tra 3: bùng nổ số chiều + đa cộng tuyến
    ratio = len(feat_names) / max(1, len(X_tr))
    dim = {"cols_before": n_before, "cols_after": len(feat_names), "rows_train": len(X_tr), "ratio": round(ratio, 4), "high_corr_pairs": []}
    if ratio > DIM_RATIO_MAX:
        reasons.append(f"Tỷ lệ số cột/số dòng tập học = {ratio:.2f} > {DIM_RATIO_MAX} (nguy cơ mô hình học thuộc)")
    if len(num_cols) >= 2:
        cm = X_tr[num_cols].corr().abs()
        pairs = [(a, b, round(float(cm.loc[a, b]), 3)) for i, a in enumerate(num_cols) for b in num_cols[i + 1:] if cm.loc[a, b] > 0.9]
        dim["high_corr_pairs"] = pairs[:10]
        if pairs:
            warnings_.append(f"Đa cộng tuyến: {len(pairs)} cặp biến số có |r| > 0.9 (ảnh hưởng tới diễn giải, không chặn mô hình)")

    return {
        "mode": "classification" if is_clf else "regression", "target": target,
        "X_train": Xt_tr, "X_test": Xt_te, "y_train": y_tr, "y_test": y_te, "X_test_raw": X_te, "X_train_raw": X_tr,
        "prep": prep, "feature_names": feat_names, "raw_features": num_cols + cat_cols,
        "class_weight": "balanced" if imb and imb["method"].startswith("class_weight") else None,
        "preprocessing_log": log, "leakage_check": leak, "imbalance_check": imb, "dimensionality_check": dim,
        "status": "REQUIRE_HUMAN_REVIEW" if reasons else "PASS", "reasons": reasons, "warnings": warnings_,
    }


def summary_for_log(res: dict) -> dict:
    keep = ("mode", "target", "preprocessing_log", "leakage_check", "imbalance_check", "dimensionality_check", "status", "reasons", "warnings", "raw_features")
    return {k: res.get(k) for k in keep if k in res}
