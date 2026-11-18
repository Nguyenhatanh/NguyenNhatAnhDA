"""BƯỚC 5 — Kiểm định thống kê, mô hình hoá, phân rã chỉ số & Trạm kiểm định mô hình."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.pipeline import Pipeline

from . import charts
from .llm import LLMClient, to_facts
from .profiling import is_money_like

ALPHA = 0.05
GAP_CLF, GAP_REG = 0.10, 0.15     # chênh lệch Train-Test tối đa (F1 / R²)
MIN_F1, MIN_R2 = 0.60, 0.30       # ngưỡng hiệu năng tối thiểu


def _bh(pvals: list[float]) -> list[float]:
    """Hiệu chỉnh Benjamini-Hochberg cho nhiều kiểm định cùng lúc (giảm dương tính giả)."""
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(n)
    cum = 1.0
    for rank, idx in reversed(list(enumerate(order, 1))):
        cum = min(cum, p[idx] * n / rank)
        adj[idx] = cum
    return adj.tolist()


# ---------------------------------------------------------- kiểm định thống kê
def statistical_tests(df: pd.DataFrame, types: dict, outcome: str, is_clf: bool, exclude: list[str]) -> list[dict]:
    res = []
    y = df[outcome]
    for c in df.columns:
        if c == outcome or c in exclude or types.get(c) in ("datetime", "id", "text"):
            continue
        sub = df[[c, outcome]].dropna()
        if len(sub) < 20:
            continue
        try:
            if types.get(c) in ("categorical", "boolean"):
                if sub[c].nunique() < 2 or sub[c].nunique() > 30:
                    continue
                if is_clf:
                    ct = pd.crosstab(sub[c], sub[outcome])
                    chi2, p, dof, _ = stats.chi2_contingency(ct)
                    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else 0
                    res.append({"feature": c, "test": "Chi-square", "stat": round(chi2, 3), "p_value": p, "effect": round(float(v), 3), "effect_name": "Cramér's V"})
                else:
                    groups = [g[outcome].values for _, g in sub.groupby(c, observed=True) if len(g) >= 5]
                    if len(groups) < 2:
                        continue
                    h, p = stats.kruskal(*groups)
                    eps = (h - len(groups) + 1) / (len(sub) - len(groups))  # epsilon-squared
                    means = sub.groupby(c, observed=True)[outcome].mean().sort_values(ascending=False)
                    res.append({"feature": c, "test": "Kruskal-Wallis", "stat": round(h, 3), "p_value": p, "effect": round(float(max(eps, 0)), 4),
                                "effect_name": "epsilon²", "group_means": {str(k): round(float(v), 2) for k, v in means.head(6).items()},
                                "best_vs_worst_pct": round(100 * (means.iloc[0] - means.iloc[-1]) / abs(means.iloc[-1]), 1) if means.iloc[-1] else None})
            elif types.get(c) == "numeric":
                if is_clf:
                    groups = [g[c].values for _, g in sub.groupby(outcome, observed=True)]
                    if len(groups) < 2:
                        continue
                    h, p = stats.kruskal(*groups)
                    res.append({"feature": c, "test": "Kruskal-Wallis", "stat": round(h, 3), "p_value": p,
                                "effect": None, "effect_name": "", "class_medians": sub.groupby(outcome, observed=True)[c].median().round(2).astype(float).to_dict()})
                else:
                    r, p = stats.spearmanr(sub[c], sub[outcome])
                    res.append({"feature": c, "test": "Spearman", "stat": round(float(r), 4), "p_value": p, "effect": round(float(r), 4), "effect_name": "rho"})
        except Exception:
            continue
    adj = _bh([r["p_value"] for r in res])
    for r, a in zip(res, adj):
        r["p_value"] = float(r["p_value"])
        r["p_adj"] = round(float(a), 6)
        r["significant"] = bool(a < ALPHA)
    res.sort(key=lambda r: r["p_adj"])
    return res


# -------------------------------------------------------- phân rã chỉ số
def metric_tree(df: pd.DataFrame, date_col: str | None, kpi: str | None) -> dict | None:
    """KPI = Số giao dịch × Giá trị TB/giao dịch. So sánh nửa sau vs nửa đầu giai đoạn (log-decomposition, MECE)."""
    if not date_col or not kpi or not is_money_like(kpi):
        return None
    d = df[[date_col, kpi]].dropna().sort_values(date_col)
    if len(d) < 50:
        return None
    mid = d[date_col].min() + (d[date_col].max() - d[date_col].min()) / 2
    a, b = d[d[date_col] < mid], d[d[date_col] >= mid]
    if a.empty or b.empty or a[kpi].sum() <= 0 or b[kpi].sum() <= 0:
        return None
    n_a, n_b = len(a), len(b)
    aov_a, aov_b = a[kpi].mean(), b[kpi].mean()
    total = np.log(b[kpi].sum() / a[kpi].sum())
    comp = {"Số giao dịch": np.log(n_b / n_a), "Giá trị TB/giao dịch": np.log(aov_b / aov_a)}
    to_pct = lambda v: round(100 * (np.exp(v) - 1), 2)  # noqa: E731
    return {"period_a": f"{a[date_col].min():%d/%m/%Y}–{a[date_col].max():%d/%m/%Y}", "period_b": f"{b[date_col].min():%d/%m/%Y}–{b[date_col].max():%d/%m/%Y}",
            "kpi_a": round(float(a[kpi].sum()), 2), "kpi_b": round(float(b[kpi].sum()), 2), "transactions_a": n_a, "transactions_b": n_b,
            "aov_a": round(float(aov_a), 2), "aov_b": round(float(aov_b), 2), "total_change_pct": to_pct(total),
            "components": {k: to_pct(v) for k, v in comp.items()},
            "log_share_pct": {k: round(100 * v / total, 1) if total else None for k, v in comp.items()},
            "main_driver": max(comp, key=lambda k: abs(comp[k]))}


def segment_drill(df: pd.DataFrame, date_col: str | None, kpi: str | None, segs: list[str]) -> list[dict]:
    """Nhóm nào đóng góp nhiều nhất vào thay đổi KPI giữa hai nửa giai đoạn (bao trùm toàn bộ thay đổi)."""
    if not date_col or not kpi:
        return []
    d = df.dropna(subset=[date_col, kpi])
    mid = d[date_col].min() + (d[date_col].max() - d[date_col].min()) / 2
    d = d.assign(_half=np.where(d[date_col] < mid, "A", "B"))
    out = []
    for s in segs:
        p = d.pivot_table(index=s, columns="_half", values=kpi, aggfunc="sum", observed=True).fillna(0)
        if not {"A", "B"} <= set(p.columns):
            continue
        p["delta"] = p["B"] - p["A"]
        tot = p["delta"].sum()
        p = p.reindex(p["delta"].abs().sort_values(ascending=False).index)
        out.append({"segment": s, "total_delta": round(float(tot), 2),
                    "top_contributors": [{"group": str(i), "A": round(float(r.A), 2), "B": round(float(r.B), 2), "delta": round(float(r.delta), 2),
                                          "share_of_total_delta_pct": round(100 * r.delta / tot, 1) if tot else None} for i, r in p.head(5).iterrows()]})
    return out


# ----------------------------------------------------------------- mô hình
def train_model(pp: dict, out_dir: str) -> dict:
    is_clf = pp["mode"] == "classification"
    Xtr, Xte, ytr, yte = pp["X_train"], pp["X_test"], pp["y_train"], pp["y_test"]
    if is_clf:
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=-1, random_state=42,
                                       class_weight=pp.get("class_weight"))
        base = DummyClassifier(strategy="most_frequent")
    else:
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=42)
        base = DummyRegressor(strategy="mean")
    model.fit(Xtr, ytr)
    base.fit(Xtr, ytr)

    def score(m, X, y):
        p = m.predict(X)
        if is_clf:
            out = {"accuracy": round(accuracy_score(y, p), 4), "f1_macro": round(f1_score(y, p, average="macro"), 4)}
            if len(np.unique(y)) == 2 and hasattr(m, "predict_proba"):
                try:
                    pos = m.classes_[1]
                    out["roc_auc"] = round(roc_auc_score((np.asarray(y) == pos).astype(int), m.predict_proba(X)[:, 1]), 4)
                except Exception:
                    pass
            return out
        return {"r2": round(r2_score(y, p), 4), "rmse": round(float(np.sqrt(mean_squared_error(y, p))), 4), "mae": round(mean_absolute_error(y, p), 4)}

    metrics = {"model": type(model).__name__, "train": score(model, Xtr, ytr), "test": score(model, Xte, yte), "baseline_test": score(base, Xte, yte)}
    # Tầm quan trọng hoán vị trên tập TEST, theo biến gốc (dễ diễn giải cho kinh doanh)
    pipe = Pipeline([("prep", pp["prep"]), ("model", model)])
    scoring = "f1_macro" if is_clf else "r2"
    pi = permutation_importance(pipe, pp["X_test_raw"], yte, n_repeats=5, random_state=42, scoring=scoring, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=pp["raw_features"]).clip(lower=0).sort_values(ascending=False)
    metrics["importance"] = {k: round(float(v), 4) for k, v in imp.head(10).items()}
    chart = charts.importance_bar(imp[imp > 0].head(10) if (imp > 0).any() else imp.head(5), out_dir, "F1" if is_clf else "R²")
    return {"metrics": metrics, "chart": chart}


# --------------------------------------------------------------- kiểm định
def validate(metrics: dict | None, tests: list[dict], mode: str) -> dict:
    flags, ok = [], []
    sig = [t for t in tests if t["significant"]]
    if tests and not sig:
        flags.append("Không có yếu tố nào có ý nghĩa thống kê ở độ tin cậy 95% — các nhận định dễ mang tính cảm tính.")
    else:
        ok.append(f"{len(sig)}/{len(tests)} yếu tố có ý nghĩa thống kê ở độ tin cậy 95%.")
    if metrics:
        tr, te, bl = metrics["train"], metrics["test"], metrics["baseline_test"]
        if mode == "classification":
            gap = tr["f1_macro"] - te["f1_macro"]
            if gap > GAP_CLF:
                flags.append(f"Mô hình học thuộc: điểm tập học {tr['f1_macro']} so với tập kiểm tra {te['f1_macro']} (chênh {gap:.2f} > {GAP_CLF}).")
            if te["f1_macro"] < MIN_F1 or te["f1_macro"] < bl["f1_macro"] + 0.05:
                flags.append(f"Độ chính xác thấp: điểm tập kiểm tra {te['f1_macro']} (mô hình đối chứng {bl['f1_macro']}, ngưỡng {MIN_F1}).")
        else:
            gap = tr["r2"] - te["r2"]
            if gap > GAP_REG:
                flags.append(f"Mô hình học thuộc: mức giải thích tập học {tr['r2']} so với tập kiểm tra {te['r2']} (chênh {gap:.2f} > {GAP_REG}).")
            if te["r2"] < MIN_R2:
                flags.append(f"Độ chính xác thấp: mô hình chỉ giải thích được {te['r2']:.0%} biến động trên tập kiểm tra (ngưỡng {MIN_R2:.0%}).")
        if not flags or all("thống kê" in f for f in flags):
            ok.append("Mô hình ổn định giữa tập học và tập kiểm tra, vượt mô hình đối chứng.")
    return {"flags": flags, "ok": ok, "status": "REQUIRE_HUMAN_REVIEW" if flags else "PASS"}


def insights(llm: LLMClient, biz: dict, facts: dict, val: dict, feedback: str | None = None) -> dict:
    fb_ins = []
    for t in [t for t in facts.get("tests", []) if t["significant"]][:5]:
        ev = (f"Trung bình theo nhóm: {t['group_means']}" if t.get("group_means") else
              f"Tương quan {'thuận' if (t.get('effect') or 0) > 0 else 'nghịch'} mức {abs(t.get('effect') or 0):.2f}" if t["test"] == "Spearman" else "Khác biệt rõ giữa các nhóm")
        fb_ins.append({"insight": f"{t['feature']} ảnh hưởng rõ rệt tới {facts.get('outcome')} (độ tin cậy ≥ 95%).",
                       "evidence": ev, "action_hint": f"Ưu tiên tối ưu theo {t['feature']}."})
    mt = facts.get("metric_tree")
    if mt:
        fb_ins.insert(0, {"insight": f"{facts.get('kpi')} thay đổi {mt['total_change_pct']}% giữa hai nửa giai đoạn; động lực chính là {mt['main_driver']}.",
                          "evidence": str(mt["components"]), "action_hint": f"Tập trung vào {mt['main_driver']}."})
    fallback = {"insights": fb_ins, "validation_report": " ".join(val["ok"] + val["flags"]),
                "actionable_question_for_human": "Bạn có muốn điều chỉnh biến mục tiêu hoặc bổ sung dữ liệu không?" if val["flags"] else ""}
    prompt = (
        "VAI TRÒ: Chuyên gia khoa học dữ liệu kiểm định kết quả.\n"
        f"MỤC TIÊU KINH DOANH: {to_facts(biz.get('objectives', {}), 1500)}\n"
        f"KẾT QUẢ (tính bằng code): {to_facts(facts, 9000)}\n"
        f"KIỂM ĐỊNH TỰ ĐỘNG: {to_facts(val)}\nFEEDBACK BA: {feedback or '(không có)'}\n"
        "YÊU CẦU: Rút ra 3-6 insight trả lời đúng câu hỏi cốt lõi. CHỈ dùng kết luận có p_adj < 0.05 hoặc từ phân rã chỉ số. "
        "Mỗi insight phải có bằng chứng số liệu trích nguyên văn từ KẾT QUẢ và gợi ý hành động.\n"
        'JSON: {"insights":[{"insight":"","evidence":"","action_hint":""}], "validation_report":"", "actionable_question_for_human":""}'
    )
    draft = llm.json(prompt, fallback=fallback)
    if not isinstance(draft, dict) or not draft.get("insights"):
        draft = fallback
    draft, issues = llm.critique_revise(draft, "Insights từ kết quả thống kê", [
        "Insight có dựa trên p_adj < 0.05 hoặc phân rã chỉ số không?", "Có trả lời câu hỏi cốt lõi của doanh nghiệp không?",
        "Có hành động cụ thể không?"], to_facts(facts, 8000))
    draft["critic_issues"] = issues
    return draft
