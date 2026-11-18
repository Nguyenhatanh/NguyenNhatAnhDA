"""Biểu đồ nền trắng, trung thực: bar bắt đầu từ 0, không pie chart, tối đa 10 danh mục, không chartjunk."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PRIMARY, ACCENT, GREY, NEG = "#1F4E79", "#E07A1F", "#9AA5B1", "#C0392B"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#444444",
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def _fmt(v: float) -> str:
    a = abs(v)
    return f"{v/1e9:.2f} tỷ" if a >= 1e9 else f"{v/1e6:.2f} tr" if a >= 1e6 else f"{v/1e3:.1f} k" if a >= 1e4 else f"{v:,.2f}"


def _save(fig, out_dir: str, cid: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cid}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _chart(cid, path, title, caption, section, kind, facts):
    return {"id": cid, "path": path, "title": title, "caption": caption, "section": section, "kind": kind,
            "facts": facts, "y_zero": kind in ("bar", "barh", "hist"), "n_categories": facts.get("n_categories", 0)}


def missing_bar(prof: dict, out_dir: str) -> dict | None:
    data = sorted([(c["column"], c["missing_pct"]) for c in prof["columns"] if c["missing_pct"] > 0], key=lambda x: x[1])[-10:]
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(data) + 1.2))
    names, vals = zip(*data)
    colors = [NEG if v > 20 else PRIMARY for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(20, color=NEG, ls="--", lw=1)
    ax.text(20.5, -0.6, "ngưỡng 20%", color=NEG, fontsize=8)
    ax.set_xlim(0, max(25, max(vals) * 1.15))
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_title("Tỷ lệ giá trị thiếu theo cột (dữ liệu thô)")
    ax.grid(axis="y", visible=False)
    return _chart("missing", _save(fig, out_dir, "missing"), "Tỷ lệ giá trị thiếu theo cột",
                  "Hình: Tỷ lệ % ô trống trên từng cột của dữ liệu thô; cột đỏ vượt ngưỡng 20%.", 2, "barh",
                  {"top": dict(data), "n_categories": len(data)})


def distribution(df: pd.DataFrame, col: str, out_dir: str) -> dict:
    x = df[col].dropna().astype(float)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(x, bins=min(40, max(10, int(np.sqrt(len(x))))), color=PRIMARY, edgecolor="white")
    ax.axvline(x.mean(), color=ACCENT, lw=1.6, label=f"Trung bình {_fmt(x.mean())}")
    ax.axvline(x.median(), color=GREY, lw=1.6, ls="--", label=f"Trung vị {_fmt(x.median())}")
    ax.set_ylim(bottom=0)
    ax.set_xlabel(col)
    ax.set_ylabel("Số bản ghi")
    ax.legend(frameon=False)
    ax.set_title(f"Phân phối {col}")
    skew = float(x.skew())
    return _chart(f"dist_{col}", _save(fig, out_dir, f"dist_{col}"), f"Phân phối {col}",
                  f"Hình: Phân phối của {col} sau làm sạch (n={len(x):,}).", 3, "hist",
                  {"mean": round(x.mean(), 2), "median": round(x.median(), 2), "skew": round(skew, 2), "n": len(x)})


def trend(df: pd.DataFrame, date_col: str, kpi: str, out_dir: str, agg: str = "sum") -> dict | None:
    d = df[[date_col, kpi]].dropna()
    if d.empty:
        return None
    span_days = (d[date_col].max() - d[date_col].min()).days
    freq, label = ("W", "tuần") if span_days < 120 else ("MS", "tháng")
    s = d.set_index(date_col)[kpi].resample(freq).agg(agg)
    if len(s) < 3:
        return None
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(s.index, s.values, color=PRIMARY, lw=2, marker="o", ms=3)
    ax.set_ylim(bottom=0 if s.min() >= 0 else None)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: _fmt(v)))
    peak, low = s.idxmax(), s.idxmin()
    ax.annotate(f"Đỉnh {_fmt(s.max())}", (peak, s.max()), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=PRIMARY)
    ax.annotate(f"Đáy {_fmt(s.min())}", (low, s.min()), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=NEG)
    ax.set_title(f"Xu hướng {kpi} theo {label} ({'tổng' if agg == 'sum' else 'trung bình'})")
    ax.grid(axis="x", visible=False)
    half = len(s) // 2
    first, second = float(s.iloc[:half].mean()), float(s.iloc[half:].mean())
    change = round(100 * (second - first) / first, 2) if first else None
    return _chart("trend", _save(fig, out_dir, "trend"), f"Xu hướng {kpi} theo thời gian",
                  f"Hình: {kpi} ({'tổng' if agg == 'sum' else 'trung bình'}) theo {label}, {s.index.min():%m/%Y}–{s.index.max():%m/%Y}.", 3, "line",
                  {"periods": len(s), "peak": str(peak.date()), "peak_value": round(float(s.max()), 2), "low": str(low.date()),
                   "low_value": round(float(s.min()), 2), "first_half_avg": round(first, 2), "second_half_avg": round(second, 2),
                   "half_over_half_change_pct": change, "freq": label})


def category_bar(df: pd.DataFrame, cat: str, kpi: str, out_dir: str, agg: str = "sum") -> dict:
    g = df.groupby(cat, observed=True)[kpi].agg(agg).sort_values(ascending=False)
    top = g.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(top) + 1.3))
    ax.barh(top.index.astype(str), top.values, color=[ACCENT if i == len(top) - 1 else PRIMARY for i in range(len(top))])
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: _fmt(v)))
    for i, v in enumerate(top.values):
        ax.text(v, i, f" {_fmt(v)}", va="center", fontsize=8)
    ax.set_title(f"{kpi} ({'tổng' if agg == 'sum' else 'trung bình'}) theo {cat}" + (" — top 10" if len(g) > 10 else ""))
    ax.grid(axis="y", visible=False)
    share = (g / g.sum() * 100).round(2) if agg == "sum" and g.sum() else None
    return _chart(f"cat_{cat}", _save(fig, out_dir, f"cat_{cat}"), f"{kpi} theo {cat}",
                  f"Hình: So sánh {kpi} giữa các nhóm {cat}; cột cam là nhóm dẫn đầu.", 3, "barh",
                  {"values": {str(k): round(float(v), 2) for k, v in g.head(10).items()},
                   "share_pct": {str(k): float(v) for k, v in share.head(10).items()} if share is not None else None,
                   "n_categories": int(min(len(g), 10)), "total_groups": int(len(g))})


def corr_heatmap(df: pd.DataFrame, cols: list[str], out_dir: str) -> dict | None:
    cols = cols[:10]
    if len(cols) < 3:
        return None
    c = df[cols].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 2.5, 0.65 * len(cols) + 1.8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(c.values[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Ma trận tương quan giữa các biến số")
    pairs = [(cols[i], cols[j], round(float(c.values[i, j]), 2)) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    pairs.sort(key=lambda p: -abs(p[2]))
    return _chart("corr", _save(fig, out_dir, "corr"), "Ma trận tương quan",
                  "Hình: Mức độ đi cùng nhau giữa các biến số (−1: ngược chiều, 1: cùng chiều).", 3, "heatmap",
                  {"strongest_pairs": pairs[:5]})


def class_balance(y: pd.Series, out_dir: str) -> dict:
    vc = y.astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(vc.index, vc.values, color=PRIMARY)
    ax.set_ylim(bottom=0)
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{v:,} ({100*v/vc.sum():.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"Phân bố nhãn {y.name}")
    ax.grid(axis="x", visible=False)
    return _chart("class_balance", _save(fig, out_dir, "class_balance"), f"Phân bố nhãn {y.name}",
                  f"Hình: Số lượng bản ghi theo từng nhãn của {y.name}.", 3, "bar",
                  {"counts": vc.to_dict(), "n_categories": len(vc)})


def importance_bar(imp: pd.Series, out_dir: str, metric_label: str) -> dict:
    imp = imp[imp >= 0.01 * imp.max()] if imp.max() > 0 else imp.head(5)  # bỏ biến ảnh hưởng ~0
    top = imp.sort_values(ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(top) + 1.3))
    ax.barh(top.index, top.values, color=[ACCENT if v == top.max() else PRIMARY for v in top.values])
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Mức giảm {metric_label} khi xáo trộn biến")
    ax.set_title("Yếu tố tác động mạnh nhất")
    ax.grid(axis="y", visible=False)
    return _chart("importance", _save(fig, out_dir, "importance"), "Yếu tố tác động mạnh nhất",
                  "Hình: Mức độ đóng góp của từng yếu tố vào khả năng dự báo (đo trên tập kiểm tra).", 4, "barh",
                  {"ranking": {k: round(float(v), 4) for k, v in top[::-1].items()}, "n_categories": len(top)})


def metric_tree_bar(contrib: dict, out_dir: str, kpi: str) -> dict:
    names = list(contrib["components"].keys())
    vals = [contrib["components"][k] for k in names]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.bar(names + [f"Tổng {kpi}"], vals + [contrib["total_change_pct"]],
           color=[NEG if v < 0 else PRIMARY for v in vals] + [GREY])
    ax.axhline(0, color="#444444", lw=0.8)
    for i, v in enumerate(vals + [contrib["total_change_pct"]]):
        ax.text(i, v, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylabel("% thay đổi (log-đóng góp)")
    ax.set_title(f"Phân rã thay đổi {kpi}: kỳ sau so với kỳ trước")
    ax.grid(axis="x", visible=False)
    return _chart("metric_tree", _save(fig, out_dir, "metric_tree"), f"Phân rã chỉ số {kpi}",
                  f"Hình: {kpi} = Số giao dịch × Giá trị trung bình/giao dịch; đóng góp của từng thành phần vào thay đổi tổng.", 4, "bar",
                  {**contrib, "n_categories": len(names) + 1})
