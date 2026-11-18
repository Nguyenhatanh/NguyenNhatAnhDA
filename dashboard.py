"""
Bảng điều khiển 2 tầng.
- Tầng I  (tính toán): dữ liệu nguồn → bảng tổng hợp → bảng chỉ số cốt lõi → bảng tham số.
- Tầng II (trực quan): biểu đồ Plotly có hiệu ứng chuyển động và cửa sổ thông số khi rê chuột.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

NAVY, ORANGE, GREY, RED, GREEN, AMBER = "#1F4E79", "#E07A1F", "#9AA5B1", "#C0392B", "#2E7D32", "#B7791F"
FONT = "Montserrat, sans-serif"


# ============================================================ TẦNG I: TÍNH TOÁN
def build_params(df: pd.DataFrame, mplan: dict, types: dict) -> dict:
    """Bảng điều kiện & tham số: mốc so sánh, ngưỡng cảnh báo, cột phân khúc."""
    dc, kpi = mplan.get("date_col"), mplan.get("kpi")
    segs = [c for c in mplan.get("segment_cols", []) if c in df.columns]
    p = {"date_col": dc, "kpi": kpi, "segments": segs,
         "nguong_giam_do": -10.0, "nguong_giam_vang": 0.0}  # % tăng trưởng tháng/tháng
    if dc and dc in df:
        p["ngay_dau"], p["ngay_cuoi"] = df[dc].min(), df[dc].max()
    return p


def apply_filters(df: pd.DataFrame, p: dict, date_range=None, seg_filters: dict | None = None) -> pd.DataFrame:
    d = df
    if date_range and p.get("date_col"):
        a, b = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1)
        d = d[(d[p["date_col"]] >= a) & (d[p["date_col"]] < b)]
    for col, vals in (seg_filters or {}).items():
        if vals:
            d = d[d[col].astype(str).isin(vals)]
    return d


def summary_tables(d: pd.DataFrame, p: dict) -> dict:
    """Bảng tổng hợp theo thời gian và theo từng phân khúc."""
    kpi, dc = p["kpi"], p["date_col"]
    out = {}
    if kpi and dc and not d.empty:
        m = d.set_index(dc).resample("MS").agg(**{"Doanh số": (kpi, "sum"), "Số đơn": (kpi, "size")})
        m["Giá trị TB/đơn"] = (m["Doanh số"] / m["Số đơn"].replace(0, np.nan)).round(0)
        m["Tăng trưởng tháng (%)"] = (m["Doanh số"].pct_change() * 100).round(2)
        m.index = m.index.strftime("%m/%Y")
        m.index.name = "Tháng"
        out["theo_thang"] = m
    for s in p["segments"]:
        if kpi and not d.empty:
            g = d.groupby(s, observed=True)[kpi].agg(["sum", "size"]).rename(columns={"sum": "Doanh số", "size": "Số đơn"})
            g["Tỷ trọng (%)"] = (100 * g["Doanh số"] / g["Doanh số"].sum()).round(2)
            g["Giá trị TB/đơn"] = (g["Doanh số"] / g["Số đơn"]).round(0)
            out[f"theo_{s}"] = g.sort_values("Doanh số", ascending=False)
    return out


def kpi_table(d: pd.DataFrame, p: dict, tables: dict) -> pd.DataFrame:
    """Bảng chỉ số cốt lõi, tính sẵn kèm so sánh kỳ trước."""
    kpi = p["kpi"]
    rows = []
    if not kpi or d.empty:
        return pd.DataFrame(columns=["Chỉ số", "Giá trị", "Kỳ trước", "Thay đổi (%)", "Ghi chú"])
    m = tables.get("theo_thang")
    cur = prev = None
    if m is not None and len(m) >= 2:
        cur, prev = m.iloc[-1], m.iloc[-2]

    def chg(a, b):
        return round(100 * (a - b) / b, 2) if b else None

    rows.append(["Tổng doanh số", float(d[kpi].sum()), None, None, "Toàn bộ giai đoạn đã lọc"])
    rows.append(["Số đơn hàng", float(len(d)), None, None, "Toàn bộ giai đoạn đã lọc"])
    rows.append(["Giá trị trung bình/đơn", float(d[kpi].mean()), None, None, "Doanh số ÷ số đơn"])
    if cur is not None:
        rows.append([f"Doanh số tháng {m.index[-1]}", float(cur["Doanh số"]), float(prev["Doanh số"]), chg(cur["Doanh số"], prev["Doanh số"]), f"So với tháng {m.index[-2]}"])
        rows.append([f"Giá trị TB/đơn tháng {m.index[-1]}", float(cur["Giá trị TB/đơn"]), float(prev["Giá trị TB/đơn"]), chg(cur["Giá trị TB/đơn"], prev["Giá trị TB/đơn"]), f"So với tháng {m.index[-2]}"])
    if m is not None and len(m) >= 13:
        rows.append(["Tăng trưởng cùng kỳ năm trước", float(m.iloc[-1]["Doanh số"]), float(m.iloc[-13]["Doanh số"]),
                     chg(m.iloc[-1]["Doanh số"], m.iloc[-13]["Doanh số"]), f"So với tháng {m.index[-13]}"])
    else:
        rows.append(["Tăng trưởng cùng kỳ năm trước", None, None, None, "Chưa đủ 13 tháng dữ liệu"])
    rows.append(["Tỷ lệ chuyển đổi", None, None, None, "Dữ liệu không có số lượt truy cập/khách tiềm năng"])
    return pd.DataFrame(rows, columns=["Chỉ số", "Giá trị", "Kỳ trước", "Thay đổi (%)", "Ghi chú"])


def alerts(tables: dict, p: dict) -> list[dict]:
    """Vùng cảnh báo: đỏ/vàng/xanh theo ngưỡng tăng trưởng."""
    out = []
    m = tables.get("theo_thang")
    if m is not None and len(m) >= 2:
        g = m["Tăng trưởng tháng (%)"].iloc[-1]
        lvl = "do" if g <= p["nguong_giam_do"] else "vang" if g < p["nguong_giam_vang"] else "xanh"
        out.append({"muc": lvl, "noi_dung": f"Doanh số tháng {m.index[-1]} giảm {vn(-g)}% so với tháng trước." if g < 0 else f"Doanh số tháng {m.index[-1]} tăng {vn(g)}% so với tháng trước."})
        a = 100 * (m["Giá trị TB/đơn"].iloc[-1] - m["Giá trị TB/đơn"].iloc[-2]) / m["Giá trị TB/đơn"].iloc[-2]
        out.append({"muc": "do" if a <= -10 else "vang" if a < 0 else "xanh", "noi_dung": (f"Giá trị trung bình/đơn giảm {vn(-a)}% so với tháng trước." if a < 0 else f"Giá trị trung bình/đơn tăng {vn(a)}% so với tháng trước.")})
    for s in p["segments"][:2]:
        t = tables.get(f"theo_{s}")
        if t is not None and len(t) >= 2:
            low = t.index[-1]
            out.append({"muc": "vang", "noi_dung": f"{s}: nhóm “{low}” đóng góp thấp nhất ({vn(t['Tỷ trọng (%)'].iloc[-1])}% doanh số)."})
    return out


# ============================================================ TẦNG II: TRỰC QUAN
def vn(v: float, dec: int = 1) -> str:
    return f"{v:,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def money(v: float) -> str:
    a = abs(v)
    return f"{vn(v/1e9, 1)} tỷ" if a >= 1e9 else f"{vn(v/1e6, 0)} tr" if a >= 1e6 else vn(v, 0)


def money_axis(fig: go.Figure, vmax: float, axis: str = "y") -> go.Figure:
    """Nhãn trục theo đơn vị tỷ/triệu thay vì ký hiệu B/M."""
    if not vmax or vmax <= 0:
        return fig
    step = 10 ** np.floor(np.log10(vmax / 4))
    for m in (1, 2, 2.5, 5, 10):
        if vmax / (step * m) <= 5:
            step *= m
            break
    ticks = np.arange(0, vmax * 1.05 + step, step)
    upd = dict(tickvals=ticks, ticktext=[money(t) if t else "0" for t in ticks])
    (fig.update_yaxes if axis == "y" else fig.update_xaxes)(**upd)
    return fig


def _layout(fig: go.Figure, title: str, h: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(family=FONT, size=16, color="#16263A"), x=0, xanchor="left"),
        font=dict(family=FONT, size=13, color="#2B3A4A"), template="plotly_white", height=h,
        margin=dict(l=10, r=10, t=56, b=10), separators=",.",
        hoverlabel=dict(bgcolor="white", bordercolor=NAVY, font=dict(family=FONT, size=13)),
        transition=dict(duration=600, easing="cubic-in-out"), showlegend=False,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor="#D5DDE6")
    fig.update_yaxes(gridcolor="#EEF1F5", zeroline=False)
    return fig


def fig_trend(m: pd.DataFrame, kpi_label: str) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=m.index, y=m["Doanh số"], mode="lines+markers", fill="tozeroy", line=dict(color=NAVY, width=3, shape="spline"),
        fillcolor="rgba(31,78,121,0.10)", marker=dict(size=7, color=NAVY), customdata=np.stack([m["Số đơn"], m["Giá trị TB/đơn"], m["Tăng trưởng tháng (%)"].fillna(0)], axis=-1),
        hovertemplate="<b>Tháng %{x}</b><br>Doanh số: %{y:,.0f}<br>Số đơn: %{customdata[0]:,.0f}<br>Giá trị TB/đơn: %{customdata[1]:,.0f}<br>So với tháng trước: %{customdata[2]:+.1f}%<extra></extra>"))
    fig.update_yaxes(rangemode="tozero")
    money_axis(fig, float(m["Doanh số"].max()), "y")
    return _layout(fig, "Doanh số theo tháng có tăng hay giảm?")


def fig_bar(t: pd.DataFrame, seg: str) -> go.Figure:
    t = t.head(10).iloc[::-1]
    colors = [ORANGE if i == len(t) - 1 else NAVY for i in range(len(t))]
    fig = go.Figure(go.Bar(
        x=t["Doanh số"], y=t.index.astype(str), orientation="h", marker=dict(color=colors),
        customdata=np.stack([t["Tỷ trọng (%)"], t["Số đơn"], t["Giá trị TB/đơn"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Doanh số: %{x:,.0f}<br>Tỷ trọng: %{customdata[0]:.1f}%<br>Số đơn: %{customdata[1]:,.0f}<br>Giá trị TB/đơn: %{customdata[2]:,.0f}<extra></extra>"))
    fig.update_xaxes(rangemode="tozero")
    money_axis(fig, float(t["Doanh số"].max()), "x")
    return _layout(fig, f"Nhóm {seg} nào mang lại doanh số cao nhất?")


def fig_donut(t: pd.DataFrame, seg: str) -> go.Figure:
    t = t.head(6)
    palette = [NAVY, ORANGE, "#4F7CAC", "#F2B880", GREY, "#7C8B9C"]
    fig = go.Figure(go.Pie(
        labels=t.index.astype(str), values=t["Doanh số"], hole=0.62, sort=False, marker=dict(colors=palette[: len(t)], line=dict(color="white", width=2)),
        textinfo="percent", hovertemplate="<b>%{label}</b><br>Doanh số: %{value:,.0f}<br>Tỷ trọng: %{percent}<extra></extra>"))
    fig = _layout(fig, f"Cơ cấu doanh số theo {seg}")
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.05))
    return fig


# --------------------------- biểu đồ tương tác cho các bước 2–5 (thay ảnh tĩnh)
def interactive(chart: dict, S: dict) -> go.Figure | None:
    """Dựng lại biểu đồ Plotly từ dữ liệu gốc theo mã biểu đồ; None nếu không dựng được."""
    cid = chart["id"]
    df = S.get("clean", {}).get("df")
    mp = S.get("mplan", {})
    kpi, dc = mp.get("kpi"), mp.get("date_col")
    try:
        if cid == "missing":
            data = sorted([(c["column"], c["missing_pct"], c["missing"]) for c in S["prof"]["columns"] if c["missing_pct"] > 0], key=lambda x: x[1])[-10:]
            n, v, k = zip(*data)
            fig = go.Figure(go.Bar(x=v, y=n, orientation="h", marker_color=[RED if x > 20 else NAVY for x in v], customdata=k,
                                   hovertemplate="<b>%{y}</b><br>Tỷ lệ thiếu: %{x:.1f}%<br>Số ô trống: %{customdata:,}<extra></extra>"))
            fig.add_vline(x=20, line_dash="dash", line_color=RED, annotation_text="Ngưỡng 20%")
            fig.update_xaxes(range=[0, max(25, max(v) * 1.15)], ticksuffix="%")
            return _layout(fig, "Cột nào thiếu dữ liệu nhiều nhất?")
        if cid.startswith("dist_"):
            col = cid[5:]
            x = df[col].dropna()
            fig = go.Figure(go.Histogram(x=x, nbinsx=40, marker_color=NAVY, hovertemplate="Khoảng: %{x}<br>Số đơn: %{y:,}<extra></extra>"))
            fig.add_vline(x=x.mean(), line_color=ORANGE, annotation_text="Trung bình")
            fig.add_vline(x=x.median(), line_color=GREY, line_dash="dash", annotation_text="Trung vị")
            return _layout(fig, f"Giá trị {col} phân bố như thế nào?")
        if cid == "trend" and kpi and dc:
            p = {"kpi": kpi, "date_col": dc, "segments": []}
            return fig_trend(summary_tables(df, p)["theo_thang"], kpi)
        if cid.startswith("cat_") and kpi:
            seg = cid[4:]
            t = summary_tables(df, {"kpi": kpi, "date_col": None, "segments": [seg]})[f"theo_{seg}"]
            return fig_bar(t, seg)
        if cid == "corr":
            pairs = chart["facts"].get("strongest_pairs", [])
            cols = sorted({a for a, b, _ in pairs} | {b for a, b, _ in pairs})
            c = df[cols].astype(float).corr(method="spearman")
            fig = go.Figure(go.Heatmap(z=c.values, x=cols, y=cols, zmin=-1, zmax=1, colorscale="RdBu_r", text=np.round(c.values, 2), texttemplate="%{text}",
                                       hovertemplate="%{y} và %{x}<br>Mức tương quan: %{z:.2f}<extra></extra>"))
            return _layout(fig, "Các biến số đi cùng nhau ra sao?", 420)
        if cid == "class_balance":
            cnt = chart["facts"]["counts"]
            fig = go.Figure(go.Bar(x=list(cnt), y=list(cnt.values()), marker_color=NAVY, hovertemplate="Nhãn %{x}: %{y:,} bản ghi<extra></extra>"))
            return _layout(fig, "Các nhãn có cân bằng không?")
        if cid == "metric_tree":
            f = chart["facts"]
            names = list(f["components"]) + ["Tổng doanh số"]
            vals = list(f["components"].values()) + [f["total_change_pct"]]
            fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[RED if v < 0 else NAVY for v in vals[:-1]] + [GREY], text=[f"{v:+.1f}%" for v in vals], textposition="outside",
                                   hovertemplate="<b>%{x}</b><br>Thay đổi: %{y:+.2f}%<extra></extra>"))
            fig.add_hline(y=0, line_color="#444")
            fig.update_yaxes(ticksuffix="%")
            return _layout(fig, "Doanh số thay đổi do số đơn hay do giá trị mỗi đơn?")
        if cid == "importance":
            r = chart["facts"]["ranking"]
            items = list(r.items())[::-1]
            fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                                   marker_color=[ORANGE if v == max(r.values()) else NAVY for _, v in items],
                                   hovertemplate="<b>%{y}</b><br>Mức đóng góp vào khả năng dự báo: %{x:.3f}<extra></extra>"))
            fig.update_xaxes(rangemode="tozero")
            return _layout(fig, "Yếu tố nào tác động mạnh nhất?")
    except Exception:
        return None
    return None
