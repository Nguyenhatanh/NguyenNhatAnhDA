"""BƯỚC 2 — Thu thập dữ liệu & Đánh giá chất lượng (Data Profiling). 100% tính bằng Pandas."""
from __future__ import annotations

import io
import re
import warnings

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .llm import LLMClient, to_facts

MONEY_QTY_HINTS = ("revenue", "doanh", "amount", "price", "gia", "giá", "qty", "quantity", "so_luong", "số lượng",
                   "sales", "total", "tien", "tiền", "cost", "chi_phi", "profit", "loi_nhuan", "value", "spend")
DATE_HINTS = ("date", "ngay", "ngày", "time", "thoi_gian", "created", "updated", "_at", "month", "thang")
ID_HINTS = ("id", "ma_", "mã", "code", "key", "uuid")


# ------------------------------------------------------------ đọc dữ liệu
def load_data(name: str, data: bytes) -> pd.DataFrame:
    ext = name.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xlsm", "xls"):
        return pd.read_excel(io.BytesIO(data))
    last_exc = None
    for enc in ("utf-8-sig", "utf-8", "cp1258", "cp1252", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc, sep=None, engine="python")
        except Exception as exc:  # thử encoding kế tiếp
            last_exc = exc
    raise ValueError(f"Không đọc được tệp CSV: {last_exc}")


# ------------------------------------------------------- suy luận kiểu dữ liệu
def is_texty(s: pd.Series) -> bool:
    return not (ptypes.is_numeric_dtype(s) or ptypes.is_datetime64_any_dtype(s) or ptypes.is_bool_dtype(s))


def _parse_number(v) -> float:
    """'1.234.567 ₫' / '1,234.5' / '12,5' / '(300)' -> float. Không parse được -> NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if not s:
        return np.nan
    neg = s.startswith("(") and s.endswith(")")
    s = re.sub(r"(?i)vnd|vnđ|usd|đồng|dong|₫|đ|\$|€|%|\s|\(|\)", "", s)
    if re.search(r"[^\d,.\-+]", s):  # còn chữ cái => không phải số (ví dụ mã 'DH00012')
        return np.nan
    s = s.replace("+", "")
    if not re.search(r"\d", s):
        return np.nan
    if "," in s and "." in s:  # ký tự xuất hiện sau cùng là dấu thập phân
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif s.count(".") > 1 or re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return -abs(x) if neg else x


def coerce_numeric(s: pd.Series) -> pd.Series:
    return s.map(_parse_number).astype(float)


DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d/%m/%Y", "%d/%m/%Y %H:%M", "%d-%m-%Y", "%d.%m.%Y",
                "%m/%d/%Y", "%Y%m%d")


def coerce_datetime(s: pd.Series) -> pd.Series:
    """Ưu tiên định dạng tường minh (tránh đảo ngày/tháng với ISO); ưu tiên dd/mm (chuẩn VN) khi hoà."""
    if ptypes.is_datetime64_any_dtype(s):
        return s
    txt = s.astype("string").str.strip()
    nonnull = txt.notna().sum() or 1
    best, best_ok = None, 0
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(txt, format=fmt, errors="coerce")
        ok = parsed.notna().sum()
        if ok > best_ok:  # '>' nghiêm ngặt => giữ định dạng đứng trước khi hoà
            best, best_ok = parsed, ok
    if best is not None and best_ok / nonnull >= 0.9:
        return best
    best, best_ok = None, -1
    for dayfirst in (True, False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
            except (TypeError, ValueError):
                parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        ok = parsed.notna().sum()
        if ok > best_ok:
            best, best_ok = parsed, ok
    return best


def infer_type(s: pd.Series, name: str) -> str:
    n = s.notna().sum()
    lname = name.lower()
    if n == 0:
        return "empty"
    if ptypes.is_bool_dtype(s):
        return "boolean"
    if ptypes.is_datetime64_any_dtype(s):
        return "datetime"
    if ptypes.is_numeric_dtype(s):
        if any(h in lname for h in ID_HINTS) and s.nunique() > 0.9 * n:
            return "id"
        if set(pd.unique(s.dropna())) <= {0, 1}:
            return "boolean"
        return "numeric"
    nonnull = s.dropna().astype(str)
    sample = nonnull.sample(min(len(nonnull), 2000), random_state=0)
    if any(h in lname for h in DATE_HINTS) or sample.str.contains(r"\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}", regex=True).mean() > 0.6:
        if coerce_datetime(sample).notna().mean() >= 0.9:
            return "datetime_as_text"
    if coerce_numeric(sample).notna().mean() >= 0.9:
        return "numeric_as_text"
    if any(h in lname for h in ID_HINTS) and s.nunique() > 0.9 * n:
        return "id"
    if sample.str.lower().isin(["true", "false", "yes", "no", "có", "không", "y", "n"]).mean() > 0.95:
        return "boolean"
    if s.nunique() <= max(50, 0.05 * n):
        return "categorical"
    return "text"


def is_money_like(name: str) -> bool:
    return any(h in name.lower() for h in MONEY_QTY_HINTS)


def numeric_view(df: pd.DataFrame, types: dict) -> dict[str, pd.Series]:
    """Trả về các cột số (kể cả cột số bị lưu dạng chữ) đã ép kiểu."""
    out = {}
    for c, t in types.items():
        if t == "numeric":
            out[c] = df[c].astype(float)
        elif t == "numeric_as_text":
            out[c] = coerce_numeric(df[c])
    return out


def datetime_view(df: pd.DataFrame, types: dict) -> dict[str, pd.Series]:
    out = {}
    for c, t in types.items():
        if t == "datetime":
            out[c] = pd.to_datetime(df[c], errors="coerce")
        elif t == "datetime_as_text":
            out[c] = coerce_datetime(df[c])
    return out


def iqr_bounds(x: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def detect_date_pairs(dt_cols: list[str]) -> list[tuple[str, str]]:
    """Tìm cặp (ngày trước, ngày sau) theo tên: order/đặt -> ship/giao/delivery."""
    early = [c for c in dt_cols if re.search(r"order|dat|đặt|create|start|bat_dau|purchase|mua", c.lower())]
    late = [c for c in dt_cols if re.search(r"ship|giao|deliver|end|ket_thuc|complete|close|return|tra", c.lower())]
    return [(a, b) for a in early for b in late if a != b]


# ----------------------------------------------------------------- profiling
def profile(df: pd.DataFrame, today: pd.Timestamp | None = None) -> dict:
    today = today or pd.Timestamp.today().normalize()
    n = len(df)
    types = {c: infer_type(df[c], c) for c in df.columns}
    nums, dts = numeric_view(df, types), datetime_view(df, types)

    cols = []
    for c in df.columns:
        miss = int(df[c].isna().sum() + (df[c].astype(str).str.strip() == "").sum() if is_texty(df[c]) else df[c].isna().sum())
        info = {
            "column": c, "pandas_dtype": str(df[c].dtype), "inferred_type": types[c],
            "missing": miss, "missing_pct": round(100 * miss / n, 2) if n else 0.0,
            "unique": int(df[c].nunique(dropna=True)),
            "sample": [str(v) for v in df[c].dropna().unique()[:3]],
        }
        if c in nums:
            x = nums[c].dropna()
            if len(x):
                lo, hi = iqr_bounds(x)
                info.update({
                    "mean": round(float(x.mean()), 4), "median": round(float(x.median()), 4),
                    "min": round(float(x.min()), 4), "max": round(float(x.max()), 4),
                    "outliers_iqr": int(((x < lo) | (x > hi)).sum()),
                    "negatives": int((x < 0).sum()),
                })
                if types[c] == "numeric_as_text":
                    info["unparseable"] = int(df[c].notna().sum() - nums[c].notna().sum())
        if c in dts:
            d = dts[c]
            info.update({"min_date": str(d.min())[:10], "max_date": str(d.max())[:10],
                         "future_dates": int((d > today).sum()),
                         "unparseable": int(df[c].notna().sum() - d.notna().sum())})
        cols.append(info)

    # ------- "Bệnh" của dữ liệu
    issues = []
    critical_missing = [ci for ci in cols if ci["missing_pct"] > 20]
    for ci in critical_missing:
        issues.append({"column": ci["column"], "type": "missing_critical", "severity": "high",
                       "detail": f"Thiếu {ci['missing_pct']}% ({ci['missing']} ô)"})
    for ci in cols:
        if ci["inferred_type"] in ("numeric_as_text", "datetime_as_text"):
            issues.append({"column": ci["column"], "type": "wrong_dtype", "severity": "medium",
                           "detail": f"Lưu dạng chữ ({ci['pandas_dtype']}) nhưng bản chất là "
                                     f"{'số' if ci['inferred_type'] == 'numeric_as_text' else 'ngày tháng'}"
                                     + (f"; {ci.get('unparseable', 0)} giá trị không ép kiểu được" if ci.get("unparseable") else "")})
        if ci.get("negatives", 0) and is_money_like(ci["column"]):
            issues.append({"column": ci["column"], "type": "negative_values", "severity": "high",
                           "detail": f"{ci['negatives']} giá trị âm ở cột tiền/số lượng"})
        if ci.get("outliers_iqr", 0) and n and ci["outliers_iqr"] / n > 0.005:
            issues.append({"column": ci["column"], "type": "outliers", "severity": "medium",
                           "detail": f"{ci['outliers_iqr']} điểm vượt ngưỡng tứ phân vị ×1,5 (nhỏ nhất={ci['min']}, lớn nhất={ci['max']}, trung vị={ci['median']})"})
        if ci.get("future_dates", 0):
            issues.append({"column": ci["column"], "type": "future_dates", "severity": "high",
                           "detail": f"{ci['future_dates']} ngày ở tương lai so với {today.date()}"})
    dup = int(df.duplicated().sum())
    if dup:
        issues.append({"column": "(toàn bảng)", "type": "duplicates", "severity": "medium", "detail": f"{dup} dòng trùng lặp hoàn toàn"})
    logic = []
    for a, b in detect_date_pairs(list(dts)):
        bad = int((dts[b] < dts[a]).sum())
        logic.append({"rule": f"{b} >= {a}", "violations": bad})
        if bad:
            issues.append({"column": f"{a}/{b}", "type": "logic", "severity": "high", "detail": f"{bad} dòng có {b} trước {a}"})

    # ------- Điểm chất lượng (1-10): trừ điểm theo từng nhóm, có trần
    avg_missing = float(np.mean([ci["missing_pct"] for ci in cols])) if cols else 0
    penalty = (
        min(3.0, avg_missing / 10)
        + min(2.0, 0.5 * len(critical_missing))
        + min(1.5, 20 * dup / n if n else 0)
        + min(1.5, 0.3 * sum(i["type"] == "wrong_dtype" for i in issues))
        + min(1.0, 0.3 * sum(i["type"] in ("negative_values", "outliers") for i in issues))
        + min(1.0, 0.5 * sum(i["type"] in ("future_dates", "logic") for i in issues))
    )
    score = round(max(1.0, min(10.0, 10 - penalty)), 1)
    risk = ("Thấp — có thể phân tích sau khi làm sạch nhẹ" if score >= 8 else
            "Trung bình — cần làm sạch có kiểm soát trước khi phân tích" if score >= 5 else
            "Cao — phân tích ngay sẽ cho kết luận sai lệch")
    sev_rank = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: sev_rank[i["severity"]])
    return {
        "overview": {"rows": n, "columns": df.shape[1], "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
                     "duplicates": dup, "avg_missing_pct": round(avg_missing, 2), "profiled_at": str(today.date())},
        "columns": cols, "types": types, "issues": issues, "logic_checks": logic,
        "quality_score": score, "risk_comment": risk,
    }


def commentary(llm: LLMClient, prof: dict, biz: dict) -> str:
    facts = to_facts({"overview": prof["overview"], "issues": prof["issues"][:25], "score": prof["quality_score"]})
    fb = f"Điểm chất lượng {prof['quality_score']}/10 — rủi ro {prof['risk_comment'].lower()}. Có {len(prof['issues'])} vấn đề được phát hiện."
    return llm.text(
        "Bạn là Data Quality Engineer. Dựa DUY NHẤT trên kết quả quét sau, viết 4-6 gạch đầu dòng nhận xét rủi ro "
        "nghiệp vụ nếu đem dữ liệu đi phân tích ngay (liên hệ bối cảnh doanh nghiệp). Không thêm số mới.\n"
        f"BỐI CẢNH: {to_facts(biz, 2500)}\nKẾT QUẢ QUÉT: {facts}",
        fallback=fb,
    )
