"""
Lớp giao tiếp LLM (Google Gemini) + vòng lặp Generator/Critic.

Nguyên tắc:
- LLM KHÔNG BAO GIỜ được tự tính số liệu. Mọi con số đưa vào prompt đều đến từ Pandas/Scipy.
- Mọi lệnh gọi JSON đều có `fallback` => nếu API lỗi/mất mạng, pipeline vẫn chạy bằng luật xác định.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from typing import Any

SYSTEM_BASE = (
    "Bạn là AI Business Analyst Agent cẩn trọng. Tuyệt đối không bịa số liệu: chỉ dùng các con số "
    "có trong dữ liệu đầu vào. Nếu thông tin chưa được cung cấp, ghi rõ 'Chưa cung cấp'. "
    "Viết hoàn toàn bằng tiếng Việt chuẩn mực, không chèn từ tiếng Anh (dùng: chỉ số thay KPI, nhận định thay insight, giá trị trung bình mỗi đơn thay AOV). "
    "Văn phong phân tích nghiệp vụ: đi thẳng vào trọng tâm, từ hiện tượng đến nguyên nhân rồi đến hành động; không dùng từ cảm tính hay ẩn dụ."
)


class LLMClient:
    """Bọc Gemini API. Tự chuyển sang chế độ offline (mock) nếu không có API key."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        self.mock = os.getenv("BA_AGENT_MOCK") == "1" or not self.api_key
        self.trace: list[dict] = []  # Nhật ký từng lần gọi (phục vụ audit)
        self.last_error: str | None = None
        self._client = None
        if not self.mock:
            try:
                from google import genai  # import trễ để app vẫn chạy khi thiếu SDK

                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:  # SDK thiếu hoặc key sai định dạng
                self.last_error = f"Không khởi tạo được Gemini client: {exc}"
                self.mock = True

    # ------------------------------------------------------------------ core
    def _call(self, prompt: str, system: str | None, json_mode: bool, temperature: float, retries: int = 2) -> str:
        if self.mock:
            return ""
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system or SYSTEM_BASE,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        delay = 2.0
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                resp = self._client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                text = (resp.text or "").strip()
                self.trace.append({"ok": True, "chars_in": len(prompt), "chars_out": len(text), "sec": round(time.time() - t0, 2)})
                self.last_error = None
                return text
            except Exception as exc:  # lỗi mạng / quota / key
                self.last_error = str(exc)[:300]
                self.trace.append({"ok": False, "error": self.last_error, "attempt": attempt})
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2
        return ""

    def text(self, prompt: str, system: str | None = None, fallback: str = "", temperature: float = 0.3) -> str:
        out = self._call(prompt, system, json_mode=False, temperature=temperature)
        return out or fallback

    def json(self, prompt: str, system: str | None = None, fallback: Any = None, temperature: float = 0.2) -> Any:
        raw = self._call(prompt, system, json_mode=True, temperature=temperature)
        parsed = parse_json(raw)
        if parsed is None:
            return copy.deepcopy(fallback)
        return parsed

    # --------------------------------------------------- Generator ⇄ Critic
    def critique_revise(self, draft: Any, task: str, criteria: list[str], facts: str, system: str | None = None) -> tuple[Any, list[str]]:
        """Critic soi bản nháp theo tiêu chí; nếu có lỗi => bắt Generator sửa (ít nhất 1 vòng)."""
        if self.mock:
            return draft, []
        prompt = (
            f"VAI TRÒ: Critic/Validator độc lập.\nNHIỆM VỤ GỐC: {task}\n"
            f"DỮ KIỆN ĐÃ TÍNH BẰNG CODE (nguồn sự thật duy nhất):\n{facts}\n\n"
            f"BẢN NHÁP CỦA GENERATOR (JSON):\n{json.dumps(draft, ensure_ascii=False)[:12000]}\n\n"
            "TIÊU CHÍ KIỂM TRA:\n- " + "\n- ".join(criteria) + "\n"
            "- Mọi con số trong bản nháp phải khớp dữ kiện; con số không có nguồn là lỗi nghiêm trọng.\n\n"
            'Trả về JSON: {"issues": ["..."], "revised": <bản đã sửa, cùng cấu trúc với bản nháp>}. '
            'Nếu không có lỗi: {"issues": [], "revised": null}.'
        )
        res = self.json(prompt, system=system, fallback={"issues": [], "revised": None})
        issues = [str(i) for i in (res or {}).get("issues", []) if i]
        revised = (res or {}).get("revised")
        if issues and isinstance(revised, type(draft)) and revised:
            return revised, issues
        return draft, issues


def parse_json(raw: str) -> Any:
    """Bóc JSON từ phản hồi LLM (chịu được ```json fences và văn bản thừa)."""
    if not raw:
        return None
    s = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(s[i : j + 1])
            except json.JSONDecodeError:
                continue
    return None


def to_facts(obj: Any, limit: int = 9000) -> str:
    """Chuyển dict số liệu sang chuỗi gọn để đưa vào prompt."""
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "...(cắt bớt)"
