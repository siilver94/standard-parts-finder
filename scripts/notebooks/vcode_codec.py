# notebooks/vcode_codec.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Dict, List, Tuple, Optional

import pandas as pd


# ----------------------------
# 소도구
# ----------------------------
def _s(x) -> str:
    """None/NaN 안전 문자열 변환"""
    if x is None:
        return ""
    try:
        import pandas as _pd
        if _pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def _slot_to_range(slot: str) -> Optional[Tuple[int, int]]:
    """
    '7–8', '7-8', '7-8'(NB hyphen), '~', ':' 등을 모두 허용.
    단일 '5'도 허용 -> (5,5)
    """
    if not slot:
        return None
    s = _s(slot).strip().replace(" ", "")
    s = (s.replace("\u2013", "-").replace("\u2012", "-")
           .replace("\u2011", "-").replace("\u2212", "-")
           .replace("~", "-").replace(":", "-"))
    if s == "" or s.lower() == "nan":
        return None
    if "-" in s:
        a, b = s.split("-", 1)
        return (int(a), int(b))
    i = int(s)
    return (i, i)


def _parse_int_codec(codec: str) -> Tuple[int, str]:
    """
    int 코덱 파싱: 'int:width=3,pad=0' -> (3,'0')
    없으면 (1,'')
    """
    width = 1
    pad = ""
    c = _s(codec)
    m = re.search(r"width\s*=\s*(\d+)", c)
    if m:
        width = int(m.group(1))
    m = re.search(r"pad\s*=\s*([0-9A-Za-z])", c)
    if m:
        pad = m.group(1)
    return width, pad


def _apply_codec(value, codec: str, width_hint: Optional[int] = None) -> str:
    """
    codec 규칙에 따라 value를 문자열 세그먼트로 변환
    - lookup:* 은 코드 문자열 그대로 사용
    - int:* 는 width/pad에 맞춰 정수화 + padding
    """
    c = _s(codec).strip()
    if c.startswith("lookup:"):
        return _s(value).strip()

    if c.startswith("int:"):
        width, pad = _parse_int_codec(c)
        sv = _s(value).strip()
        if sv == "":
            return "".rjust(width, pad or "0")  # 빈값이면 폭만큼 패딩
        try:
            s = str(int(sv))
        except Exception:
            s = re.sub(r"\D", "", sv) or "0"
        return s.rjust(width, pad) if pad else s.rjust(width)

    v = _s(value).strip()
    if width_hint and len(v) < width_hint:
        return v.rjust(width_hint, "0")
    return v


def _pair_prefixes(union_df: pd.DataFrame, pair_id: str) -> Tuple[str, str]:
    rows = union_df[union_df["pair_id"] == pair_id]
    if rows.empty:
        raise ValueError(f"pair_id '{pair_id}' 를 union_schema에서 찾지 못했습니다.")
    ik_pt = rows["ik_part_type"].iloc[0]
    ok_pt = rows["ok_part_type"].iloc[0]
    return str(ik_pt), str(ok_pt)


# ----------------------------
# 공개 API
# ----------------------------
def required_keys(union_df: pd.DataFrame, pair_id: str, side: str) -> List[str]:
    side = side.upper()
    col = "required_ik" if side == "IK" else "required_ok"
    return union_df[(union_df["pair_id"] == pair_id) & (union_df[col] == True)]["key"].tolist()


def extra_keys_from_other_side(union_df: pd.DataFrame, pair_id: str, base_side: str) -> List[str]:
    """
    base_side=IK이면 OK에서만 필수인 키 목록(= IK 폼에서 '추가 입력'으로 보여줄 키)
    """
    base_side = base_side.upper()
    if base_side == "IK":
        need_other = required_keys(union_df, pair_id, "OK")
        need_base = required_keys(union_df, pair_id, "IK")
    else:
        need_other = required_keys(union_df, pair_id, "IK")
        need_base = required_keys(union_df, pair_id, "OK")
    return [k for k in need_other if k not in need_base]


def missing_required_keys(union_df: pd.DataFrame, pair_id: str, side: str, attrs: Dict) -> List[str]:
    needs = required_keys(union_df, pair_id, side)
    miss = []
    for k in needs:
        v = attrs.get(k, None)
        if v is None or _s(v).strip() == "":
            miss.append(k)
    return miss


def encode_code(
    side: str,
    union_df: pd.DataFrame,
    pair_id: str,
    attrs: Dict,
    base_prefix: Optional[str] = None,
    fill_char: str = "?",
    overrides: Optional[Dict[str, str]] = None,   # ✅ overrides 지원
) -> str:
    """
    side: "IK" 또는 "OK"
    union_df: union_schema DataFrame
    pair_id: 예) "V111_2655"
    attrs: {"material": "2", ...}  # key 기준
    overrides: {"material": "7", ...}  # attrs보다 우선 (side별 인코딩 오염 방지용)
    """
    overrides = overrides or {}

    side = side.upper()
    if base_prefix is None:
        ik_pt, ok_pt = _pair_prefixes(union_df, pair_id)
        base_prefix = ik_pt if side == "IK" else ok_pt

    # ✅ 반드시 있어야 함 (S가 없으면 NameError)
    S = union_df[union_df["pair_id"] == pair_id]

    code = list(" " * 11)

    # prefix 삽입 (1부터 시작)
    for i, ch in enumerate(str(base_prefix), start=1):
        if i <= 11:
            code[i - 1] = ch

    slot_col = "ik_slot" if side == "IK" else "ok_slot"
    codec_col = "ik_codec" if side == "IK" else "ok_codec"

    for _, r in S.iterrows():
        key = r["key"]
        slot = r[slot_col]
        codec = r[codec_col]

        rng = _slot_to_range(slot)
        if rng is None:
            continue

        a, b = rng
        width = b - a + 1

        # ✅ overrides 우선 적용
        raw_val = overrides.get(key, None)
        if raw_val is None or _s(raw_val).strip() == "":
            raw_val = attrs.get(key, None)

        if raw_val is None or _s(raw_val).strip() == "":
            continue

        enc = _apply_codec(raw_val, codec, width_hint=width)

        # 길이 보정
        if len(enc) < width:
            enc = enc.rjust(width, "0")
        elif len(enc) > width:
            enc = enc[-width:]  # 뒤에서 width만큼 사용

        for off, ch in enumerate(enc):
            idx = a - 1 + off
            if 0 <= idx < 11:
                code[idx] = ch

    # 남은 공백 fill_char 처리
    code = [ch if ch != " " else fill_char for ch in code]
    return "".join(code)


def encode_both(
    union_df: pd.DataFrame,
    pair_id: str,
    attrs: Dict,
    fill_char: str = "?",
    ik_overrides: Optional[Dict[str, str]] = None,  # ✅ 추가
    ok_overrides: Optional[Dict[str, str]] = None,  # ✅ 추가
) -> Tuple[str, str]:
    ik_pt, ok_pt = _pair_prefixes(union_df, pair_id)

    ik = encode_code(
        "IK", union_df, pair_id, attrs,
        base_prefix=ik_pt, fill_char=fill_char,
        overrides=ik_overrides
    )
    ok = encode_code(
        "OK", union_df, pair_id, attrs,
        base_prefix=ok_pt, fill_char=fill_char,
        overrides=ok_overrides
    )
    return ik, ok


# ----------------------------
# 11자리 → attrs 역변환(디코더)
# ----------------------------
def _decode_slice(raw: str, codec: str) -> str:
    c = _s(codec).strip()
    if c.startswith("lookup:"):
        return raw  # 룩업은 코드 그대로
    if c.startswith("int:"):
        # int 포맷이면 왼쪽 패딩 제거 → 정수 → 문자열
        try:
            return str(int(raw))
        except Exception:
            return raw.lstrip("0") or "0"
    return raw


def decode_attrs_from_code(union_df: pd.DataFrame, side: str, code11: str):
    """
    11자리 코드 → (pair_id, attrs dict, part_type)
    """
    side = side.upper()
    code11 = _s(code11).upper().strip()
    code11 = re.sub(r"[^A-Z0-9]", "", code11)
    if len(code11) != 11:
        raise ValueError("코드는 반드시 11자리여야 합니다.")

    # 후보 pair 찾기
    if side == "IK":
        cand = union_df[union_df["ik_part_type"].astype(str) == code11[:4]]
    else:
        ok4 = union_df["ok_part_type"].astype(str).str.len() == 4
        ok5 = union_df["ok_part_type"].astype(str).str.len() == 5
        cand5 = union_df[ok5 & (union_df["ok_part_type"].astype(str) == code11[:5])]
        cand4 = union_df[ok4 & (union_df["ok_part_type"].astype(str) == code11[:4])]
        cand = cand5 if not cand5.empty else cand4

    if cand.empty:
        return None, {}, code11[:4]

    pair_id = cand["pair_id"].iloc[0]
    part_type = cand["ik_part_type"].iloc[0] if side == "IK" else cand["ok_part_type"].iloc[0]
    slot_col = "ik_slot" if side == "IK" else "ok_slot"
    codec_col = "ik_codec" if side == "IK" else "ok_codec"

    S = union_df[union_df["pair_id"] == pair_id]
    attrs = {}
    for _, r in S.iterrows():
        key = r["key"]
        slot = _s(r[slot_col])
        codec = _s(r[codec_col])
        rng = _slot_to_range(slot)
        if not rng:
            continue
        a, b = rng
        raw = code11[a - 1:b]
        attrs[key] = _decode_slice(raw, codec)

    return pair_id, attrs, part_type
