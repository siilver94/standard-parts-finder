#!/usr/bin/env python
# coding: utf-8
"""
V/KM-Code 통합 조회 Demo (union_schema + vcode_codec 통합 확장판)
- 카테고리 → 세부명칭(파트타입) 선택
- 기준 측(IK/OK)을 고르면: 기준 필수 + 상대측 부족분만 입력 UI 자동 생성
- 입력 한 번 → IK/OK 11자리 동시에 생성 (encode_both)
- matched_parts.csv 있으면 결과 확인/보강
- 좌/우(익산/옥천) 이미지 가로 출력
"""

import re
import streamlit as st
import math
import pandas as pd

# 프로젝트용 로더/헬퍼
from utils.loaders import (
    load_catalog,           # part_master.csv (+category)
    load_images,            # images/<part_type>*.{jpg,png}
    load_code_schema,       # codeSchema_IK.csv / codeSchema_OK.csv (참고: 일부 util에서만 사용)
    load_lookups,           # 7종 lookup dict
    lookup_options,         # lookup 테이블에서 part_type별 옵션 dict 추출 {code: label}
    load_crossmap,          # Cross_Map.csv → (ik2ok, ok2ik)
    load_matched_full,      # matched_parts.csv (안전 로더)
    load_union_schema,      # union_schema.csv 로더
)

# 이미지
# from utils.images import find_images
from utils.images import find_images_with_prefix_fallback
from PIL import Image 

# vcode_codec (11자리 조립/해석기)
from notebooks.vcode_codec import (
    encode_both, required_keys, extra_keys_from_other_side, missing_required_keys, _slot_to_range,
    decode_attrs_from_code,
)
import re

from utils.attr_rules import load_match_table_csv, compile_rules, apply_rules_to_attrs, build_schema_maps

# -------------------------------------------------------------------
# (핵심) pending do_query 복원은 widget_updates와 무관하게 항상 처리해야 함
# - shared lookup conflict는 overrides만 바뀌고 widget_updates는 없을 수 있음
# -------------------------------------------------------------------
if st.session_state.pop("__pending_do_query__", False):
    st.session_state["__do_query__"] = True
if "__pending_widget_updates__" in st.session_state:
    upd = st.session_state.pop("__pending_widget_updates__", {})
    for k, v in upd.items():
        st.session_state[k] = v  # ✅ 이 시점은 위젯 instantiate 전이라 안전

    # ✅ (추가) pending으로 넘어온 "조회 의도"가 있으면 다음 run에서 강제 복원
    if st.session_state.pop("__pending_do_query__", False):
        st.session_state["__do_query__"] = True

    # 부가 상태들도 꺼내서 유지(원하면)
    st.session_state["__infos__"] = st.session_state.pop("__pending_infos__", [])
    st.session_state["__blockers__"] = st.session_state.pop("__pending_blockers__", [])
    st.session_state["__conflicts__"] = st.session_state.pop("__pending_conflicts__", [])
    
    # ✅ 수정(키가 있을 때만 반영)
    st.session_state["__run_ik_overrides__"] = st.session_state.pop("__pending_ik_overrides__", {})
    st.session_state["__run_ok_overrides__"] = st.session_state.pop("__pending_ok_overrides__", {})

def _auto_fill_nominal(attrs: dict) -> dict:
    if not attrs:
        return attrs

    raw_nom   = str(attrs.get("nominal", "")).strip()
    raw_nom10 = str(attrs.get("nominalX10", "")).strip()

    # 이미 둘 다 있으면 손대지 않음
    if raw_nom and raw_nom10:
        return attrs

    # 1) nominal만 있는 경우 → nominalx10 자동 생성 (호칭 × 10)
    if raw_nom and not raw_nom10:
        if raw_nom.isdigit():
            n = int(raw_nom)
            attrs["nominalX10"] = str(n * 10)
        return attrs

    # 2) nominalx10만 있는 경우 → nominal 자동 생성 (호칭x10 / 10)
    if raw_nom10 and not raw_nom:
        if raw_nom10.isdigit():
            n10 = int(raw_nom10)
            # ★ 더 이상 10의 배수 체크하지 않고 그냥 나눠서 사용
            attrs["nominal"] = str(n10 // 10)
        return attrs

    # 둘 다 비어 있으면 그대로
    return attrs


V3_RE = re.compile(r"^(V)(\d{2})(\d)$", re.IGNORECASE)  # V111, V802...
V2_RE = re.compile(r"^(V)(\d{2})$",     re.IGNORECASE)  # V11, V80...

def _ik_group_key(ptype: str) -> str:
    s = (ptype or "").strip().upper()
    m3 = V3_RE.match(s)
    if m3:
        return f"{m3.group(1)}{m3.group(2)}"
    return s

def _candidate_keys(system: str, ptype_raw: str) -> list[str]:
    """IK: [정확, 그룹] / OK: [정확]"""
    s = (ptype_raw or "").strip().upper()
    if system == "IK":
        g = _ik_group_key(s)
        return [s] if s == g else [s, g]
    return [s]

def _extract_spec_common(bundle):
    """
    load_lookups()[table] 이 반환하는 여러 포맷을 spec/common 두 dict로 표준화.
    지원 포맷:
      - (spec, common)
      - (spec, common, meta...)
      - {"spec": dict, "common": dict}
      - {"by_part_type": {(pt,code):label}, "*": {code:label}}
      - {"V11": {code:label}, "V801": {...}, "*": {...}}  # 중첩 dict → (pt,code)로 평탄화
    """
    spec, common = {}, {}

    # 튜플/리스트 포맷
    if isinstance(bundle, (list, tuple)):
        if len(bundle) >= 2:
            spec, common = bundle[0], bundle[1]
            if not isinstance(spec, dict):  spec = {}
            if not isinstance(common, dict): common = {}
            return spec, common

    # 딕셔너리 포맷
    if isinstance(bundle, dict):
        # 흔한 키 우선 탐색
        cand_spec_keys   = ("spec", "by_part_type", "pt", "map", "part_type")
        cand_common_keys = ("common", "*", "global")
        for k in cand_spec_keys:
            if k in bundle and isinstance(bundle[k], dict):
                spec = bundle[k]
                break
        for k in cand_common_keys:
            if k in bundle and isinstance(bundle[k], dict):
                common = bundle[k]
                break

        # 아직도 spec 비었으면: { "V11": {...}, "V801": {...}, "*": {...} } 스타일 평탄화
        if not spec:
            flat = {}
            for k, v in bundle.items():
                if k == "*" or not isinstance(v, dict):
                    continue
                # k = part_type, v = {code: label}
                for code, label in v.items():
                    flat[(str(k).upper().strip(), str(code))] = label
            if flat:
                spec = flat

        # common이 없고 '*'가 있으면 사용
        if not common and "*" in bundle and isinstance(bundle["*"], dict):
            common = bundle["*"]

        return spec or {}, common or {}

    # 알 수 없는 포맷
    return {}, {}

def _merged_lookup_options(lookups: dict, table: str, system: str, ptype_raw: str) -> dict:
    """
    반환: {code: label}
    정책(중요):
      - 정확키에 값이 하나라도 있으면: 정확키만 반환 (공통(*) 섞지 않음)
      - 정확키가 비면: 그룹키만 반환 (공통(*) 섞지 않음)
      - 둘 다 비면: 공통(*)만 반환
    """
    bundle = lookups.get(table)
    if not bundle:
        return {}

    # loaders.py 포맷: {"spec": {(pt,code):label}, "common":{code:label}, ...}
    spec = bundle.get("spec", {}) if isinstance(bundle, dict) else {}
    common = bundle.get("common", {}) if isinstance(bundle, dict) else {}

    # 1) 정확키/그룹키 중 "처음 발견되는" 쪽만 사용
    for key in _candidate_keys(system, ptype_raw):
        out = {}
        # spec이 (pt,code)->label
        for (pt, code), label in spec.items():
            if str(pt).upper().strip() == str(key).upper().strip():
                out[str(code)] = str(label)
        if out:
            return out  # ★ 공통은 섞지 않는다

    # 2) spec에 아무것도 없으면 공통만
    return {str(code): str(label) for code, label in common.items()}

# ---------------------------------------------------------------------
# 기본 페이지 설정 (wide + 제목/아이콘)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Standard Parts Finder", page_icon="🔎", layout="wide")
st.title("Standard Parts Finder")

if "__await_conflict__" not in st.session_state:
    st.session_state["__await_conflict__"] = False
if "__active_conflicts__" not in st.session_state:
    st.session_state["__active_conflicts__"] = []
if "__conflict_pair_id__" not in st.session_state:
    st.session_state["__conflict_pair_id__"] = None
if "__conflict_base_side__" not in st.session_state:
    st.session_state["__conflict_base_side__"] = None

# (변경) 이번 조회(run)에서만 쓰는 overrides (저장/누적 금지)
if "__run_ik_overrides__" not in st.session_state:
    st.session_state["__run_ik_overrides__"] = {}
if "__run_ok_overrides__" not in st.session_state:
    st.session_state["__run_ok_overrides__"] = {}

# ---------------------------------------------------------------------
# (추가) 마지막 "성공" 코드 결과 저장소
# - 충돌/에러 run에서는 코드가 절대 보이지 않게 하려고 session_state에 따로 보관
# ---------------------------------------------------------------------
if "__last_codes__" not in st.session_state:
    st.session_state["__last_codes__"] = {"pair_id": None, "ik": None, "ok": None}
# ---------------------------------------------------------------------
# (추가) 충돌 해결값 저장소
# - 사용자가 한 번 선택한 충돌은 다음 조회부터 다시 묻지 않기 위해 저장
# - key: (pair_id, side, attr_key, lookup) -> chosen_code
# ---------------------------------------------------------------------
if "__conflict_resolutions__" not in st.session_state:
    st.session_state["__conflict_resolutions__"] = {}

# ---------------------------------------------------------------------
# 0) 공통 유틸
# ---------------------------------------------------------------------
def _norm(s: str) -> str:
    """코드 비교용 정규화: 공백/하이픈 제거 + 대문자화"""
    return re.sub(r"[\s\-]+", "", str(s or "")).upper()

# (참고 util) 스키마 기반 폭 산출/패딩 — 현재는 직접 사용하지 않지만 호환성 위해 보관
def _attr_width_from_schema(site: str, part_type: str, attr_name: str):
    schema = load_code_schema(site.upper())

    if site.upper() == "IK":
        # V131 → ["V131", "V13"] 같이 그룹키까지 함께 검색
        keys = _candidate_keys("IK", part_type)
        r = schema[
            (schema.part_type.astype(str).isin(keys)) &
            (schema.attr_name == attr_name)
        ]
    else:
        r = schema[
            (schema.part_type.astype(str) == str(part_type)) &
            (schema.attr_name == attr_name)
        ]

    if r.empty:
        return None
    try:
        pf = int(r.iloc[0]["pos_from"])
        pt = int(r.iloc[0]["pos_to"])
        return max(1, pt - pf + 1)
    except Exception:
        return None


def normalize_selected_by_schema(site: str, part_type: str, selected: dict) -> dict:
    """
    codeSchema의 pos_from/pos_to 기준으로 자리수 맞춰 0-padding.
    예: width=2, value='3' → '03'
    """
    if not selected:
        return {}
    out = {}
    for k, v in selected.items():
        s = str(v).strip()
        width = _attr_width_from_schema(site, part_type, k)
        if width and s:
            s = s.zfill(width)
        out[k] = s
    return out

def get_attr_label_kor(site: str, part_type: str, attr_name: str) -> str:
    """
    codeSchema_IK / codeSchema_OK 에서 attr_name에 대응하는 한글 라벨(kor 컬럼)을 찾아서 반환.
    - 우선순위 1: (part_type, attr_name) 일치하는 행의 kor
    - 우선순위 2: attr_name만 일치하는 행의 kor
    - 찾지 못하면 원래 attr_name을 그대로 사용
    """
    try:
        schema = load_code_schema(site.upper())
    except Exception:
        # 로드 실패 시 그냥 원래 영문 attr_name 사용
        return attr_name

    if "kor" not in schema.columns:
        # kor 컬럼이 없으면 그대로 반환
        return attr_name

    # 1) part_type + attr_name 같이 맞는 행을 우선 검색
    hit = schema[
        (schema["part_type"].astype(str) == str(part_type)) &
        (schema["attr_name"].astype(str) == str(attr_name))
    ]
    if not hit.empty:
        txt = str(hit.iloc[0]["kor"]).strip()
        if txt:
            return txt

    # 2) attr_name만 기준으로 검색 (여러 part_type에서 공통 사용되는 경우)
    hit2 = schema[schema["attr_name"].astype(str) == str(attr_name)]
    if not hit2.empty:
        txt = str(hit2.iloc[0]["kor"]).strip()
        if txt:
            return txt

    # 3) 끝까지 못 찾으면 그냥 원래 이름
    return attr_name

def assemble_by_schema(site: str, part_type: str, selected: dict) -> str | None:
    schema = load_code_schema(site.upper())

    if site.upper() == "IK":
        keys = _candidate_keys("IK", part_type)  # ["V131", "V13"]
        rules = schema[schema.part_type.astype(str).isin(keys)].copy()
    else:
        rules = schema[schema.part_type.astype(str) == str(part_type)].copy()

    if rules.empty or not selected:
        return None

    segs = []
    for _, r in rules.iterrows():
        attr = (r.get("attr_name", "") or "").strip()
        if not attr or attr not in selected:
            continue
        val = str(selected.get(attr, "")).strip()
        if not val:
            continue
        segs.append(val)

    # ★ 완성 코드는 실제 part_type (예: V131) + 속성
    return f"{part_type}{''.join(segs)}" if segs else None

# ---------------------------------------------------------------------
# 1) 상태: 카테고리/파트타입 프리필 + 11자리 프리필 attrs
# ---------------------------------------------------------------------
if "pref_cat" not in st.session_state:
    st.session_state.pref_cat = None
if "pref_pt" not in st.session_state:
    st.session_state.pref_pt = None
# ★ 변경: 11자리 해석으로 얻은 attrs 저장용
if "prefill_attrs" not in st.session_state:
    st.session_state.prefill_attrs = {}
if "__basis_override" not in st.session_state:
    st.session_state.__basis_override = None

# 카탈로그 로드 (site/category/part_type/remark 등)
df = load_catalog()

# ---------------------------------------------------------------------
# 2) 빠른 검색 (V*** / ####/##### / ★ 11자리)
# ---------------------------------------------------------------------
with st.expander("🔎 빠른 검색 (표준품번)"):
    # 폼으로 묶어서 Enter 로도 제출 가능하게
    with st.form("quick_search_form"):
        q = st.text_input("", placeholder="예: V111 / 2655 / V111260408")
        submitted_q = st.form_submit_button("찾기", use_container_width=False)

    if submitted_q:
        s = (q or "").strip().upper()

        # (A) 11자리 완전코드 → 자동 프리필
        if re.fullmatch(r"[A-Z0-9]{11}", s):
            udf_local = load_union_schema()
            side = "IK" if s.startswith("V") else "OK"
            pair_id, attrs, pt = decode_attrs_from_code(udf_local, side, s)

            # 카탈로그에서 part_type 찾아 화면 이동
            hit = df[df.part_type.astype(str) == pt]
            if hit.empty and side == "OK":
                # OK 5자리 우선이라면 4자리 보정
                hit = df[df.part_type.astype(str) == s[:4]]

            if hit.empty:
                st.warning("카탈로그에서 해당 part_type을 찾지 못했습니다.")
            else:
                st.session_state.pref_pt  = hit.iloc[0]["part_type"]
                st.session_state.pref_cat = hit.iloc[0]["category"]

            # ★ 변경: 프리필 값과 기준방향 세션 저장
            st.session_state.prefill_attrs = {k: ("" if attrs.get(k) is None else str(attrs.get(k))) for k in attrs}
            st.session_state.__basis_override = ("익산 코드 입력 → 옥천 자동 조회" if side=="IK"
                                                 else "옥천 코드 입력 → 익산 자동 조회")

            st.success(f"선택 이동: {pt} (코드 프리필 완료)")
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

        # (B) part_type만 검색 (V### 또는 ####/#####)
        elif re.fullmatch(r"V\d{3}", s) or re.fullmatch(r"\d{4,5}", s):
            hit = df[df.part_type == s]
            if hit.empty:
                st.warning("해당 Part Type이 part_master에 없습니다.")
            else:
                st.session_state.pref_pt  = s
                st.session_state.pref_cat = hit.iloc[0]["category"]
                # ★ 변경: 프리필 초기화
                st.session_state.prefill_attrs = {}
                st.session_state.__basis_override = None
                st.success(f"선택 이동: {s}")
        else:
            st.error("형식이 올바르지 않습니다. V### / #### / ##### / 또는 11자리 코드")

# ---------------------------------------------------------------------
# 3) 대분류 → 세부명칭 (IK 우선 / Cross_Map 라벨 표시)
# ---------------------------------------------------------------------
cats = sorted(df["category"].dropna().unique())
cat_idx = cats.index(st.session_state.pref_cat) if st.session_state.pref_cat in cats else 0
cat = st.selectbox("대분류", cats, index=cat_idx)

df_sub = df[df.category == cat].copy()
if df_sub.empty:
    st.warning("이 대분류에 등록된 품목이 없습니다.")
    st.stop()

# IK 우선 노출 (없으면 OK만)
is_iksan = df_sub["part_type"].astype(str).str.startswith("V", na=False)
df_ik    = df_sub[is_iksan]
df_show  = df_ik if not df_ik.empty else df_sub[~is_iksan]

# Cross_Map 라벨: "V111 ↔ 2655"
ik2ok, ok2ik = load_crossmap()

# ★ Cross_Map.csv에서 부품명(note) 가져오기
try:
    cross_df = pd.read_csv("data/Cross_Map.csv", dtype=str)
except Exception:
    cross_df = pd.DataFrame(columns=["ik_part_type", "ok_km_code", "note"])

# 컬럼명이 'remark'인 경우도 대비
name_col = "note"
if "remark" in cross_df.columns:
    name_col = "remark"

note_by_ik = {}
note_by_ok = {}

for _, row in cross_df.iterrows():
    ik = str(row.get("ik_part_type") or "").strip()
    ok = str(row.get("ok_km_code") or "").strip()
    nm = str(row.get(name_col) or "").strip()
    if not nm:
        continue
    if ik:
        note_by_ik[ik] = nm
    if ok:
        note_by_ok[ok] = nm


label_map = {}
for _, r in df_show.iterrows():
    pt = str(r.part_type)

    # IK 시작 / OK 시작에 따라 페어링
    if pt.startswith("V"):
        paired   = ik2ok.get(pt, "")
        pair_txt = f"{pt} ↔ {paired}" if paired else pt
    else:
        paired   = ok2ik.get(pt, "")
        pair_txt = f"{paired} ↔ {pt}" if paired else pt

    # 1순위: part_master.remark (현재 df_show의 remark)
    remark = ""
    if "remark" in r.index and r["remark"] is not None:
        remark = str(r["remark"]).strip()

    # 2순위: Cross_Map 의 note/remark (자기 코드 기준)
    if not remark:
        if pt.startswith("V"):
            remark = note_by_ik.get(pt, "")  # V코드
        else:
            remark = note_by_ok.get(pt, "")  # KM코드

    # 3순위: Cross_Map 에서 매칭된 상대 코드 기준 이름
    if not remark and paired:
        if str(paired).startswith("V"):
            remark = note_by_ik.get(str(paired), "")
        else:
            remark = note_by_ok.get(str(paired), "")

    # 최종 label
    key = f"{remark} ({pair_txt})" if remark else pair_txt
    label_map[key] = (pt, paired)


labels = list(label_map.keys())
if not labels:
    st.error("선택 가능한 세부명칭이 없습니다.")
    st.stop()

# 빠른검색 프리필 인덱스 처리
if st.session_state.pref_pt:
    pre = next(
        (k for k, (pt, op) in label_map.items()
         if pt == st.session_state.pref_pt or op == st.session_state.pref_pt),
        None
    )
    sel_idx = labels.index(pre) if pre in labels else 0
else:
    sel_idx = 0

sel_label         = st.selectbox("세부명칭", labels, index=sel_idx)
sel_pt, paired_pt = label_map[sel_label]

# 좌/우 part_type 확정
ik_pt = sel_pt if sel_pt.startswith("V") else (paired_pt or "")
ok_pt = (paired_pt or "") if sel_pt.startswith("V") else sel_pt

# ★ IK/OK 존재 여부 플래그
has_ik = bool(ik_pt)
has_ok = bool(ok_pt)
has_both = has_ik and has_ok
ik_only = has_ik and not has_ok
ok_only = has_ok and not has_ik

# 좌/우 part_type 확정
ik_pt = sel_pt if sel_pt.startswith("V") else (paired_pt or "")
ok_pt = (paired_pt or "") if sel_pt.startswith("V") else sel_pt
#st.caption(f"선택된 Pair  |  IK: {ik_pt or '-'}  /  OK: {ok_pt or '-'}")

# union 스키마 + pair_id (양쪽 다 있을 때만 사용)
udf = load_union_schema()
pair_id = f"{ik_pt}_{ok_pt}" if has_both else None

if pair_id and udf[udf["pair_id"] == pair_id].empty:
    st.warning(f"union_schema에 pair_id '{pair_id}' 행이 없습니다. (빌더 최신화 확인)")

@st.cache_resource(show_spinner=False)
def _load_compiled_attr_rules():
    df_rules = load_match_table_csv("data/매칭테이블.csv")
    return compile_rules(df_rules)

ATTR_RULES = _load_compiled_attr_rules()

def auto_target_keys_from_rules(udf: pd.DataFrame, pair_id: str, base_side: str, compiled_rules: dict) -> set[str]:
    base_side = (base_side or "").upper().strip()
    direction = "익산->옥천" if base_side == "IK" else "옥천->익산"
    other_side = "OK" if base_side == "IK" else "IK"

    smap = build_schema_maps_all(udf, pair_id)
    IK_L2K = smap.get("IK_lookup_to_key", {}) or {}
    OK_L2K = smap.get("OK_lookup_to_key", {}) or {}

    def l2k(side: str, lookup: str) -> str | None:
        return IK_L2K.get(lookup) if side == "IK" else OK_L2K.get(lookup)

    maps_for_pair = compiled_rules.get("maps", {}).get(direction, {}).get(pair_id, {})
    if not maps_for_pair:
        return set()

    auto_keys: set[str] = set()
    for (_trig_lk, _trig_cd), actions in maps_for_pair.items():
        for (tgt_lk, _tgt_cd) in actions:
            tgt_key = l2k(other_side, str(tgt_lk).strip())
            if tgt_key:
                auto_keys.add(str(tgt_key).strip())

    return auto_keys

def build_schema_maps_all(udf: pd.DataFrame, pair_id: str) -> dict:
    """
    union_schema의 pair_id 블록에서 required 여부와 관계 없이
    lookup -> key 매핑을 만든다. (extra 키도 잡히게)
    """
    block = udf[udf["pair_id"] == pair_id].copy()
    if block.empty:
        return {"IK_lookup_to_key": {}, "OK_lookup_to_key": {}}

    for c in ("lookup", "key", "ik_slot", "ok_slot"):
        if c in block.columns:
            block[c] = block[c].fillna("").astype(str).str.strip()

    ik_l2k, ok_l2k = {}, {}

    for _, r in block.iterrows():
        lk = r.get("lookup", "")
        k  = r.get("key", "")
        if not lk or not k:
            continue

        # IK쪽 속성이 존재하는 행(ik_slot이 있거나 required_ik든 뭐든 "IK에 속함"이면)
        if str(r.get("ik_slot", "")).strip():
            ik_l2k.setdefault(lk, k)

        # OK쪽 속성이 존재하는 행
        if str(r.get("ok_slot", "")).strip():
            ok_l2k.setdefault(lk, k)

    return {"IK_lookup_to_key": ik_l2k, "OK_lookup_to_key": ok_l2k}

# ---------------------------------------------------------------------
# 4) 입력 기준 선택
# ---------------------------------------------------------------------
basis = st.radio(
    "입력 기준을 선택하세요",
    ["익산 코드 입력 → 옥천 자동 조회", "옥천 코드 입력 → 익산 자동 조회"],
    horizontal=True,
)
# ★ 변경: 11자리 프리필에서 기준 자동 전환
if st.session_state.__basis_override in ["익산 코드 입력 → 옥천 자동 조회","옥천 코드 입력 → 익산 자동 조회"]:
    basis = st.session_state.__basis_override

basis_ik   = basis.startswith("익산")
base_side  = "IK" if basis_ik else "OK"
other_side = "OK" if basis_ik else "IK"

# ---------------------------------------------------------------------
# 5) 속성 렌더러 (slot 순 정렬 + 프리필 주입)
# ---------------------------------------------------------------------
def _slot_range(slot: str):
    try:
        rng = _slot_to_range(slot)
        if not rng:
            return (999, 999)
        return rng    
    except Exception:
        return (999, 999)

def _order_keys_by_slot(udf, pair_id: str, side: str, keys: list[str]) -> list[str]:
    col_slot = "ik_slot" if side.upper()=="IK" else "ok_slot"
    S = _build_S_for_pair(udf, pair_id)   # ★ 여기로 통일
    pairs = []
    for k in keys:
        if k in S.index:
            
            val = S.loc[k, col_slot]

            # 중복 key가 있으면 pandas Series로 떨어질 수 있음 → 첫 행만 사용
            if hasattr(val, "iloc"):
                val = val.iloc[0]
            
            a, b = _slot_range(str(val or ""))

            
            pairs.append((a, b, k))
        else:
            pairs.append((999, 999, k))
    pairs.sort()
    return [k for _,__,k in pairs]

# ★ 변경: 프리필을 실제 위젯 기본값으로 주입하는 헬퍼
def _prime_default(widget_key: str, value: str):
    """해당 위젯 key가 아직 세션에 없을 때만 초기값을 주입."""
    if value is None:
        return
    if widget_key not in st.session_state:
        st.session_state[widget_key] = str(value)
        
def _build_S_for_pair(udf: pd.DataFrame, pair_id: str) -> pd.DataFrame:
    """
    union_schema에서 같은 pair_id 안에 key가 중복인 경우가 있어도
    UI가 안정적으로 동작하도록 key 기준으로 1개만 남겨 index로 만든다.
    """
    dfp = udf[udf["pair_id"] == pair_id].copy()

    # 혹시라도 공백/NaN 섞이면 방어
    dfp["key"] = dfp["key"].fillna("").astype(str).str.strip()

    # 빈 key 제거
    dfp = dfp[dfp["key"] != ""]

    # ★ 핵심: key 중복 제거 (첫 행 유지)
    dfp = dfp.drop_duplicates(subset=["key"], keep="first")

    # index 구성
    return dfp.set_index("key")

def _render_inputs_for_side(
    udf, pair_id: str, side: str, pt_for_lookup: str,
    keys: list[str], tag_suffix: str=""
) -> dict:
    lookups = load_lookups()
    #S = udf[udf["pair_id"] == pair_id].copy().set_index("key")
    S = _build_S_for_pair(udf, pair_id)
    keys_sorted = _order_keys_by_slot(udf, pair_id, side, keys)

    # ★ 변경: 11자리에서 해석해 온 프리필 딕셔너리
    prefill = st.session_state.get("prefill_attrs", {}) or {}

    attrs = {}
    cols = st.columns(2)
    i = 0
    for k in keys_sorted:
        if k not in S.index:
            st.warning(f"{k} : union_schema에 없음")
            continue
        r = S.loc[k]
        dtype   = str(r.get("dtype","") or "").strip()
        lookup  = str(r.get("lookup","") or "").strip()
        ik_slot = str(r.get("ik_slot","") or "")
        ok_slot = str(r.get("ok_slot","") or "")

        slot = ik_slot if side.upper()=="IK" else ok_slot
        a,b = _slot_range(slot)
        width_hint = 0 if a==999 else (b-a+1)

        c = cols[i % 2]; i += 1

        # ★ 여기서 attr_name → kor 라벨로 변환
        #   site: IK/OK (side), part_type: pt_for_lookup, attr_name: k
        kor_label = get_attr_label_kor(side, pt_for_lookup, k)

        # 화면에 보이는 건 kor + (추가) 태그만 붙여서 사용
        label = f"{kor_label}{tag_suffix}"

        # key는 그대로 영문 attr_name을 사용 (내부 로직 및 세션 키 안정성 유지)
        key = f"U:{pair_id}:{side}:{k}"
        
        # ✅ 이번 run에서 실제로 렌더링되는 위젯 키 기록
        st.session_state["__visible_widget_keys__"].add(key)
        
        # ★ 0으로 한다(00, 000…) 자동 채움: attr_name == "0"
        if str(k) == "0":
            # 자릿수만큼 0 채우기 (slot 기반)
            zero_width = width_hint if width_hint > 0 else 1
            attrs[k] = "0" * zero_width
            # 입력 위젯을 아예 만들지 않고 넘어간다
            continue

        # 프리필 값(문자열) 준비
        pre_val = "" if prefill.get(k) is None else str(prefill.get(k)).strip()

        if dtype == "lookup" and lookup:
            system_local = "IK" if str(pt_for_lookup).upper().startswith("V") else "OK"
            opts = _merged_lookup_options(lookups, lookup, system_local, pt_for_lookup)  # {code: label}
            if opts:
                codes = list(opts.keys())
                # ✅ 기준측이면, 매칭테이블(룰)에 존재하는 trigger 코드만 남긴다
                try:
                    # side가 기준측일 때만 제한
                    if pair_id and (side.upper() == base_side.upper()):
                        direction = "익산->옥천" if base_side.upper() == "IK" else "옥천->익산"
                        trig_map = ATTR_RULES.get("trigger_codes", {}).get(direction, {}).get(pair_id, {})
                        allow = trig_map.get(lookup, None)
                
                        # trigger_codes가 있으면 그걸 최우선으로 사용 (v130이면 material에서 2/6만 남음)
                        if allow:
                            allow = set([str(x).strip() for x in allow])
                            codes = [c for c in codes if str(c).strip() in allow]
                except Exception:
                    pass
                # ★ 변경: selectbox 기본값 주입
                if pre_val and pre_val in codes:
                    _prime_default(key, pre_val)
                    sel = c.selectbox(label, codes, format_func=lambda code: f"{code} - {opts.get(code,'')}",
                                      key=key)
                else:
                    sel = c.selectbox(label, codes, format_func=lambda code: f"{code} - {opts.get(code,'')}",
                                      key=key)
                attrs[k] = sel
            else:
                # 룩업 테이블이 비어있으면 코드 직접 입력
                _prime_default(key, pre_val)
                attrs[k] = c.text_input(f"{label} [코드]", key=key)
        else:
            # 자유 입력 — 자리 힌트 제공
            _prime_default(key, pre_val)  # ★ 변경: text_input 기본값
            ph = f"{width_hint}자리 숫자" if width_hint>0 else ""
            attrs[k] = c.text_input(label, key=key, placeholder=ph)

    return attrs

def _render_single_side_inputs(site: str, part_type: str) -> dict:
    schema = load_code_schema(site.upper())

    if site.upper() == "IK":
        cand_keys = _candidate_keys("IK", part_type)   # 예: V131 → ["V131", "V13"]
        rules = schema[schema.part_type.astype(str).isin(cand_keys)].copy()
    else:
        rules = schema[schema.part_type.astype(str) == str(part_type)].copy()

    if rules.empty:
        st.info("이 part_type에 대한 속성 스키마가 없습니다.")
        return {}

    lookups = load_lookups()
    system_local = "IK" if site.upper() == "IK" else "OK"

    attrs = {}
    cols = st.columns(2)
    i = 0

    for _, r in rules.iterrows():
        attr = (r.get("attr_name", "") or "").strip()
        if not attr:
            continue

        # 자리수 힌트
        width_hint = None
        try:
            pf = int(r.get("pos_from", 0))
            pt = int(r.get("pos_to", 0))
            width_hint = max(1, pt - pf + 1)
        except Exception:
            width_hint = None

         # ★ 0으로 한다(00, 000…) 자동 채움
        if attr == "0":
            attrs[attr] = "0" * width_hint
            continue


        # ★ lookup_table 기준으로 lookup 여부 판단
        lookup_table = str(r.get("lookup_table", "") or "").strip()

        c = cols[i % 2]; i += 1

        kor_label = get_attr_label_kor(site, part_type, attr)  # 이미 쓰고 있는 helper
        label = kor_label or attr
        key = f"S:{site}:{part_type}:{attr}"

        # 현재 값 (세션 상태)
        pre_val = st.session_state.get(key, "")

        if lookup_table:
            # 룩업 테이블에서 옵션 가져오기
            opts = _merged_lookup_options(lookups, lookup_table, system_local, part_type)
            if opts:
                codes = list(opts.keys())
                # 기본값 인덱스
                if pre_val and pre_val in codes:
                    idx = codes.index(pre_val)
                else:
                    idx = 0
                sel = c.selectbox(
                    label,
                    codes,
                    index=idx,
                    format_func=lambda code: f"{code} - {opts.get(code, '')}",
                    key=key,
                )
                attrs[attr] = sel
                continue

        # lookup_table이 없거나 옵션이 비어 있으면 텍스트 입력
        ph = f"{width_hint}자리 숫자" if width_hint else ""
        val = c.text_input(label, key=key, placeholder=ph)
        attrs[attr] = val

    return attrs

def get_required_sets(udf, pair_id: str, base_side: str):
    need_base  = required_keys(udf, pair_id, base_side)
    need_extra = extra_keys_from_other_side(udf, pair_id, base_side)
                                           
    # ✅ (핵심) 룰로 자동 채워질 수 있는 상대측 key는 extra 입력 UI에서 숨김
    try:
        auto_keys = auto_target_keys_from_rules(udf, pair_id, base_side, ATTR_RULES)
        if auto_keys:
            need_extra = [k for k in need_extra if k not in auto_keys]
    except Exception:
        pass
    
    if not need_base and not need_extra:
        all_keys = udf[udf["pair_id"] == pair_id]["key"].dropna().unique().tolist()
        need_base = all_keys

    # ★ 추가: 호칭 계열 키는 기준측에만 표시하고 상대측(extra)에서는 숨김
    nominal_family = {"nominal", "nominalX10"}
    if any(k in need_base for k in nominal_family):
        need_extra = [k for k in need_extra if k not in nominal_family]

    def get_required_sets(udf, pair_id: str, base_side: str):
        need_base  = required_keys(udf, pair_id, base_side)
        need_extra = extra_keys_from_other_side(udf, pair_id, base_side)
    
        if not need_base and not need_extra:
            all_keys = udf[udf["pair_id"] == pair_id]["key"].dropna().unique().tolist()
            need_base = all_keys
    
        # ★ 추가: 호칭 계열 키는 기준측에만 표시하고 상대측(extra)에서는 숨김
        nominal_family = {"nominal", "nominalX10"}
        if any(k in need_base for k in nominal_family):
            need_extra = [k for k in need_extra if k not in nominal_family]
    
        # ✅ (핵심) 룰로 자동 채워질 수 있는 '상대측 키'는 extra 입력 UI에서 숨김
        try:
            auto_keys = auto_target_keys_from_rules(udf, pair_id, base_side, ATTR_RULES)
            if auto_keys:
                need_extra = [k for k in need_extra if k not in auto_keys]
        except Exception:
            # 필터 실패해도 기존 동작 유지
            pass
 
    return need_base, need_extra

if ok_pt == "2258":
    st.info("참고: 옥천 2258은 11번째 자리(재료종류)에 따라 익산 품번이 달라질 수 있습니다. (예: 2→V130, 4→V132, 6→V134) 필요 시 익산 품번을 확인 후 진행해 주세요.")
# ---------------------------------------------------------------------
# 6 + 7) 좌/우 패널 렌더 + 조회/생성 (Enter 지원 위해 하나의 form으로 통합)
# ---------------------------------------------------------------------
st.session_state["__visible_widget_keys__"] = set()
with st.form("main_query_form"):

    ik_selected, ok_selected = {}, {}
    col_left, col_right = st.columns(2)

    need_base, need_extra = ([], [])
    if pair_id:
        need_base, need_extra = get_required_sets(udf, pair_id, base_side)

    # -----------------
    # 익산 패널
    # -----------------
    with col_left:
        st.subheader("익산")

        if has_both and pair_id:
            if basis_ik:
                ik_selected = _render_inputs_for_side(
                    udf, pair_id, "IK", ik_pt, need_base
                )
                st.caption(f"part_type: {ik_pt} (기준 측)")
            else:
                ik_selected = _render_inputs_for_side(
                    udf, pair_id, "IK", ik_pt, need_extra, tag_suffix=" (추가)"
                )
                st.caption(f"part_type: {ik_pt} (상대측 추가)")

        elif ik_only:
            ik_selected = _render_single_side_inputs("IK", ik_pt)
            st.caption(f"part_type: {ik_pt} (익산 전용 표준품)")

        else:
            st.caption(f"part_type: {ik_pt or '-'} (자동 조회 대상)")

    # -----------------
    # 옥천 패널
    # -----------------
    with col_right:
        st.subheader("옥천")

        if has_both and pair_id:
            if basis_ik:
                ok_selected = _render_inputs_for_side(
                    udf, pair_id, "OK", ok_pt, need_extra, tag_suffix=" (추가)"
                )
                st.caption(f"part_type: {ok_pt} (상대측 추가)")
            else:
                ok_selected = _render_inputs_for_side(
                    udf, pair_id, "OK", ok_pt, need_base
                )
                st.caption(f"part_type: {ok_pt} (기준 측)")

        elif ok_only:
            ok_selected = _render_single_side_inputs("OK", ok_pt)
            st.caption(f"part_type: {ok_pt} (옥천 전용 표준품)")

        elif ik_only:
            st.caption("해당 표준품은 옥천에 등록된 대응 코드가 없습니다.")

        else:
            st.caption(f"part_type: {ok_pt or '-'} (자동 조회 대상)")

    st.divider()

    # 폼용 Submit 버튼 (Enter로도 실행됨)
    do_query = st.form_submit_button("조회")
    
    # ✅ rerun이 발생해도 '조회 의도'를 유지하기 위한 래치(latch)
    # - submit이 눌린 run에서 __do_query__를 True로 올려둠
    # - 이후 룰 적용 과정에서 st.rerun()이 일어나도, 다음 run에서 do_query를 True로 복원 가능
    if do_query:
        st.session_state["__do_query__"] = True
    
        # ✅ (핵심) 조회를 새로 시작할 때마다 충돌 선택값 초기화
        st.session_state["__run_ik_overrides__"] = {}
        st.session_state["__run_ok_overrides__"] = {}
    
    # 최종 do_query는 session_state latch까지 포함
    do_query = bool(st.session_state.get("__do_query__", False))
# ============= 폼 밖에서 조회 실행 =============
if do_query:
    # --------------------------
    # IK + OK 모두 있는 경우
    # --------------------------
    if has_both:
        if not pair_id:
            st.error("IK/OK pair가 확정되지 않았습니다.")
            st.stop()

        # ✅ 현재 저장된(사용자 확정) overrides (shared lookup 전용)
        user_ik_over = st.session_state.get("__run_ik_overrides__", {}) or {}
        user_ok_over = st.session_state.get("__run_ok_overrides__", {}) or {}

        # ✅ shared lookup 판별용 schema map
        smap = build_schema_maps(udf, pair_id)
        IK_L2K = smap["IK_lookup_to_key"]
        OK_L2K = smap["OK_lookup_to_key"]

        def _is_shared_lookup_name(lk: str) -> bool:
            ikk = IK_L2K.get(lk)
            okk = OK_L2K.get(lk)
            return bool(ikk) and (ikk == okk)

        # 1) 입력값 수집
        attrs = {}
        attrs.update(ik_selected or {})
        attrs.update(ok_selected or {})
        attrs = _auto_fill_nominal(attrs)

        # 2) 룰 엔진 적용
        #    - 중요: 현재까지 사용자가 확정한 overrides(user_ik_over/user_ok_over)를 "현재값"으로 같이 전달
        rule_result = apply_rules_to_attrs(
            udf=udf,
            compiled=ATTR_RULES,
            pair_id=pair_id,
            base_side=base_side,
            attrs_in=attrs,
            ik_overrides_in=user_ik_over,
            ok_overrides_in=user_ok_over,
        )

        # 3) 룰이 자동으로 위젯값을 바꿔야 하면 pending → rerun
        #    (non-shared lookup만 updates에 담김)
        # 3) 룰이 자동으로 위젯값을 바꿔야 하면 pending → rerun
        if rule_result.updates:
            visible_keys = st.session_state.get("__visible_widget_keys__", set()) or set()
        
            # ✅ (핵심) 이번 run에서 실제로 화면에 있는 위젯만 rerun 대상으로 삼는다
            visible_updates = {k: v for k, v in rule_result.updates.items() if k in visible_keys}
            hidden_updates  = {k: v for k, v in rule_result.updates.items() if k not in visible_keys}
        
            # (1) 숨겨진 위젯 업데이트는 rerun 없이 session_state에만 반영 (루프 방지)
            for k, v in hidden_updates.items():
                st.session_state[k] = v
        
            # (2) 보이는 위젯 업데이트는 pending으로 넘기고 rerun (값이 바뀌는 경우에만)
            changed_visible = {k: v for k, v in visible_updates.items() if str(st.session_state.get(k, "")) != str(v)}
        
            if changed_visible:
                st.session_state["__pending_widget_updates__"] = dict(changed_visible)
                st.session_state["__pending_infos__"] = list(rule_result.infos)
                st.session_state["__pending_blockers__"] = list(rule_result.blockers)
                st.session_state["__pending_conflicts__"] = list(rule_result.conflicts)
        
                # ✅ overrides 누적
                st.session_state["__pending_ik_overrides__"] = {**user_ik_over, **dict(rule_result.ik_overrides)}
                st.session_state["__pending_ok_overrides__"] = {**user_ok_over, **dict(rule_result.ok_overrides)}
        
                st.session_state["__do_query__"] = True
                st.rerun()
        
            # ✅ visible 업데이트도 없고(hidden만 처리)면 rerun 안 하고 다음 단계(encode)로 진행

        # 4) 정보 메시지
        for msg in rule_result.infos:
            st.info(msg)

        # 5) 차단(조회 중단)
        if rule_result.blockers:
            for e in rule_result.blockers:
                st.error(e)
            st.stop()

        # 6) 충돌 발생 시: 이 run에서 UI 렌더하지 말고 "충돌 상태"만 저장하고 rerun
        #    (중요: 충돌 UI를 do_query 밖에서 렌더해야 '선택 적용'이 정상 처리됨)
        if rule_result.conflicts:
            st.session_state["__await_conflict__"] = True
            st.session_state["__active_conflicts__"] = list(rule_result.conflicts)
            st.session_state["__conflict_pair_id__"] = pair_id
            st.session_state["__conflict_base_side__"] = base_side

            # 조회 파이프라인은 잠시 중단
            st.session_state["__do_query__"] = False
            st.rerun()

        # 7) (중요) 충돌이 없을 때만 코드 생성
        attrs = rule_result.attrs

        miss_base  = missing_required_keys(udf, pair_id, base_side,  attrs)
        miss_other = missing_required_keys(udf, pair_id, other_side, attrs)

        if miss_base:
            st.error(f"기준({base_side}) 필수 누락: {miss_base}")
            st.stop()

        if miss_other:
            st.warning(f"상대({other_side}) 필수 누락: {miss_other} — 이 키들까지 입력하면 완전한 11자리 생성")

        # ✅ 최종 overrides = (사용자 확정 overrides) + (룰이 만든 overrides)
        final_ik_over = {**user_ik_over, **dict(rule_result.ik_overrides)}
        final_ok_over = {**user_ok_over, **dict(rule_result.ok_overrides)}

        ik_code, ok_code = encode_both(
            udf, pair_id, attrs,
            ik_overrides=final_ik_over,
            ok_overrides=final_ok_over,
        )

        if ik_code:
            st.success(f"IK 코드: `{ik_code}`")
        if ok_code:
            st.success(f"OK 코드: `{ok_code}`")
            
        st.session_state["__do_query__"] = False
# ---------------------------------------------------------------------
# ✅ 충돌 해결 UI는 do_query와 분리 (SPF_6 핵심 구조)
# - do_query가 False여도, "선택 적용" 클릭을 반드시 처리할 수 있어야 한다.
# ---------------------------------------------------------------------
if (
    st.session_state.get("__await_conflict__", False)
    and st.session_state.get("__conflict_pair_id__") == pair_id
    and has_both and pair_id
):
    conflicts = st.session_state.get("__active_conflicts__", []) or []
    if conflicts:
        st.warning("속성 자동매칭 중 충돌이 발견되었습니다. 아래에서 후보를 선택해 주세요.")
        lookups = load_lookups()

        # shared lookup 판별용 schema map
        smap = build_schema_maps(udf, pair_id)
        IK_L2K = smap["IK_lookup_to_key"]
        OK_L2K = smap["OK_lookup_to_key"]

        def _is_shared_lookup_name(lk: str) -> bool:
            ikk = IK_L2K.get(lk)
            okk = OK_L2K.get(lk)
            return bool(ikk) and (ikk == okk)

        # 현재 저장된(사용자 확정) overrides
        user_ik_over = st.session_state.get("__ik_overrides__", {}) or {}
        user_ok_over = st.session_state.get("__ok_overrides__", {}) or {}

        with st.form(f"conflict_form_outside::{pair_id}"):
            form_rows = []  # (Conflict, widget_key)

            for i, c in enumerate(conflicts, start=1):
                pt_for_lookup = ik_pt if c.side == "IK" else ok_pt
                system_local = "IK" if str(pt_for_lookup).upper().startswith("V") else "OK"
                opts = _merged_lookup_options(lookups, c.lookup, system_local, pt_for_lookup)

                side_kor = "익산" if c.side == "IK" else "옥천"
                kor_label = get_attr_label_kor(c.side, pt_for_lookup, c.key)

                # 기본값: shared lookup이면 side별 overrides 우선, 아니면 U: 위젯값
                cur_val = ""
                if _is_shared_lookup_name(c.lookup):
                    cur_val = (user_ik_over.get(c.key, "") if c.side == "IK" else user_ok_over.get(c.key, ""))
                if not cur_val:
                    cur_val = st.session_state.get(f"U:{pair_id}:{c.side}:{c.key}", "")

                default = cur_val if cur_val in c.candidates else c.candidates[0]

                wkey = f"CF_OUT:{pair_id}:{c.side}:{c.key}:{c.lookup}:{i}"
                st.selectbox(
                    f"{i}) {side_kor} / {kor_label} ({c.lookup}) — {c.reason}",
                    c.candidates,
                    index=c.candidates.index(default) if default in c.candidates else 0,
                    format_func=lambda code: f"{code} - {opts.get(code,'')}" if opts else code,
                    key=wkey,
                )
                form_rows.append((c, wkey))

            applied = st.form_submit_button("선택 적용", type="primary")

        if applied:
            new_user_ik = dict(user_ik_over)
            new_user_ok = dict(user_ok_over)

            pending = {}

            for c, wkey in form_rows:
                chosen_code = str(st.session_state.get(wkey, "")).strip()

                if _is_shared_lookup_name(c.lookup):
                    # shared → overrides로 저장
                    if c.side == "IK":
                        new_user_ik[c.key] = chosen_code
                    else:
                        new_user_ok[c.key] = chosen_code
                else:
                    # non-shared → 위젯 값으로 반영
                    pending[f"U:{pair_id}:{c.side}:{c.key}"] = chosen_code

            # ✅ 저장
            st.session_state["__run_ik_overrides__"] = new_user_ik
            st.session_state["__run_ok_overrides__"] = new_user_ok

            if pending:
                st.session_state["__pending_widget_updates__"] = pending

            # ✅ 충돌 상태 해제
            st.session_state["__await_conflict__"] = False
            st.session_state["__active_conflicts__"] = []
            st.session_state["__conflict_pair_id__"] = None

            # ✅ 다음 run에서 조회 파이프라인 재가동
            st.session_state["__pending_do_query__"] = True
            st.rerun()

        # 충돌 해결 전에는 아래로 내려가지 않게 중단
        st.stop()
# ---------------------------------------------------------------------
# 8) 이미지 출력 (좌=IK / 우=OK)
# ---------------------------------------------------------------------
# 한 줄에 몇 개씩 보여줄지 고정
COLS_PER_ROW = 5

import math  # 이미 있으면 생략

def render_images(part_code: str, site: str):
    imgs, used_key = find_images_with_prefix_fallback(
        part_code=part_code,
        site=site,
        base_dir="images",
        max_n=30,          # 최대 30장까지 표시 (15장이면 여유)
        min_prefix_len=3,
    )

    st.subheader("이미지")
    if not imgs:
        st.info("등록된 이미지가 없습니다.")
        return

    # -------------------------------
    #  상태: 현재 선택된 이미지 / 썸네일 페이지
    # -------------------------------
    sel_key = f"selected_image_{site}_{part_code}"
    page_key = f"thumb_page_{site}_{part_code}"

    if sel_key not in st.session_state:
        st.session_state[sel_key] = imgs[0].name  # 첫 이미지 기본 선택

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    per_page = 5  # ★ 한 페이지당 항상 5개 칸
    total = len(imgs)
    n_pages = max(1, math.ceil(total / per_page))

    cur_page = st.session_state[page_key]
    cur_page = max(0, min(cur_page, n_pages - 1))  # 방어 코드
    st.session_state[page_key] = cur_page

    start = cur_page * per_page
    end = min(start + per_page, total)

    # -------------------------------
    #  페이지 네비게이션 (◀ 1/2 ▶)
    # -------------------------------
    nav_l, nav_c, nav_r = st.columns([1, 2, 1])

    with nav_l:
        if st.button("◀", disabled=(cur_page == 0), key=f"prev_{site}_{part_code}"):
            st.session_state[page_key] = max(0, cur_page - 1)
            st.rerun()

    with nav_c:
        st.caption(f"{cur_page + 1} / {n_pages} 페이지")

    with nav_r:
        if st.button("▶", disabled=(cur_page >= n_pages - 1), key=f"next_{site}_{part_code}"):
            st.session_state[page_key] = min(n_pages - 1, cur_page + 1)
            st.rerun()

    # -------------------------------
    #  썸네일 그리드 (항상 5칸, 없으면 빈 칸)
    # -------------------------------
    cols = st.columns(per_page)
    thumb_height = 220  # 썸네일 세로 픽셀 고정 (원하는 값으로 조절 가능)
    
    for i in range(per_page):
        global_idx = start + i
        with cols[i]:
            if global_idx < total:
                img_path = imgs[global_idx]
                name = img_path.name
                is_selected = (st.session_state[sel_key] == name)
    
                # 위에 선택 버튼
                if st.button(
                    "선택됨" if is_selected else "선택",
                    key=f"pick_{site}_{part_code}_{name}",
                ):
                    st.session_state[sel_key] = name
                    st.rerun()
    
                # 썸네일: 세로 높이를 통일해서 리사이즈
                img = Image.open(img_path)
                w, h = img.size
                if h > 0:
                    new_w = int(w * (thumb_height / h))
                    img = img.resize((new_w, thumb_height))
    
                # width / use_container_width 없이 그대로 픽셀 크기로 표시
                st.image(
                    img,
                    caption=name,
                )
            else:
                # 이 페이지에 이미지가 5개 미만이면 나머지는 빈 칸
                st.write("")


    # -------------------------------
    #  크게 보기 selectbox (썸네일 선택과 동기화)
    # -------------------------------
    all_names = [p.name for p in imgs]
    cur_name = st.session_state[sel_key]
    try:
        cur_index = all_names.index(cur_name)
    except ValueError:
        cur_index = 0

    picked = st.selectbox(
        "크게 보기",
        all_names,
        index=cur_index,
        key=f"big_view_{site}_{part_code}",
    )

    # selectbox로 바꾼 것도 상태 반영
    if picked != st.session_state[sel_key]:
        st.session_state[sel_key] = picked

    big = next(p for p in imgs if p.name == st.session_state[sel_key])

    # -------------------------------
    #  큰 이미지 출력
    # -------------------------------
    st.image(
        str(big),
        use_container_width=True,   # 좌/우 컬럼 폭에 맞게 크게
    )


st.divider()
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("익산 이미지")
    if ik_pt:
        # ik_pt가 'V111' 형태여야 파일형(V111_1.jpg)과 매칭됩니다.
        render_images(part_code=ik_pt, site="IK")
    else:
        st.info("IK part_type 미선택")

with col_r:
    st.subheader("옥천 이미지")
    if ok_pt:
        render_images(part_code=ok_pt, site="OK")
    else:
        st.info("OK part_type 미선택")

st.session_state["__do_query__"] = False

st.caption(f"[DBG] auto_keys={sorted(list(auto_target_keys_from_rules(udf, pair_id, base_side, ATTR_RULES)))}")
