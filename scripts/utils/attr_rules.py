# utils/attr_rules.py
# -*- coding: utf-8 -*-

"""
속성간 매칭 룰 엔진 (SPF)
- data/매칭테이블.csv 를 MAP 룰 + CONSTRAINT(FIX/ALLOWLIST)로 컴파일
- 조회(조회 버튼 클릭) 시 attrs(dict)에 룰을 적용
- 충돌(후보 2개 이상)은 자동 결정하지 않고 conflict로 반환 → UI에서 사용자 선택
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 데이터 모델
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Conflict:
    """사용자 선택이 필요한 충돌"""
    side: str          # "IK" or "OK"
    key: str           # union_schema의 key (attr_name)
    lookup: str        # lookup table name
    candidates: List[str]  # 후보 코드들(문자열)
    reason: str        # 설명용


@dataclass
class RuleApplyResult:
    attrs: Dict[str, str]
    updates: Dict[str, str]     # session_state에 넣을 위젯키 → 값
    infos: List[str]            # 참고 메시지
    blockers: List[str]         # 조회 중단 에러 메시지
    conflicts: List[Conflict]   # 사용자 선택 필요


# ─────────────────────────────────────────────────────────────────────────────
# 룰 테이블 로드/컴파일
# ─────────────────────────────────────────────────────────────────────────────
def load_match_table_csv(path: str = "data/매칭테이블.csv") -> pd.DataFrame:
    """
    매칭테이블.csv 로드
    - dtype=str로 강제(앞자리 0 보존)
    - 익산품번은 대문자 정규화
    """
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    if "익산품번" in df.columns:
        df["익산품번"] = df["익산품번"].str.upper().str.strip()
    if "옥천품번" in df.columns:
        df["옥천품번"] = df["옥천품번"].str.strip()

    if "익산_코드" in df.columns:
        df["익산_코드"] = df["익산_코드"].str.strip()
    if "옥천_코드" in df.columns:
        df["옥천_코드"] = df["옥천_코드"].str.strip()

    df["pair_id"] = df["익산품번"] + "_" + df["옥천품번"]

    return df


def _row_type(r: pd.Series) -> str:
    ik_present = (r.get("익산속성_lookup", "") != "" and r.get("익산_코드", "") != "")
    ok_present = (r.get("옥천속성_lookup", "") != "" and r.get("옥천_코드", "") != "")
    if ik_present and ok_present:
        return "MAP"
    if ik_present and not ok_present:
        return "IK_ONLY"   # 제약/고정(익산측)
    if ok_present and not ik_present:
        return "OK_ONLY"   # 제약/고정(옥천측)
    return "EMPTY"


def compile_rules(df: pd.DataFrame) -> Dict:
    """
    매칭테이블을 '룰 엔진' 실행에 최적화된 dict로 컴파일한다.

    반환 구조(핵심만):
    - maps[direction][pair_id][(trigger_lookup, trigger_code)] -> list[(target_lookup, target_code), ...]
    - trigger_codes[direction][pair_id][trigger_lookup] -> set(codes)
    - allowed_codes[pair_id][side][lookup] -> set(codes)          # (요구사항 5) 허용값 검증용
    - constraints[pair_id][side][lookup] -> {"type": FIX/ALLOWLIST, "codes": set(...)}
    """
    df = df.copy()
    df["row_type"] = df.apply(_row_type, axis=1)

    maps = {"익산->옥천": defaultdict(lambda: defaultdict(list)),
            "옥천->익산": defaultdict(lambda: defaultdict(list))}
    trigger_codes = {"익산->옥천": defaultdict(lambda: defaultdict(set)),
                     "옥천->익산": defaultdict(lambda: defaultdict(set))}

    allowed_codes = defaultdict(lambda: {"IK": defaultdict(set), "OK": defaultdict(set)})
    constraints_raw = defaultdict(lambda: {"IK": defaultdict(set), "OK": defaultdict(set)})

    for _, r in df.iterrows():
        pair_id = r["pair_id"]
        direction = r.get("기준", "").strip()
        rt = r["row_type"]

        ik_lk = r.get("익산속성_lookup", "").strip()
        ik_cd = r.get("익산_코드", "").strip()
        ok_lk = r.get("옥천속성_lookup", "").strip()
        ok_cd = r.get("옥천_코드", "").strip()

        # (요구사항 5) 허용코드 집합: 테이블에 한 번이라도 등장한 코드는 '현실 허용 후보'로 간주
        if ik_lk and ik_cd:
            allowed_codes[pair_id]["IK"][ik_lk].add(ik_cd)
        if ok_lk and ok_cd:
            allowed_codes[pair_id]["OK"][ok_lk].add(ok_cd)

        if rt == "IK_ONLY":
            # 익산측 lookup/code가 있으면: FIX 또는 ALLOWLIST 후보 (코드 개수로 판정)
            constraints_raw[pair_id]["IK"][ik_lk].add(ik_cd)

        elif rt == "OK_ONLY":
            constraints_raw[pair_id]["OK"][ok_lk].add(ok_cd)

        elif rt == "MAP":
            if direction == "익산->옥천":
                trig = (ik_lk, ik_cd)
                act = (ok_lk, ok_cd)
                maps[direction][pair_id][trig].append(act)
                trigger_codes[direction][pair_id][ik_lk].add(ik_cd)

            elif direction == "옥천->익산":
                trig = (ok_lk, ok_cd)
                act = (ik_lk, ik_cd)
                maps[direction][pair_id][trig].append(act)
                trigger_codes[direction][pair_id][ok_lk].add(ok_cd)

    # 제약 타입(FIX/ALLOWLIST) 최종화
    constraints = defaultdict(lambda: {"IK": {}, "OK": {}})
    for pair_id, sides in constraints_raw.items():
        for side in ("IK", "OK"):
            for lk, codes in sides[side].items():
                codes = set([c for c in codes if str(c).strip() != ""])
                if not codes:
                    continue
                ctype = "FIX" if len(codes) == 1 else "ALLOWLIST"
                constraints[pair_id][side][lk] = {"type": ctype, "codes": codes}

    return {
        "maps": maps,
        "trigger_codes": trigger_codes,
        "allowed_codes": allowed_codes,
        "constraints": constraints,
    }


# ─────────────────────────────────────────────────────────────────────────────
# union_schema 기반: lookup ↔ key 매핑
# ─────────────────────────────────────────────────────────────────────────────
def build_schema_maps(udf: pd.DataFrame, pair_id: str) -> Dict:
    """
    union_schema(udf)에서 pair_id에 대해 lookup<->key 매핑을 만든다.
    lint에서 lookup_key_ambiguity가 0이었으므로 lookup->key는 1:1로 가정 가능.
    """
    block = udf[udf["pair_id"] == pair_id].copy()
    if block.empty:
        return {
            "IK_lookup_to_key": {},
            "OK_lookup_to_key": {},
            "IK_key_to_lookup": {},
            "OK_key_to_lookup": {},
        }

    for c in ("lookup", "key", "dtype"):
        if c in block.columns:
            block[c] = block[c].fillna("").astype(str).str.strip()

    ik_l2k = {}
    ok_l2k = {}

    # lookup이 있는(=lookup dtype) 속성만 대상으로
    # required_ik / required_ok가 bool로 들어가 있는 상태를 기대
    ik_rows = block[block.get("required_ik", False) == True]
    ok_rows = block[block.get("required_ok", False) == True]

    for _, r in ik_rows.iterrows():
        lk = r.get("lookup", "")
        k = r.get("key", "")
        if lk and k:
            ik_l2k[lk] = k

    for _, r in ok_rows.iterrows():
        lk = r.get("lookup", "")
        k = r.get("key", "")
        if lk and k:
            ok_l2k[lk] = k

    ik_k2l = {k: lk for lk, k in ik_l2k.items()}
    ok_k2l = {k: lk for lk, k in ok_l2k.items()}

    return {
        "IK_lookup_to_key": ik_l2k,
        "OK_lookup_to_key": ok_l2k,
        "IK_key_to_lookup": ik_k2l,
        "OK_key_to_lookup": ok_k2l,
    }


def widget_key(pair_id: str, side: str, key: str) -> str:
    """
    app.py의 _render_inputs_for_side 에서 쓰는 위젯 키 규칙과 동일하게 맞춘다.
    """
    return f"U:{pair_id}:{side}:{key}"


# ─────────────────────────────────────────────────────────────────────────────
# 룰 적용
# ─────────────────────────────────────────────────────────────────────────────
def apply_rules_to_attrs(
    udf: pd.DataFrame,
    compiled: Dict,
    pair_id: str,
    base_side: str,         # "IK" or "OK"
    attrs_in: Dict[str, str]
) -> RuleApplyResult:
    """
    조회 시점에 attrs에 룰 적용.
    - FIX 먼저 반영(강제)
    - (요구사항 5) 허용목록 위반이면 blockers에 추가
    - MAP 룰로 other_side 속성 자동 세팅
    - 충돌이면 conflicts로 반환(사용자 선택 필요)
    """
    base_side = base_side.upper()
    direction = "익산->옥천" if base_side == "IK" else "옥천->익산"
    other_side = "OK" if base_side == "IK" else "IK"

    attrs = {k: ("" if v is None else str(v).strip()) for k, v in (attrs_in or {}).items()}

    schema_maps = build_schema_maps(udf, pair_id)
    IK_L2K = schema_maps["IK_lookup_to_key"]
    OK_L2K = schema_maps["OK_lookup_to_key"]
    IK_K2L = schema_maps["IK_key_to_lookup"]
    OK_K2L = schema_maps["OK_key_to_lookup"]

    def l2k(side: str, lookup: str) -> Optional[str]:
        return IK_L2K.get(lookup) if side == "IK" else OK_L2K.get(lookup)

    def k2l(side: str, key: str) -> Optional[str]:
        return IK_K2L.get(key) if side == "IK" else OK_K2L.get(key)

    infos: List[str] = []
    blockers: List[str] = []
    conflicts: List[Conflict] = []
    updates: Dict[str, str] = {}

    # ── 1) 제약(FIX/ALLOWLIST) 적용/검증
    constraints = compiled["constraints"].get(pair_id, {"IK": {}, "OK": {}})

    for side in ("IK", "OK"):
        for lk, obj in constraints.get(side, {}).items():
            key = l2k(side, lk)
            if not key:
                continue

            ctype = obj["type"]
            codes = sorted(list(obj["codes"]))

            cur = attrs.get(key, "").strip()

            if ctype == "FIX":
                fixed = codes[0]
                if cur != fixed:
                    attrs[key] = fixed
                    updates[widget_key(pair_id, side, key)] = fixed
                    infos.append(f"[고정] {side}:{lk} 는 '{fixed}'로 고정되어 자동 설정되었습니다.")
            else:
                # ALLOWLIST: 사용자가 입력한 값이 허용목록 밖이면 차단
                if cur and cur not in obj["codes"]:
                    blockers.append(f"[허용값 위반] {side}:{lk} 에 '{cur}'를 입력했지만 허용값은 {codes} 입니다.")

    # ── 2) (요구사항 5) 테이블 기반 허용코드 검증
    allowed = compiled["allowed_codes"].get(pair_id, {"IK": {}, "OK": {}})

    for side in ("IK", "OK"):
        for lk, allowed_set in allowed.get(side, {}).items():
            key = l2k(side, lk)
            if not key:
                continue
            cur = attrs.get(key, "").strip()
            if cur and cur not in allowed_set:
                blockers.append(
                    f"[매칭테이블 미정의 코드] {side}:{lk} 의 '{cur}'는 매칭테이블에 정의된 코드가 아닙니다. "
                    f"(허용: {sorted(list(allowed_set))})"
                )

    # ── 3) 방향 기준 트리거 값이 "매칭 룰에 존재하는지" 검증 (없으면 '매칭되는게 없다')
    trig_side = base_side
    trig_codes_map = compiled["trigger_codes"][direction].get(pair_id, defaultdict(set))

    for trig_lk, allowed_trig_codes in trig_codes_map.items():
        trig_key = l2k(trig_side, trig_lk)
        if not trig_key:
            continue
        cur = attrs.get(trig_key, "").strip()
        if cur and cur not in allowed_trig_codes:
            blockers.append(
                f"[매칭 없음] {trig_side}:{trig_lk} 에 '{cur}'를 선택했지만 이 값에 대한 매칭 룰이 없습니다. "
                f"(가능: {sorted(list(allowed_trig_codes))})"
            )

    if blockers:
        return RuleApplyResult(attrs=attrs, updates=updates, infos=infos, blockers=blockers, conflicts=[])

    # ── 4) MAP 룰 적용(자동 세팅 후보 수집)
    maps = compiled["maps"][direction].get(pair_id, {})
    proposals: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)
    # proposals key: (side, key, lookup) -> set(codes)

    for (trig_lk, trig_cd), actions in maps.items():
        trig_key = l2k(trig_side, trig_lk)
        if not trig_key:
            continue
        cur = attrs.get(trig_key, "").strip()
        if cur != trig_cd:
            continue

        # 트리거 성립 → 액션 제안
        for (tgt_lk, tgt_cd) in actions:
            tgt_side = other_side
            tgt_key = l2k(tgt_side, tgt_lk)
            if not tgt_key:
                continue
            proposals[(tgt_side, tgt_key, tgt_lk)].add(str(tgt_cd).strip())

    # ── 5) 제안값 반영 / 충돌은 분리
    for (tgt_side, tgt_key, tgt_lk), cand_set in proposals.items():
        cand = sorted([c for c in cand_set if c != ""])
        if not cand:
            continue

        cur = attrs.get(tgt_key, "").strip()

        if len(cand) == 1:
            only = cand[0]
            if cur == "":
                attrs[tgt_key] = only
                updates[widget_key(pair_id, tgt_side, tgt_key)] = only
                infos.append(f"[자동매칭] {tgt_side}:{tgt_lk} = '{only}' 로 자동 설정되었습니다.")
            elif cur != only:
                # 사용자 입력/기존값과 충돌 → 선택 필요
                cands = sorted(set([cur, only]))
                conflicts.append(
                    Conflict(
                        side=tgt_side, key=tgt_key, lookup=tgt_lk,
                        candidates=cands,
                        reason=f"기존값 '{cur}' vs 자동매칭 '{only}' 충돌"
                    )
                )
        else:
            # 후보 2개 이상 → 선택 필요 (현재값도 후보에 포함)
            cands = set(cand)
            if cur:
                cands.add(cur)
            conflicts.append(
                Conflict(
                    side=tgt_side, key=tgt_key, lookup=tgt_lk,
                    candidates=sorted(cands),
                    reason="자동매칭 후보가 2개 이상"
                )
            )

    return RuleApplyResult(attrs=attrs, updates=updates, infos=infos, blockers=[], conflicts=conflicts)
