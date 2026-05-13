"""
virtual_strip.py -- Phase 4.0 Step 2
가상 띠장(VirtualStrip) 생성기

책임 범위:
  - StripPlan 목록을 VirtualStrip 객체 목록으로 변환 (VirtualStripFactory)
  - 전역 수량 풀(pool)을 관리하여 동일 부품이 여러 플랜에서 중복 소모되지 않도록 보장
  - 어떤 StripPlan에도 포함되지 못한 잔여 부품을 leftover_parts로 반환

설계 원칙:
  - core.py를 일절 수정하지 않는다. Dims, Part, Stock을 그대로 활용한다.
  - VirtualStrip.dims는 3D Packer가 Part처럼 취급할 수 있는 외형 치수이다.
  - 수량 관리의 단위는 Part.id 기준 전역 풀이다.
    (동일 Part가 회전으로 여러 PartGroup에 등록되더라도 실제 수량은 1개 풀에서만 차감)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core import Dims, Part, Stock
from strip_solver import PartGroup, StripPlan


# ═══════════════════════════════════════════════════════════════════
# 예외
# ═══════════════════════════════════════════════════════════════════

class VirtualStripError(Exception):
    """virtual_strip 기본 예외"""


class QuantityExceededError(VirtualStripError):
    """StripPlan이 요구하는 수량이 전역 풀의 잔여 수량을 초과할 때"""


# ═══════════════════════════════════════════════════════════════════
# 내부 상수
# ═══════════════════════════════════════════════════════════════════

# 길이 계산 시 부동소수점 오차 허용 범위 (mm)
_LENGTH_EPSILON: float = 0.5


# ═══════════════════════════════════════════════════════════════════
# VirtualStrip
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VirtualStrip:
    """
    현장의 '쪽내기 묶음' 하나를 나타내는 가상 부품 객체.

    3D Packer(packer.py)는 이 객체를 단일 Part처럼 취급한다.
    즉, dims(외형 치수)만 보고 배치하며 내부 구조는 모른다.

    내부적으로는 어떤 Part들이 어떤 순서로 배열되는지 보관하여
    3D 배치 완료 후 CNC 지시서 생성 시 활용된다.

    Attributes:
        strip_id:        고유 식별자 (UUID 8자리)
        dims:            외형 치수 (L=부품 총 길이+kerf 합, W=그룹폭, T=그룹두께)
                         3D Packer가 이 값을 기준으로 공간 적합성을 판단한다.
        internal_parts:  내부 부품 배열 순서 목록.
                         각 항목은 (Part 객체, 배치 방향 Dims, 수량) 의 튜플.
                         길이 내림차순 정렬 (긴 부품 먼저) — 현장 표준 재단 순서.
        waste_internal:  스트립 내부 잔재 길이 (mm).
                         원장 길이 - total_used_length 에 해당하는 빈 공간.
        source_stock:    이 스트립이 목표로 하는 원장.
                         3D Packer가 Best-Fit Bin 선택 시 힌트로 활용한다.
        source_plan:     이 스트립을 생성한 StripPlan 원본 참조 (디버깅·역추적용).
    """
    strip_id: str
    dims: Dims
    internal_parts: List[Tuple[Part, Dims, int]]   # (part, orientation_dims, qty)
    waste_internal: float
    source_stock: Stock
    source_plan: StripPlan

    def __repr__(self) -> str:
        parts_summary = ", ".join(
            f"{p.id}×{q}" for p, _, q in self.internal_parts
        )
        return (
            f"VirtualStrip(id={self.strip_id}, "
            f"dims={self.dims.l:.1f}×{self.dims.w}×{self.dims.t}, "
            f"parts=[{parts_summary}], "
            f"waste={self.waste_internal:.1f})"
        )

    @property
    def total_part_count(self) -> int:
        """스트립 내 총 부품 개수"""
        return sum(qty for _, _, qty in self.internal_parts)

    @property
    def part_ids(self) -> List[str]:
        """스트립 내 part_id 목록 (중복 포함)"""
        result = []
        for part, _, qty in self.internal_parts:
            result.extend([part.id] * qty)
        return result


# ═══════════════════════════════════════════════════════════════════
# VirtualStripFactory
# ═══════════════════════════════════════════════════════════════════

class VirtualStripFactory:
    """
    StripPlan 목록을 VirtualStrip 목록으로 변환하는 팩토리.

    핵심 책임:
      1. 전역 수량 풀 관리: 동일 Part가 여러 PartGroup에 등록되어 있어도
         실제 수량은 Part.id 단위의 단일 풀에서만 차감한다.
      2. 플랜 적용 가능성 검사: 플랜이 요구하는 수량이 풀 잔여분을 초과하면
         해당 플랜은 건너뛴다 (QuantityExceededError를 발생시키지 않고 스킵).
      3. leftover 반환: 모든 StripPlan 처리 후 풀에 qty > 0 이 남은 부품을
         leftover_parts로 반환한다.
    """

    def build_strips(
        self,
        plans: List[StripPlan],
        parts: List[Part],
        kerf: float,
    ) -> Tuple[List[VirtualStrip], List[Part]]:
        """
        StripPlan 목록을 VirtualStrip 목록으로 변환한다.

        Args:
            plans:  StripKnapsackSolver가 생성한 StripPlan 목록.
                    잔재 오름차순(waste_length 낮은 순)으로 정렬되어 있어야
                    가장 효율적인 플랜이 먼저 처리된다.
            parts:  원본 Part 목록 (전역 수량 풀 초기화에 사용).
            kerf:   톱날 두께 (mm). VirtualStrip 외형 치수 계산에 사용.

        Returns:
            (strips, leftover_parts)
            strips:          생성된 VirtualStrip 목록. 부피 내림차순 정렬.
            leftover_parts:  어떤 StripPlan에도 포함되지 못한 잔여 부품 목록.
                             qty > 0 인 경우만 포함. 3D Packer GRASP fallback 입력.

        처리 순서:
          1. 원본 Part 목록으로 전역 수량 풀 초기화
          2. plans 순서대로 각 플랜 처리
             a. 플랜 요구 수량 vs 풀 잔여 수량 검사 → 초과 시 스킵
             b. 풀에서 수량 차감
             c. VirtualStrip 생성 (외형 치수, internal_parts 계산)
          3. 풀에 남은 부품 → leftover_parts 조립
        """
        # ── 1. 전역 수량 풀 초기화 ─────────────────────────────────
        # key: part_id, value: 잔여 수량
        # Part 객체 참조도 보관해 leftover 조립 시 재사용
        pool: Dict[str, int] = {}
        part_by_id: Dict[str, Part] = {}
        for p in parts:
            if p.qty > 0:
                pool[p.id] = p.qty
                part_by_id[p.id] = p

        # ── 2. 플랜별 처리 ─────────────────────────────────────────
        strips: List[VirtualStrip] = []

        for plan in plans:
            # a. 수량 검사: 플랜 요구 수량이 풀 잔여분 이내인지 확인
            if not self._can_apply_plan(plan, pool):
                # 풀 잔여분 부족 → 이 플랜 스킵 (이미 다른 플랜에서 소모됨)
                continue

            # b. 풀에서 수량 차감
            self._consume_from_pool(plan, pool)

            # c. VirtualStrip 생성
            strip = self._build_single_strip(plan, part_by_id, kerf)
            strips.append(strip)

        # 부피 내림차순 정렬 (3D Packer가 큰 것부터 배치하도록)
        strips.sort(key=lambda s: -s.dims.volume)

        # ── 3. leftover 조립 ───────────────────────────────────────
        leftover_parts = self._collect_leftover(pool, part_by_id)

        return strips, leftover_parts

    # ──────────────────────────────────────────────────────────────
    # 수량 관리
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _can_apply_plan(plan: StripPlan, pool: Dict[str, int]) -> bool:
        """
        플랜이 요구하는 수량을 풀이 모두 공급할 수 있는지 검사한다.

        assigned_parts는 [(part_id, qty), ...] 형식이다.
        풀에 해당 part_id가 없거나 잔여 수량이 부족하면 False를 반환한다.

        주의: 동일 part_id가 assigned_parts에 중복 등장할 수 있으므로
              part_id별로 합산 후 비교한다.
        """
        required: Dict[str, int] = {}
        for pid, qty in plan.assigned_parts:
            required[pid] = required.get(pid, 0) + qty

        for pid, needed in required.items():
            available = pool.get(pid, 0)
            if available < needed:
                return False
        return True

    @staticmethod
    def _consume_from_pool(plan: StripPlan, pool: Dict[str, int]) -> None:
        """
        플랜의 assigned_parts 수량만큼 풀에서 차감한다.
        _can_apply_plan 통과 후 호출되므로 수량 부족은 발생하지 않는다.
        """
        for pid, qty in plan.assigned_parts:
            pool[pid] = pool.get(pid, 0) - qty

    # ──────────────────────────────────────────────────────────────
    # VirtualStrip 생성
    # ──────────────────────────────────────────────────────────────

    def _build_single_strip(
        self,
        plan: StripPlan,
        part_by_id: Dict[str, Part],
        kerf: float,
    ) -> VirtualStrip:
        """
        StripPlan 하나로부터 VirtualStrip 하나를 생성한다.

        외형 치수 계산:
          L = plan.total_used_length  (DP가 이미 (n-1)*kerf를 반영해 계산)
          W = plan.group.width
          T = plan.group.thickness

        internal_parts 순서:
          길이(effective_length) 내림차순 정렬 — 긴 부품 먼저 재단하는 현장 표준.
          동일 길이면 part_id 사전순으로 결정적(deterministic) 정렬.
        """
        group = plan.group

        # 그룹 내 part_id → effective_length 매핑 (이 방향에서의 L)
        eff_len_map: Dict[str, float] = {
            e.part_id: e.effective_length for e in group.entries
        }
        # 그룹 내 part_id → 방향 치수 매핑
        eff_dims_map: Dict[str, Dims] = {
            e.part_id: Dims(
                l=e.effective_length,
                w=e.effective_width,
                t=e.effective_thickness,
            )
            for e in group.entries
        }

        # internal_parts 구성 (길이 내림차순, 동일 길이면 part_id 사전순)
        assigned_sorted = sorted(
            plan.assigned_parts,
            key=lambda x: (-eff_len_map.get(x[0], 0.0), x[0]),
        )

        internal_parts: List[Tuple[Part, Dims, int]] = []
        for pid, qty in assigned_sorted:
            if qty <= 0:
                continue
            p_obj = part_by_id.get(pid)
            if p_obj is None:
                # 안전 방어: part_by_id에 없으면 group.entries에서 복원
                p_obj = self._recover_part_from_group(group, pid)
            if p_obj is None:
                continue  # 복원 불가 시 건너뜀 (비정상 플랜 방어)
            orientation = eff_dims_map.get(pid)
            if orientation is None:
                continue
            internal_parts.append((p_obj, orientation, qty))

        # 외형 치수
        strip_dims = Dims(
            l=plan.total_used_length,
            w=group.width,
            t=group.thickness,
        )

        return VirtualStrip(
            strip_id=_new_strip_id(),
            dims=strip_dims,
            internal_parts=internal_parts,
            waste_internal=plan.waste_length,
            source_stock=plan.target_stock,
            source_plan=plan,
        )

    @staticmethod
    def _recover_part_from_group(group: PartGroup, pid: str) -> Optional[Part]:
        """
        part_by_id 에 없는 경우 group.entries 에서 Part 객체를 복원한다.
        정상 흐름에서는 호출되지 않는 방어 코드.
        """
        for entry in group.entries:
            if entry.part_id == pid:
                return entry.part
        return None

    # ──────────────────────────────────────────────────────────────
    # leftover 조립
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _collect_leftover(
        pool: Dict[str, int],
        part_by_id: Dict[str, Part],
    ) -> List[Part]:
        """
        풀에 잔여 수량(qty > 0)이 남은 부품을 leftover_parts로 조립한다.

        반환된 Part 객체는 원본 Part를 qty만 수정한 새 인스턴스다.
        (원본 Part 객체를 직접 변경하지 않아 호출자의 상태를 오염시키지 않는다.)

        정렬: 부피 내림차순 (3D Packer GRASP fallback 입력 시 큰 부품 우선)
        """
        leftover: List[Part] = []

        for pid, remaining_qty in pool.items():
            if remaining_qty <= 0:
                continue
            original = part_by_id.get(pid)
            if original is None:
                continue

            # 원본 Part를 변경하지 않고 잔여 수량을 반영한 새 인스턴스 생성
            leftover_part = Part(
                id=original.id,
                dims=original.dims,
                qty=remaining_qty,
                lock_z=original.lock_z,
                allow_xy_rotation=original.allow_xy_rotation,
                priority=original.priority,
                color=original.color,
            )
            leftover.append(leftover_part)

        # 부피 내림차순 정렬
        leftover.sort(key=lambda p: -(p.dims.l * p.dims.w * p.dims.t))
        return leftover


# ═══════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ═══════════════════════════════════════════════════════════════════

def _new_strip_id() -> str:
    """VirtualStrip 고유 ID 생성 (UUID 8자리)"""
    return "vs_" + uuid.uuid4().hex[:8]


# ═══════════════════════════════════════════════════════════════════
# 공개 편의 함수
# ═══════════════════════════════════════════════════════════════════

def build_virtual_strips(
    plans: List[StripPlan],
    parts: List[Part],
    kerf: float,
) -> Tuple[List[VirtualStrip], List[Part]]:
    """
    VirtualStripFactory.build_strips의 함수형 래퍼.

    Args:
        plans:  StripKnapsackSolver가 반환한 StripPlan 목록.
                잔재 오름차순 정렬 상태를 유지해야 최적 플랜이 먼저 처리된다.
        parts:  원본 Part 목록.
        kerf:   톱날 두께 (mm).

    Returns:
        (strips, leftover_parts)
    """
    factory = VirtualStripFactory()
    return factory.build_strips(plans, parts, kerf)


# ═══════════════════════════════════════════════════════════════════
# 독립 실행 테스트
# ═══════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    """
    핵심 시나리오 4개를 콘솔에서 검증한다.

    시나리오 1: 기본 변환 — StripPlan → VirtualStrip 치수 정확성
    시나리오 2: 수량 관리 — 회전 부품이 두 플랜에서 중복 소모되지 않음
    시나리오 3: leftover — 플랜에 포함되지 않은 부품 반환
    시나리오 4: internal_parts 순서 — 길이 내림차순 정렬 확인
    """
    from core import Dims, TrimmingMargins
    from strip_solver import solve_all_strips

    KERF = 3.0
    SEP = "─" * 60

    # ── 시나리오 1: 기본 변환 ──────────────────────────────────
    print(SEP)
    print("시나리오 1: 기본 변환 — dims 정확성 검증")
    print(SEP)

    stocks1 = [Stock(id="S1", dims=Dims(l=3000, w=300, t=18), qty=1, trimming=TrimmingMargins())]
    parts1  = [
        Part(id="P1", dims=Dims(l=800, w=300, t=18), qty=3, lock_z=True, allow_xy_rotation=False),
        Part(id="P2", dims=Dims(l=600, w=300, t=18), qty=2, lock_z=True, allow_xy_rotation=False),
    ]

    plans1, _ = solve_all_strips(parts1, stocks1, KERF, time_budget=3.0)
    strips1, leftover1 = build_virtual_strips(plans1, parts1, KERF)

    print(f"  플랜 수: {len(plans1)}  →  생성된 스트립 수: {len(strips1)}")
    for s in strips1:
        print(f"  {s}")
        # dims.l 검증: plan.total_used_length 와 일치해야 함
        matched_plan = s.source_plan
        assert abs(s.dims.l - matched_plan.total_used_length) < _LENGTH_EPSILON, (
            f"dims.l={s.dims.l} != plan.total_used_length={matched_plan.total_used_length}"
        )
        assert s.dims.w == matched_plan.group.width, "dims.w 불일치"
        assert s.dims.t == matched_plan.group.thickness, "dims.t 불일치"
    print(f"  leftover: {[f'{p.id}×{p.qty}' for p in leftover1]}")
    print("  ✅ 외형 치수 정확성 검증 통과")

    # ── 시나리오 2: 회전 부품 수량 중복 소모 방지 ───────────────
    print()
    print(SEP)
    print("시나리오 2: 회전 부품 수량 관리 — 중복 소모 방지")
    print(SEP)

    stocks2 = [Stock(id="S1", dims=Dims(l=2400, w=500, t=18), qty=2, trimming=TrimmingMargins())]
    # P_ROT: allow_xy_rotation=True → 두 그룹(W=200, W=500) 에 등록
    parts2 = [
        Part(id="P_ROT", dims=Dims(l=500, w=200, t=18), qty=4,
             lock_z=True, allow_xy_rotation=True),
        Part(id="P_FIX", dims=Dims(l=700, w=500, t=18), qty=3,
             lock_z=True, allow_xy_rotation=False),
    ]

    plans2, groups2 = solve_all_strips(parts2, stocks2, KERF, time_budget=3.0)
    print(f"  그룹 수: {len(groups2)}")
    for g in groups2:
        print(f"    {g}")
    print(f"  플랜 수: {len(plans2)}")
    for p in plans2:
        print(f"    {p}")

    strips2, leftover2 = build_virtual_strips(plans2, parts2, KERF)

    # 실제 소모된 P_ROT 수량 합산
    total_p_rot_used = sum(
        qty
        for s in strips2
        for part, _, qty in s.internal_parts
        if part.id == "P_ROT"
    )
    total_p_fix_used = sum(
        qty
        for s in strips2
        for part, _, qty in s.internal_parts
        if part.id == "P_FIX"
    )
    leftover_rot = next((p.qty for p in leftover2 if p.id == "P_ROT"), 0)
    leftover_fix = next((p.qty for p in leftover2 if p.id == "P_FIX"), 0)

    print(f"  P_ROT 원본 qty=4 → 사용={total_p_rot_used}, leftover={leftover_rot}")
    print(f"  P_FIX 원본 qty=3 → 사용={total_p_fix_used}, leftover={leftover_fix}")
    assert total_p_rot_used + leftover_rot == 4, (
        f"P_ROT 수량 합계 오류: 사용{total_p_rot_used} + 잔여{leftover_rot} != 4"
    )
    assert total_p_fix_used + leftover_fix == 3, (
        f"P_FIX 수량 합계 오류: 사용{total_p_fix_used} + 잔여{leftover_fix} != 3"
    )
    print("  ✅ 수량 보존 검증 통과 (사용 + leftover == 원본 qty)")

    # ── 시나리오 3: leftover — 플랜 없는 부품 반환 ──────────────
    print()
    print(SEP)
    print("시나리오 3: leftover — 플랜에 못 들어간 부품 반환")
    print(SEP)

    # P_BIG은 원장보다 크므로 어떤 플랜에도 들어가지 못함
    stocks3 = [Stock(id="S1", dims=Dims(l=2000, w=200, t=18), qty=1, trimming=TrimmingMargins())]
    parts3 = [
        Part(id="P_SMALL", dims=Dims(l=500, w=200, t=18), qty=3,
             lock_z=True, allow_xy_rotation=False),
        Part(id="P_BIG",   dims=Dims(l=2500, w=200, t=18), qty=1,
             lock_z=True, allow_xy_rotation=False),  # 원장 2000L 초과
    ]

    plans3, _ = solve_all_strips(parts3, stocks3, KERF, time_budget=3.0)
    strips3, leftover3 = build_virtual_strips(plans3, parts3, KERF)

    print(f"  스트립 수: {len(strips3)}")
    print(f"  leftover: {[f'{p.id}×{p.qty}' for p in leftover3]}")

    p_big_in_leftover = any(p.id == "P_BIG" for p in leftover3)
    assert p_big_in_leftover, "P_BIG이 leftover에 없습니다"
    # 원본 Part 불변성 확인
    assert parts3[1].qty == 1, "원본 Part.qty가 변경됨 (불변성 위반)"
    print("  ✅ leftover 반환 및 원본 Part 불변성 검증 통과")

    # ── 시나리오 4: internal_parts 순서 ─────────────────────────
    print()
    print(SEP)
    print("시나리오 4: internal_parts — 길이 내림차순 정렬 확인")
    print(SEP)

    stocks4 = [Stock(id="S1", dims=Dims(l=3000, w=300, t=18), qty=1, trimming=TrimmingMargins())]
    parts4 = [
        Part(id="PA", dims=Dims(l=400, w=300, t=18), qty=2, lock_z=True, allow_xy_rotation=False),
        Part(id="PB", dims=Dims(l=900, w=300, t=18), qty=1, lock_z=True, allow_xy_rotation=False),
        Part(id="PC", dims=Dims(l=600, w=300, t=18), qty=2, lock_z=True, allow_xy_rotation=False),
    ]

    plans4, _ = solve_all_strips(parts4, stocks4, KERF, time_budget=3.0)
    strips4, leftover4 = build_virtual_strips(plans4, parts4, KERF)

    print(f"  스트립 수: {len(strips4)}")
    for s in strips4:
        print(f"  {s}")
        print(f"    internal_parts: {[(p.id, d.l, q) for p, d, q in s.internal_parts]}")
        # 길이 내림차순 확인
        lengths = [d.l for _, d, _ in s.internal_parts]
        assert lengths == sorted(lengths, reverse=True), (
            f"internal_parts가 길이 내림차순이 아님: {lengths}"
        )
    print(f"  leftover: {[f'{p.id}×{p.qty}' for p in leftover4]}")
    print("  ✅ internal_parts 길이 내림차순 정렬 검증 통과")

    print()
    print("=" * 60)
    print("모든 테스트 통과 ✅")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
