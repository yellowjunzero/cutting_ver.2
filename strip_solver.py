"""
strip_solver.py — Phase 4.0 Step 1
1D 배낭 문제(Knapsack DP) 기반 쪽내기(Strip) 조합 계획기

책임 범위:
  - 부품을 두께(T)·폭(W) 기준으로 그룹화 (PartGrouper)
  - 그룹 내 길이(L) 조합의 최적 DP 계산 (StripKnapsackSolver)
  - 원장별 최적 StripPlan 반환

이 파일은 3D 공간 배치를 일절 담당하지 않습니다.
core.py 의 Part, Stock 객체와 완전 호환됩니다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# core.py 임포트 (동일 패키지 내 위치 가정)
from core import Part, Stock


# ═══════════════════════════════════════════════════════════════════
# 예외
# ═══════════════════════════════════════════════════════════════════

class StripSolverError(Exception):
    """strip_solver 기본 예외"""


class NoFeasibleCombinationError(StripSolverError):
    """주어진 조건으로 유효한 조합을 찾을 수 없을 때"""


# ═══════════════════════════════════════════════════════════════════
# 내부 상수
# ═══════════════════════════════════════════════════════════════════

# mm 단위 정수화 시 사용하는 스케일 인수
# 0.1mm 해상도: 1.0mm → 10 단위
_MM_SCALE: int = 10

# exact DP 적용 최대 원장 길이 (mm)
# 이를 초과하면 FFD 근사로 전환
_DP_EXACT_MAX_LENGTH_MM: float = 6000.0

# 그룹화 시 T/W 허용 오차 기본값 (kerf 미제공 시 사용)
_DEFAULT_TW_TOLERANCE: float = 0.5


# ═══════════════════════════════════════════════════════════════════
# 데이터 클래스
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GroupKey:
    """
    T/W 쌍의 그룹 식별자.
    두께와 폭이 허용 오차 이내인 부품들을 같은 그룹으로 묶을 때 사용한다.
    실제 대표값(representative)은 첫 번째 등록 부품의 T/W를 사용한다.
    """
    thickness: float   # 대표 T 값 (mm)
    width: float       # 대표 W 값 (mm)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GroupKey):
            return NotImplemented
        return self.thickness == other.thickness and self.width == other.width

    def __hash__(self) -> int:
        return hash((self.thickness, self.width))

    def __repr__(self) -> str:
        return f"GroupKey(T={self.thickness}, W={self.width})"


@dataclass
class PartEntry:
    """
    그룹 내 부품 하나의 참가 항목.
    부품이 회전 가능하면 동일 Part 가 두 방향으로 등록될 수 있다.
    """
    part: Part
    effective_length: float   # 이 방향에서의 L 값 (mm)
    effective_width: float    # 이 방향에서의 W 값 (그룹 W와 일치해야 함)
    effective_thickness: float  # 이 방향에서의 T 값 (그룹 T와 일치해야 함)
    qty_remaining: int        # 아직 할당되지 않은 수량 (StripKnapsackSolver가 소모)

    @property
    def part_id(self) -> str:
        return self.part.id


@dataclass
class PartGroup:
    """
    T/W가 동일(허용 오차 이내)한 부품들의 묶음.
    이 그룹 전체가 하나의 1D 배낭 문제가 된다.
    """
    key: GroupKey
    entries: List[PartEntry] = field(default_factory=list)

    @property
    def thickness(self) -> float:
        return self.key.thickness

    @property
    def width(self) -> float:
        return self.key.width

    @property
    def total_part_count(self) -> int:
        return sum(e.qty_remaining for e in self.entries)

    def length_items(self) -> List[Tuple[float, str, int]]:
        """
        DP 입력용 아이템 목록 반환.
        Returns:
            List of (effective_length, part_id, qty_remaining)
        """
        return [(e.effective_length, e.part_id, e.qty_remaining) for e in self.entries]

    def __repr__(self) -> str:
        return (
            f"PartGroup(T={self.thickness}, W={self.width}, "
            f"parts={[e.part_id for e in self.entries]})"
        )


@dataclass
class StripPlan:
    """
    DP가 찾아낸 하나의 최적 길이 조합.
    VirtualStripFactory(Step 2)가 이것을 VirtualStrip으로 변환한다.
    """
    group: PartGroup
    target_stock: Stock

    # (part_id, qty) 형식의 할당 목록
    assigned_parts: List[Tuple[str, int]]

    # 실제 소모 길이 = sum(part_l * qty) + (n_parts - 1) * kerf
    total_used_length: float

    # 원장 잔재 길이 = stock.usable_dims.l - total_used_length
    waste_length: float

    # 이 플랜에 포함된 부품 총 개수
    part_count: int

    # 수율 (0.0 ~ 1.0)
    @property
    def yield_rate(self) -> float:
        stock_l = self.target_stock.usable_dims.l
        if stock_l <= 0:
            return 0.0
        return max(0.0, min(1.0, self.total_used_length / stock_l))

    @property
    def waste_ratio(self) -> float:
        return 1.0 - self.yield_rate

    def __repr__(self) -> str:
        return (
            f"StripPlan(stock={self.target_stock.id}, "
            f"used={self.total_used_length:.1f}, "
            f"waste={self.waste_length:.1f}, "
            f"parts={self.assigned_parts})"
        )


# ═══════════════════════════════════════════════════════════════════
# PartGrouper
# ═══════════════════════════════════════════════════════════════════

class PartGrouper:
    """
    입력된 Part 목록을 두께(T)·폭(W) 기준으로 그룹화한다.

    그룹화 규칙:
      1. lock_z=True  → T는 고정, L↔W 교환(allow_xy_rotation=True 시) 가능
         ∴ 각 Part 에 대해 (T, W) 방향과 (T, L) 방향 두 후보를 생성
      2. 두 후보 중 이미 존재하는 그룹 키와 허용 오차 이내면 해당 그룹에 등록
      3. 맞는 그룹이 없으면 새 그룹 생성
      4. lock_z=False  → 6가지 방향 모두 후보 (Part.allowed_orientations 활용)
    """

    def __init__(self, tolerance: float = _DEFAULT_TW_TOLERANCE) -> None:
        self.tolerance = tolerance

    # ──────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────

    def group_by_tw(
        self,
        parts: List[Part],
        kerf: float,
    ) -> List[PartGroup]:
        """
        parts 를 T/W 기준으로 그룹화하여 PartGroup 목록 반환.

        Args:
            parts:  배치할 Part 목록 (qty 포함)
            kerf:   톱날 두께 (mm). T/W 허용 오차 기준으로도 사용.

        Returns:
            List[PartGroup], 각 그룹은 T/W가 동일한 부품들로 구성됨.
            같은 Part 가 회전에 의해 두 그룹에 모두 등록될 수 있음.
        """
        tol = max(self.tolerance, kerf)   # kerf가 더 크면 kerf 기준 사용
        groups: List[PartGroup] = []

        for part in parts:
            if part.qty <= 0:
                continue

            candidate_orientations = part.allowed_orientations()  # [C4] 준수

            # 각 허용 방향에 대해 (T, W, L_eff) 후보 생성
            for dims_candidate in candidate_orientations:
                t_cand = dims_candidate.t
                w_cand = dims_candidate.w
                l_cand = dims_candidate.l

                matched_group = self._find_matching_group(groups, t_cand, w_cand, tol)

                if matched_group is None:
                    # 새 그룹 생성
                    new_key = GroupKey(thickness=t_cand, width=w_cand)
                    new_group = PartGroup(key=new_key)
                    new_group.entries.append(PartEntry(
                        part=part,
                        effective_length=l_cand,
                        effective_width=w_cand,
                        effective_thickness=t_cand,
                        qty_remaining=part.qty,
                    ))
                    groups.append(new_group)
                else:
                    # 기존 그룹에 이 방향의 Part 등록
                    # 이미 동일 part_id + 동일 effective_length 로 등록된 항목은 중복 추가하지 않음
                    if not self._already_registered(matched_group, part.id, l_cand):
                        matched_group.entries.append(PartEntry(
                            part=part,
                            effective_length=l_cand,
                            effective_width=w_cand,
                            effective_thickness=t_cand,
                            qty_remaining=part.qty,
                        ))

        # 항목이 없는 그룹 제거 (방어적)
        groups = [g for g in groups if g.entries]
        return groups

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    @staticmethod
    def _find_matching_group(
        groups: List[PartGroup],
        t: float,
        w: float,
        tol: float,
    ) -> Optional[PartGroup]:
        """T, W 값이 허용 오차 이내인 그룹을 찾는다."""
        for g in groups:
            if abs(g.thickness - t) <= tol and abs(g.width - w) <= tol:
                return g
        return None

    @staticmethod
    def _already_registered(group: PartGroup, part_id: str, length: float) -> bool:
        """동일 (part_id, effective_length) 쌍이 이미 그룹에 있는지 확인한다."""
        for e in group.entries:
            if e.part_id == part_id and math.isclose(e.effective_length, length, rel_tol=1e-6):
                return True
        return False


# ═══════════════════════════════════════════════════════════════════
# StripKnapsackSolver
# ═══════════════════════════════════════════════════════════════════

class StripKnapsackSolver:
    """
    PartGroup 하나를 받아 주어진 원장 길이 내에서
    잔재가 가장 적은 길이 조합을 1D DP로 찾는다.

    알고리즘 전략:
      - stock.usable_dims.l ≤ _DP_EXACT_MAX_LENGTH_MM  → 정수화 Exact DP
      - 초과                                            → First-Fit Decreasing 근사
    """

    def enumerate_combinations(
        self,
        group: PartGroup,
        stocks: List[Stock],
        kerf: float,
        time_budget: float = 5.0,
    ) -> List[StripPlan]:
        """
        그룹 내 부품을 각 원장에 대해 최적 조합 탐색 후 StripPlan 목록 반환.

        동일 T/W 그룹에서 여러 원장을 커버할 수 있는 모든 플랜을 계산한다.
        원장 수량(qty)에 맞게 동일 원장 대상의 플랜이 qty 개 까지 생성될 수 있다.

        Args:
            group:       대상 PartGroup
            stocks:      사용 가능한 원장 목록 (usable_dims 기준으로 필터링)
            kerf:        톱날 두께 (mm)
            time_budget: 전체 시간 예산 (초). 초과 시 현재까지의 결과 반환.

        Returns:
            List[StripPlan], 잔재 오름차순 정렬.
        """
        deadline = time.perf_counter() + time_budget
        plans: List[StripPlan] = []

        # 원장 T/W 필터: 그룹의 T, W가 원장 usable_dims 에 들어가야 함
        eligible_stocks = self._filter_eligible_stocks(group, stocks)
        if not eligible_stocks:
            return plans

        # 각 원장에 대해 DP 실행
        # 원장 qty 를 고려하여 동일 원장 유형을 qty 번 반복 (부품 소모 누적)
        remaining_qty: Dict[str, int] = {
            e.part_id: e.qty_remaining for e in group.entries
        }

        for stock in eligible_stocks:
            for _stock_instance in range(stock.qty):
                if time.perf_counter() > deadline:
                    break

                stock_l = stock.usable_dims.l

                # 현재 남은 부품으로 아이템 목록 구성
                items = self._build_items(group, remaining_qty, kerf)
                if not items:
                    break  # 이 그룹의 모든 부품이 소진됨

                # DP 실행
                if stock_l <= _DP_EXACT_MAX_LENGTH_MM:
                    chosen = self._dp_exact(items, stock_l, kerf)
                else:
                    chosen = self._ffd_approximate(items, stock_l, kerf)

                if not chosen:
                    break  # 넣을 수 있는 조합이 없음

                # StripPlan 생성
                plan = self._build_plan(group, stock, chosen, stock_l, kerf)
                plans.append(plan)

                # 소진된 수량 차감 (다음 원장 인스턴스에서 반영)
                for pid, qty in chosen:
                    remaining_qty[pid] = remaining_qty.get(pid, 0) - qty

            if time.perf_counter() > deadline:
                break

        # 잔재 오름차순 정렬
        plans.sort(key=lambda p: p.waste_length)
        return plans

    # ──────────────────────────────────────────
    # 핵심 알고리즘: Exact DP
    # ──────────────────────────────────────────

    def _dp_exact(
        self,
        items: List[Tuple[float, str, int]],   # (length_with_kerf, part_id, qty)
        capacity: float,
        kerf: float,
    ) -> List[Tuple[str, int]]:
        """
        정수화 Bounded Knapsack DP.

        아이템의 길이를 _MM_SCALE 배 정수화하여 DP 테이블을 구성한다.
        각 부품은 길이에 kerf 가 이미 포함된 값으로 전달된다
        (단, 마지막 부품은 kerf 불필요 — _build_items 에서 처리).

        목표: capacity 이하에서 사용 길이 최대화 (= 잔재 최소화).

        Returns:
            [(part_id, qty), ...] 형식의 최적 조합.
            빈 리스트 = 유효 조합 없음.
        """
        cap_int = int(math.floor(capacity * _MM_SCALE))
        if cap_int <= 0:
            return []

        # dp[c] = c 용량에서 달성 가능한 최대 사용 길이 (정수 단위)
        dp: List[int] = [0] * (cap_int + 1)
        # backtrack[c] = (item_index, qty_used) — 역추적용
        backtrack: List[Optional[Tuple[int, int]]] = [None] * (cap_int + 1)

        for idx, (item_len, part_id, max_qty) in enumerate(items):
            item_int = int(round(item_len * _MM_SCALE))
            if item_int <= 0:
                continue

            # Bounded Knapsack: 수량 제한이 있으므로 이진 분할(Binary Split)로 처리
            # 이진 분할: max_qty → {1, 2, 4, ..., 나머지} 묶음으로 분할
            bundles = self._binary_split_bundles(max_qty)

            for bundle_qty in bundles:
                bundle_len = item_int * bundle_qty
                # 0/1 Knapsack 으로 처리 (내림차순 순회)
                for c in range(cap_int, bundle_len - 1, -1):
                    new_val = dp[c - bundle_len] + bundle_len
                    if new_val > dp[c]:
                        dp[c] = new_val
                        backtrack[c] = (idx, bundle_qty)

        # 최적 용량에서 역추적
        best_cap = cap_int
        result: Dict[int, int] = {}   # item_idx → qty 합산
        c = best_cap
        while c > 0 and backtrack[c] is not None:
            item_idx, bundle_qty = backtrack[c]
            result[item_idx] = result.get(item_idx, 0) + bundle_qty
            item_int = int(round(items[item_idx][0] * _MM_SCALE))
            c -= item_int * bundle_qty

        if not result:
            return []

        # part_id 기준으로 합산
        part_qty: Dict[str, int] = {}
        for item_idx, qty in result.items():
            pid = items[item_idx][1]
            part_qty[pid] = part_qty.get(pid, 0) + qty

        return [(pid, qty) for pid, qty in part_qty.items()]

    # ──────────────────────────────────────────
    # 핵심 알고리즘: FFD 근사 (긴 원장용)
    # ──────────────────────────────────────────

    def _ffd_approximate(
        self,
        items: List[Tuple[float, str, int]],
        capacity: float,
        kerf: float,
    ) -> List[Tuple[str, int]]:
        """
        First-Fit Decreasing 근사.
        원장이 _DP_EXACT_MAX_LENGTH_MM 초과일 때 사용한다.

        아이템을 길이 내림차순으로 정렬 후,
        남은 용량에 들어가는 첫 아이템부터 순서대로 채운다.

        Returns:
            [(part_id, qty), ...] 형식의 조합.
        """
        remaining_cap = capacity
        part_qty: Dict[str, int] = {}

        # 길이 내림차순 정렬
        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

        for item_len, pid, max_qty in sorted_items:
            if item_len <= 0 or remaining_cap <= 0:
                continue
            # 이 아이템을 몇 개나 넣을 수 있는가?
            # 마지막 아이템은 kerf 없지만 보수적으로 kerf 포함 길이로 계산
            count = min(max_qty, int(math.floor(remaining_cap / item_len)))
            if count > 0:
                part_qty[pid] = part_qty.get(pid, 0) + count
                remaining_cap -= item_len * count

        return [(pid, qty) for pid, qty in part_qty.items()]

    # ──────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────

    @staticmethod
    def _binary_split_bundles(max_qty: int) -> List[int]:
        """
        Bounded Knapsack 을 0/1 Knapsack 으로 변환하기 위한 이진 분할.
        예: max_qty=7 → [1, 2, 4] (합=7), max_qty=10 → [1, 2, 4, 3]
        """
        bundles: List[int] = []
        remaining = max_qty
        k = 1
        while k <= remaining:
            bundles.append(k)
            remaining -= k
            k *= 2
        if remaining > 0:
            bundles.append(remaining)
        return bundles

    def _filter_eligible_stocks(
        self,
        group: PartGroup,
        stocks: List[Stock],
    ) -> List[Stock]:
        """
        그룹의 T, W 가 원장 usable_dims 에 들어가는 원장만 선택.
        길이 방향(L)은 DP 에서 처리하므로 여기서 필터하지 않는다.
        """
        eligible = []
        for stock in stocks:
            try:
                ud = stock.usable_dims
            except ValueError:
                continue  # 트리밍 오류 원장 스킵
            # 그룹 T, W 가 원장의 W, T 공간에 들어가야 함
            if ud.t >= group.thickness and ud.w >= group.width:
                eligible.append(stock)
        return eligible

    def _build_items(
        self,
        group: PartGroup,
        remaining_qty: Dict[str, int],
        kerf: float,
    ) -> List[Tuple[float, str, int]]:
        """
        현재 남은 수량 기준 DP 입력 아이템 목록을 생성한다.

        각 아이템의 길이에 kerf 를 더한다.
        (스트립 내 부품 사이에 각각 하나의 절단이 발생하므로
         부품 n 개에 kerf n-1 개가 필요하지만,
         DP 에서는 단순하게 모든 아이템에 kerf 를 포함시키고
         _build_plan 에서 마지막 아이템의 kerf 를 차감하여 보정한다.)

        Returns:
            List of (length_with_kerf, part_id, qty)
        """
        items = []
        for entry in group.entries:
            pid = entry.part_id
            qty = remaining_qty.get(pid, 0)
            if qty <= 0:
                continue
            length_with_kerf = entry.effective_length + kerf
            items.append((length_with_kerf, pid, qty))
        return items

    def _build_plan(
        self,
        group: PartGroup,
        stock: Stock,
        chosen: List[Tuple[str, int]],
        stock_l: float,
        kerf: float,
    ) -> StripPlan:
        """
        DP 결과로 StripPlan 을 생성한다.

        실제 소모 길이:
          sum(length_i * qty_i) + (total_count - 1) * kerf
          (kerf 는 부품 사이 절단 수 = 총 부품 수 - 1)
        """
        # 부품 ID → effective_length 매핑
        len_map: Dict[str, float] = {
            e.part_id: e.effective_length for e in group.entries
        }

        total_parts = sum(qty for _, qty in chosen)
        total_length_parts = sum(
            len_map.get(pid, 0.0) * qty for pid, qty in chosen
        )
        # 부품 간 kerf (마지막 부품 뒤에는 절단 없음)
        total_kerf = kerf * max(0, total_parts - 1)
        total_used = total_length_parts + total_kerf
        waste = max(0.0, stock_l - total_used)

        return StripPlan(
            group=group,
            target_stock=stock,
            assigned_parts=list(chosen),
            total_used_length=total_used,
            waste_length=waste,
            part_count=total_parts,
        )


# ═══════════════════════════════════════════════════════════════════
# 공개 편의 함수
# ═══════════════════════════════════════════════════════════════════

def solve_all_strips(
    parts: List[Part],
    stocks: List[Stock],
    kerf: float,
    tw_tolerance: float = _DEFAULT_TW_TOLERANCE,
    time_budget: float = 5.0,
) -> Tuple[List[StripPlan], List[PartGroup]]:
    """
    모든 Part 에 대해 T/W 그룹화 + DP 조합을 한 번에 실행하는 편의 함수.

    Args:
        parts:        배치 대상 부품 목록
        stocks:       사용 가능한 원장 목록
        kerf:         톱날 두께 (mm)
        tw_tolerance: T/W 허용 오차 (mm). 기본 0.5mm.
        time_budget:  전체 계산 시간 예산 (초)

    Returns:
        (plans, groups)
        plans:  모든 그룹에 대한 StripPlan 목록 (잔재 오름차순)
        groups: 그룹화 결과 목록 (디버깅·Step 2 용)
    """
    grouper = PartGrouper(tolerance=tw_tolerance)
    groups = grouper.group_by_tw(parts, kerf)

    solver = StripKnapsackSolver()
    per_group_budget = time_budget / max(1, len(groups))

    all_plans: List[StripPlan] = []
    for group in groups:
        plans = solver.enumerate_combinations(
            group, stocks, kerf, time_budget=per_group_budget
        )
        all_plans.extend(plans)

    all_plans.sort(key=lambda p: p.waste_length)
    return all_plans, groups


# ═══════════════════════════════════════════════════════════════════
# 독립 실행 테스트 (python strip_solver.py 로 실행)
# ═══════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    """
    핵심 시나리오 3개를 콘솔에서 검증한다.

    시나리오 1: 긴 띠장 + 짧은 부품 — 띠장이 단독 Strip 을 가져야 함
    시나리오 2: 완전 채움 — 잔재 0 달성 확인
    시나리오 3: 회전 가능 부품 — 두 그룹에 등록되는지 확인
    """
    from core import Dims, TrimmingMargins

    KERF = 3.0
    SEP = "─" * 60

    # ── 시나리오 1: 긴 띠장 격리 ──────────────────────────────
    print(SEP)
    print("시나리오 1: 긴 띠장(3000L) + 짧은 부품(1000L) 격리 테스트")
    print(SEP)

    stock_3000 = Stock(id="S3000", dims=Dims(l=3000, w=100, t=18), qty=3)
    part_strip = Part(
        id="STRIP_3000",
        dims=Dims(l=3000, w=100, t=18),
        qty=1,
        allow_xy_rotation=False,
    )
    part_short = Part(
        id="SHORT_1000",
        dims=Dims(l=1000, w=100, t=18),
        qty=5,
        allow_xy_rotation=False,
    )

    plans1, groups1 = solve_all_strips(
        parts=[part_strip, part_short],
        stocks=[stock_3000],
        kerf=KERF,
        time_budget=5.0,
    )

    print(f"  그룹 수: {len(groups1)}")
    for g in groups1:
        print(f"    {g}")
    print(f"  플랜 수: {len(plans1)}")
    for p in plans1:
        print(f"    {p}")

    # 띠장 단독 플랜 확인
    strip_solo = [
        p for p in plans1
        if p.assigned_parts == [("STRIP_3000", 1)] and p.waste_length < 5.0
    ]
    assert strip_solo, "❌ 띠장 단독 플랜(잔재 ≈ 0)을 찾지 못했습니다!"
    print(f"  ✅ 띠장 단독 플랜 발견: waste={strip_solo[0].waste_length:.1f}mm")

    # ── 시나리오 2: 완전 채움 ────────────────────────────────
    print()
    print(SEP)
    print("시나리오 2: 완전 채움 — 잔재 0 달성")
    print(SEP)

    # 원장 3000mm, 부품 3개 × 997mm + kerf 2개 × 3mm = 2991 + 6 = 2997 ≠ 3000
    # 부품 3개 × 997mm = 2991 + kerf 2개 × 3 = 6 → 합계 2997 (잔재 3)
    # 완전 채움을 위해: 1000 + 1000 + 1000 - 2*kerf(3) = 2994 → 잔재 6
    # 부품 L=994, qty=3 → 994*3 + 3*2 = 2982+6 = 2988 (잔재 12)
    # 완전 채움: L = (3000 - 2*3) / 3 = 998   → 998*3 + 3*2 = 2994+6 = 3000 ✓
    stock_exact = Stock(id="S_EXACT", dims=Dims(l=3000, w=200, t=15), qty=1)
    part_exact = Part(
        id="P_EXACT",
        dims=Dims(l=998, w=200, t=15),
        qty=3,
        allow_xy_rotation=False,
    )

    plans2, _ = solve_all_strips(
        parts=[part_exact],
        stocks=[stock_exact],
        kerf=KERF,
        time_budget=5.0,
    )

    print(f"  플랜 수: {len(plans2)}")
    for p in plans2:
        print(f"    {p}")

    zero_waste = [p for p in plans2 if p.waste_length < 1.0]
    assert zero_waste, "❌ 잔재 0 플랜을 찾지 못했습니다!"
    print(f"  ✅ 완전 채움 플랜 발견: waste={zero_waste[0].waste_length:.1f}mm")

    # ── 시나리오 3: 회전 가능 부품 ───────────────────────────
    print()
    print(SEP)
    print("시나리오 3: 회전 가능 부품 → 두 그룹 등록 확인")
    print(SEP)

    stock_r = Stock(id="S_ROT", dims=Dims(l=2400, w=300, t=18), qty=2)
    part_rotatable = Part(
        id="P_ROT",
        dims=Dims(l=500, w=200, t=18),
        qty=4,
        allow_xy_rotation=True,   # L↔W 교환 가능
        lock_z=True,
    )

    _, groups3 = solve_all_strips(
        parts=[part_rotatable],
        stocks=[stock_r],
        kerf=KERF,
        time_budget=5.0,
    )

    print(f"  그룹 수: {len(groups3)} (회전으로 인해 2개 예상)")
    for g in groups3:
        for e in g.entries:
            print(f"    그룹 T={g.thickness} W={g.width} — "
                  f"part={e.part_id} effective_L={e.effective_length}")

    assert len(groups3) == 2, (
        f"❌ 회전 가능 부품이 2개 그룹에 등록되어야 합니다. 실제: {len(groups3)}"
    )
    print("  ✅ 회전 부품이 두 방향 그룹에 올바르게 등록됨")

    print()
    print("=" * 60)
    print("모든 테스트 통과 ✅")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
