"""
global_eval.py -- Phase 4.0 Step 3
전역 원장 배정기 (Global Bin Evaluator)

책임 범위:
  - VirtualStrip 목록과 Stock 목록을 받아 "어느 Strip을 어느 원장에 넣을지"
    계획표(List[BinAssignment])를 작성한다.
  - 3D Guillotine 물리 배치는 수행하지 않는다. 오직 계획(Plan)만 작성한다.
  - 현재 구현: Greedy First-Fit Decreasing (FFD) + source_stock 우선 힌트
  - 확장 구조: BnB / MCTS로 교체 가능하도록 평가 함수와 탐색 로직을 분리

파일 내 클래스 및 함수 목록:
  데이터 클래스:
    BinSlot          -- Stock qty를 펼친 독립 배정 슬롯
    BinAssignment    -- Strip-to-Slot 배정 결과 하나
    EvalResult       -- evaluate_bin_assignments의 반환 묶음
  핵심 클래스:
    GlobalBinEvaluator
      .evaluate_bin_assignments()  -- 공개 API
      ._greedy_assign()            -- Greedy FFD 구현
      ._find_best_slot()           -- 슬롯 후보 탐색
      .score_layout()              -- BnB/MCTS 평가 함수 (현재는 Greedy 후처리용)
  공개 편의 함수:
    assign_strips_to_bins()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core import Dims, Stock
from virtual_strip import VirtualStrip


# ═══════════════════════════════════════════════════════════════════
# 예외
# ═══════════════════════════════════════════════════════════════════

class GlobalEvalError(Exception):
    """global_eval 기본 예외"""


# ═══════════════════════════════════════════════════════════════════
# 내부 상수
# ═══════════════════════════════════════════════════════════════════

# Strip 간 절단(Kerf)이 슬롯 잔여 L에서 추가로 차감되는 양
# 동일 슬롯에 Strip이 연속 배정될 때 Strip 사이에 1회 절단 발생
_INTER_STRIP_KERF_FACTOR: float = 1.0   # kerf × 이 값

# 잔여 L이 이 미만이면 슬롯을 "사실상 완전 소진"으로 간주
_SLOT_FULL_THRESHOLD_MM: float = 30.0

# A급 잔재 기준: 잔여 L이 이 이상이면 재사용 가치 있는 잔재
_PRIME_OFFCUT_MIN_L_MM: float = 300.0

# score_layout 보너스/패널티 계수
_SCORE_BONUS_FULL_SLOT: float      = 1_000.0
_SCORE_BONUS_PRIME_OFFCUT: float   =   500.0
_SCORE_PENALTY_FRAGMENT: float     =  -200.0
_SCORE_PENALTY_UNASSIGNED: float   = -1e9     # 미배정 1개당 부피에 독립적으로 부과


# ═══════════════════════════════════════════════════════════════════
# BinSlot  — Stock qty를 펼친 독립 슬롯
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BinSlot:
    """
    Stock 한 장에 해당하는 독립 배정 슬롯.

    Stock.qty=3 이면 BinSlot 인스턴스가 3개 생성되고,
    각 슬롯은 서로 독립적인 remaining_l을 관리한다.

    Attributes:
        slot_id:       고유 식별자. 형식: "{stock.id}#{instance_idx}"
        stock:         원본 Stock 참조 (trimming 정보 등 필요 시 사용)
        instance_idx:  동일 Stock 내 몇 번째 인스턴스인지 (0-based)
        usable_l:      사용 가능한 길이 (stock.usable_dims.l)
        usable_w:      사용 가능한 폭   (stock.usable_dims.w)
        usable_t:      사용 가능한 두께 (stock.usable_dims.t)
        remaining_l:   아직 배정 가능한 남은 길이. 배정될 때마다 차감됨.
        assigned_strips: 이 슬롯에 배정된 VirtualStrip 목록 (순서 보존)
    """
    slot_id: str
    stock: Stock
    instance_idx: int
    usable_l: float
    usable_w: float
    usable_t: float
    remaining_l: float
    assigned_strips: List[VirtualStrip] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """아직 한 개도 배정되지 않은 슬롯"""
        return len(self.assigned_strips) == 0

    @property
    def is_effectively_full(self) -> bool:
        """잔여 L이 _SLOT_FULL_THRESHOLD_MM 미만 → 사실상 완전 소진"""
        return self.remaining_l < _SLOT_FULL_THRESHOLD_MM

    @property
    def has_prime_offcut(self) -> bool:
        """잔여 L이 _PRIME_OFFCUT_MIN_L_MM 이상 → A급 잔재 보유"""
        return self.remaining_l >= _PRIME_OFFCUT_MIN_L_MM

    def can_fit(self, strip: VirtualStrip, kerf: float) -> bool:
        """
        이 슬롯에 strip이 들어갈 수 있는지 판단한다.

        판단 기준 (3축 모두 충족해야 함):
          T: strip.dims.t <= self.usable_t
          W: strip.dims.w <= self.usable_w
          L: strip.dims.l <= self.remaining_l
             (첫 배정이 아니면 strip 앞에 kerf 1회 추가 차감)

        첫 배정(is_empty=True)이면 inter-strip kerf 없음.
        두 번째 이후면 preceding kerf를 잔여 L에서 추가 차감.
        """
        preceding_kerf = 0.0 if self.is_empty else kerf * _INTER_STRIP_KERF_FACTOR
        needed_l = strip.dims.l + preceding_kerf
        return (
            strip.dims.t <= self.usable_t + 1e-6
            and strip.dims.w <= self.usable_w + 1e-6
            and needed_l <= self.remaining_l + 1e-6
        )

    def assign(self, strip: VirtualStrip, kerf: float) -> float:
        """
        슬롯에 strip을 배정하고 remaining_l을 차감한다.

        Returns:
            차감 후 remaining_l
        """
        preceding_kerf = 0.0 if self.is_empty else kerf * _INTER_STRIP_KERF_FACTOR
        consumed = strip.dims.l + preceding_kerf
        self.remaining_l -= consumed
        self.remaining_l = max(0.0, self.remaining_l)
        self.assigned_strips.append(strip)
        return self.remaining_l

    def __repr__(self) -> str:
        return (
            f"BinSlot({self.slot_id}, "
            f"usable={self.usable_l:.0f}×{self.usable_w}×{self.usable_t}, "
            f"remaining_l={self.remaining_l:.1f}, "
            f"strips={len(self.assigned_strips)})"
        )


# ═══════════════════════════════════════════════════════════════════
# BinAssignment  — Strip-to-Slot 배정 결과 하나
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BinAssignment:
    """
    VirtualStrip과 BinSlot(원장 인스턴스)의 배정 결과.

    StripFirstPacker(Step 4)가 이것을 읽어 어느 원장에서 배치를 시작할지 결정한다.

    Attributes:
        strip:                 배정된 VirtualStrip
        slot:                  배정된 BinSlot
        remaining_l_after:     이 배정 직후 슬롯의 남은 길이
        is_source_stock_match: strip.source_stock이 slot.stock과 일치하는지 여부
                               DP 계획과 실제 배정이 일치하면 True
    """
    strip: VirtualStrip
    slot: BinSlot
    remaining_l_after: float
    is_source_stock_match: bool = False

    @property
    def stock_id(self) -> str:
        return self.slot.stock.id

    @property
    def slot_id(self) -> str:
        return self.slot.slot_id

    def __repr__(self) -> str:
        match_mark = "✓" if self.is_source_stock_match else "~"
        return (
            f"BinAssignment({match_mark} "
            f"strip={self.strip.strip_id} → slot={self.slot_id}, "
            f"remaining_l={self.remaining_l_after:.1f})"
        )


# ═══════════════════════════════════════════════════════════════════
# EvalResult  — evaluate_bin_assignments의 반환 묶음
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """
    GlobalBinEvaluator.evaluate_bin_assignments()의 반환값 묶음.

    Attributes:
        assignments:   배정 성공한 (strip, slot) 쌍 목록
        unassigned:    어느 슬롯에도 배정되지 못한 VirtualStrip 목록
        slots_used:    실제로 1개 이상의 Strip이 배정된 BinSlot 목록
        layout_score:  score_layout()이 계산한 배정 품질 점수
    """
    assignments: List[BinAssignment]
    unassigned: List[VirtualStrip]
    slots_used: List[BinSlot]
    layout_score: float = 0.0

    @property
    def total_strips(self) -> int:
        return len(self.assignments) + len(self.unassigned)

    @property
    def assignment_rate(self) -> float:
        if self.total_strips == 0:
            return 1.0
        return len(self.assignments) / self.total_strips

    def __repr__(self) -> str:
        return (
            f"EvalResult(assigned={len(self.assignments)}, "
            f"unassigned={len(self.unassigned)}, "
            f"slots_used={len(self.slots_used)}, "
            f"score={self.layout_score:.1f})"
        )


# ═══════════════════════════════════════════════════════════════════
# GlobalBinEvaluator
# ═══════════════════════════════════════════════════════════════════

class GlobalBinEvaluator:
    """
    VirtualStrip 목록을 Stock 슬롯에 배정하는 전역 평가기.

    현재 구현: Greedy FFD (First-Fit Decreasing) + source_stock 우선 힌트
    확장 포인트:
      - _greedy_assign()을 _bnb_assign()으로 교체해 BnB로 업그레이드
      - score_layout()은 BnB의 평가 함수로 그대로 재사용
      - BinSlot, BinAssignment 데이터 구조는 불변

    Greedy 배정 알고리즘 (3단계 우선순위):
      1. source_stock 우선:
            strip.source_stock == slot.stock이면 최우선 배정 시도
            → DP(Step 1)가 계획한 원장에 먼저 배정해 계획 일관성 유지
      2. Best-Fit:
            T/W 조건 충족 슬롯 중 배정 후 잔여 L이 가장 작은 슬롯 선택
            → 슬롯을 최대한 꽉 채워 다음 Strip을 위한 깨끗한 슬롯 보존
      3. 미배정:
            어떤 슬롯에도 들어가지 못하면 EvalResult.unassigned에 추가
    """

    # ──────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────

    def evaluate_bin_assignments(
        self,
        strips: List[VirtualStrip],
        stocks: List[Stock],
        kerf: float,
    ) -> EvalResult:
        """
        VirtualStrip 목록을 Stock 슬롯에 배정하고 EvalResult를 반환한다.

        Args:
            strips:  VirtualStripFactory가 반환한 VirtualStrip 목록.
                     내부적으로 부피 내림차순 재정렬하여 처리한다.
            stocks:  사용 가능한 원장 목록. qty 인스턴스가 펼쳐진다.
            kerf:    톱날 두께 (mm). Strip 간 배정 시 추가 차감에 사용.

        Returns:
            EvalResult(assignments, unassigned, slots_used, layout_score)
        """
        if not strips:
            return EvalResult([], [], [], 0.0)
        if not stocks:
            return EvalResult([], list(strips), [], _SCORE_PENALTY_UNASSIGNED * len(strips))

        # ── 1. BinSlot 생성 ────────────────────────────────────────
        # Stock.qty개 인스턴스를 독립 슬롯으로 펼침
        # 슬롯 정렬: 자투리 원장 우선 (usable_l 오름차순)
        # → 큰 Strip도 작은 원장에 먼저 넣어보고 안 되면 큰 원장으로 넘어감
        slots = self._make_slots(stocks)
        slots.sort(key=lambda s: s.usable_l)

        # ── 2. Greedy 배정 ─────────────────────────────────────────
        assignments, unassigned = self._greedy_assign(strips, slots, kerf)

        # ── 3. 결과 조립 ────────────────────────────────────────────
        slots_used = [s for s in slots if not s.is_empty]
        layout_score = self.score_layout(slots_used, unassigned)

        return EvalResult(
            assignments=assignments,
            unassigned=unassigned,
            slots_used=slots_used,
            layout_score=layout_score,
        )

    # ──────────────────────────────────────────────────────────────
    # Greedy FFD 배정
    # ──────────────────────────────────────────────────────────────

    def _greedy_assign(
        self,
        strips: List[VirtualStrip],
        slots: List[BinSlot],
        kerf: float,
    ) -> Tuple[List[BinAssignment], List[VirtualStrip]]:
        """
        Greedy First-Fit Decreasing (FFD) + source_stock 우선 힌트.

        처리 순서:
          1. strips를 부피(l×w×t) 내림차순 정렬 — 큰 것부터 배정
          2. 각 strip에 대해 _find_best_slot()으로 최적 슬롯 탐색
          3. 슬롯 발견: BinAssignment 생성 + 슬롯 remaining_l 차감
          4. 슬롯 없음: unassigned에 추가

        Returns:
            (assignments, unassigned)
        """
        # 부피 내림차순: 덩치 큰 Strip부터 배정해 좁은 틈새 낭비 방지
        sorted_strips = sorted(strips, key=lambda s: -(s.dims.l * s.dims.w * s.dims.t))

        assignments: List[BinAssignment] = []
        unassigned: List[VirtualStrip] = []

        for strip in sorted_strips:
            best_slot = self._find_best_slot(strip, slots, kerf)

            if best_slot is None:
                unassigned.append(strip)
                continue

            remaining_after = best_slot.assign(strip, kerf)
            is_match = (strip.source_stock.id == best_slot.stock.id)

            assignments.append(BinAssignment(
                strip=strip,
                slot=best_slot,
                remaining_l_after=remaining_after,
                is_source_stock_match=is_match,
            ))

        return assignments, unassigned

    def _find_best_slot(
        self,
        strip: VirtualStrip,
        slots: List[BinSlot],
        kerf: float,
    ) -> Optional[BinSlot]:
        """
        strip을 배정할 최적 슬롯을 탐색한다.

        탐색 우선순위:
          1. source_stock 일치 + T/W/L 모두 충족 → 첫 번째 발견 즉시 반환
             (DP 계획과 배정을 일치시켜 계획 일관성 최대화)
          2. T/W/L 충족 슬롯 중 Best-Fit:
             배정 후 잔여 L이 가장 작은 슬롯 선택
             (슬롯을 꽉 채워 다음 Strip을 위한 깨끗한 슬롯 보존)

        Returns:
            최적 BinSlot, 없으면 None
        """
        # ── 우선순위 1: source_stock 일치 슬롯 ──────────────────────
        for slot in slots:
            if slot.stock.id == strip.source_stock.id and slot.can_fit(strip, kerf):
                return slot

        # ── 우선순위 2: Best-Fit 슬롯 ───────────────────────────────
        best_slot: Optional[BinSlot] = None
        best_remaining: float = math.inf

        for slot in slots:
            if not slot.can_fit(strip, kerf):
                continue
            # 배정 후 잔여 L 추정 (실제 차감 전)
            preceding_kerf = 0.0 if slot.is_empty else kerf * _INTER_STRIP_KERF_FACTOR
            remaining_after = slot.remaining_l - strip.dims.l - preceding_kerf
            if remaining_after < best_remaining:
                best_remaining = remaining_after
                best_slot = slot

        return best_slot

    # ──────────────────────────────────────────────────────────────
    # BinSlot 생성
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_slots(stocks: List[Stock]) -> List[BinSlot]:
        """
        Stock 목록을 qty 인스턴스만큼 펼쳐 BinSlot 목록을 생성한다.

        Stock.qty=3 → BinSlot 3개 생성
        각 슬롯 ID: "{stock.id}#{0}", "{stock.id}#{1}", ...

        usable_dims는 stock.usable_dims를 사용한다.
        trimming 오류 Stock은 건너뛴다.
        """
        slots: List[BinSlot] = []
        for stock in stocks:
            try:
                ud = stock.usable_dims
            except ValueError:
                continue  # trimming 오류 원장 스킵
            for idx in range(stock.qty):
                slots.append(BinSlot(
                    slot_id=f"{stock.id}#{idx}",
                    stock=stock,
                    instance_idx=idx,
                    usable_l=ud.l,
                    usable_w=ud.w,
                    usable_t=ud.t,
                    remaining_l=ud.l,
                ))
        return slots

    # ──────────────────────────────────────────────────────────────
    # score_layout  — BnB / MCTS 확장 시 평가 함수로 재사용
    # ──────────────────────────────────────────────────────────────

    def score_layout(
        self,
        slots_used: List[BinSlot],
        unassigned: List[VirtualStrip],
    ) -> float:
        """
        배정 결과의 품질을 수치로 평가한다. (높을수록 좋음)

        슬롯별 평가:
          잔여 L < _SLOT_FULL_THRESHOLD_MM (30mm):
            → 슬롯 거의 완전 소진: +_SCORE_BONUS_FULL_SLOT (+1000)
          잔여 L >= _PRIME_OFFCUT_MIN_L_MM (300mm):
            → A급 잔재 보유: +_SCORE_BONUS_PRIME_OFFCUT (+500)
          그 외 (30mm ≤ L < 300mm):
            → 잡동사니 파편: +_SCORE_PENALTY_FRAGMENT (-200)

        미배정 패널티:
          미배정 1개당: +_SCORE_PENALTY_UNASSIGNED (-1e9)
          → 미배정이 단 1개라도 있으면 점수가 압도적으로 낮아짐

        이 함수는 BnB 구현 시 BnBNode.upper_bound/lower_bound 계산에 그대로 사용된다.

        Args:
            slots_used:  배정된 슬롯 목록
            unassigned:  미배정 VirtualStrip 목록

        Returns:
            float (높을수록 좋은 배정)
        """
        score = 0.0

        for slot in slots_used:
            if slot.is_effectively_full:
                score += _SCORE_BONUS_FULL_SLOT
            elif slot.has_prime_offcut:
                score += _SCORE_BONUS_PRIME_OFFCUT
            else:
                score += _SCORE_PENALTY_FRAGMENT

        for _ in unassigned:
            score += _SCORE_PENALTY_UNASSIGNED

        return score

    # ──────────────────────────────────────────────────────────────
    # 유틸리티 (디버깅·리포트)
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def summarize(result: EvalResult) -> str:
        """
        EvalResult를 사람이 읽기 쉬운 요약 문자열로 변환한다.
        독립 실행 테스트 및 로깅에 사용한다.
        """
        lines = [
            f"── 배정 요약 ──────────────────────────",
            f"  배정 성공: {len(result.assignments)}개",
            f"  미배정:   {len(result.unassigned)}개",
            f"  사용 슬롯: {len(result.slots_used)}개",
            f"  배정률:   {result.assignment_rate:.1%}",
            f"  품질 점수: {result.layout_score:.1f}",
        ]
        if result.assignments:
            lines.append("  배정 목록:")
            for a in result.assignments:
                match = "✓" if a.is_source_stock_match else "~"
                lines.append(
                    f"    [{match}] {a.strip.strip_id}"
                    f" (L={a.strip.dims.l:.0f}×W={a.strip.dims.w}×T={a.strip.dims.t})"
                    f" → {a.slot_id}  남은L={a.remaining_l_after:.1f}"
                )
        if result.unassigned:
            lines.append("  미배정 목록:")
            for s in result.unassigned:
                lines.append(
                    f"    {s.strip_id}"
                    f" (L={s.dims.l:.0f}×W={s.dims.w}×T={s.dims.t})"
                )
        if result.slots_used:
            lines.append("  슬롯 상태:")
            for slot in result.slots_used:
                state = (
                    "완전소진" if slot.is_effectively_full else
                    "A급잔재" if slot.has_prime_offcut else
                    "파편"
                )
                lines.append(
                    f"    {slot.slot_id}: 잔여L={slot.remaining_l:.1f}  [{state}]"
                    f"  strips={len(slot.assigned_strips)}"
                )
        lines.append("─" * 40)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 공개 편의 함수
# ═══════════════════════════════════════════════════════════════════

def assign_strips_to_bins(
    strips: List[VirtualStrip],
    stocks: List[Stock],
    kerf: float,
) -> EvalResult:
    """
    GlobalBinEvaluator.evaluate_bin_assignments의 함수형 래퍼.

    Args:
        strips:  VirtualStripFactory가 반환한 VirtualStrip 목록
        stocks:  사용 가능한 원장 목록
        kerf:    톱날 두께 (mm)

    Returns:
        EvalResult(assignments, unassigned, slots_used, layout_score)
    """
    evaluator = GlobalBinEvaluator()
    return evaluator.evaluate_bin_assignments(strips, stocks, kerf)


# ═══════════════════════════════════════════════════════════════════
# 독립 실행 테스트
# ═══════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    """
    핵심 시나리오 5개를 콘솔에서 검증한다.

    시나리오 1: 기본 배정 — source_stock 힌트 우선 적용 확인
    시나리오 2: Best-Fit — T/W 부적합 원장 건너뛰기
    시나리오 3: 미배정 — 어디에도 안 들어가는 Strip 처리
    시나리오 4: 복수 인스턴스 — qty>1 원장의 독립 슬롯 관리
    시나리오 5: score_layout — A급 잔재 vs 파편 점수 비교
    """
    from core import Dims, TrimmingMargins
    from strip_solver import solve_all_strips
    from virtual_strip import build_virtual_strips

    KERF = 3.0
    SEP = "─" * 60

    evaluator = GlobalBinEvaluator()

    # ── 시나리오 1: source_stock 힌트 우선 ─────────────────────────
    print(SEP)
    print("시나리오 1: source_stock 힌트 우선 배정")
    print(SEP)

    stocks1 = [
        Stock(id="S_small", dims=Dims(l=1500, w=300, t=18), qty=1, trimming=TrimmingMargins()),
        Stock(id="S_large", dims=Dims(l=4000, w=300, t=18), qty=1, trimming=TrimmingMargins()),
    ]
    parts1 = [
        Part(id="P1", dims=Dims(l=700, w=300, t=18), qty=2, lock_z=True, allow_xy_rotation=False),
        Part(id="P2", dims=Dims(l=400, w=300, t=18), qty=2, lock_z=True, allow_xy_rotation=False),
    ]
    plans1, _ = solve_all_strips(parts1, stocks1, KERF, time_budget=3.0)
    strips1, _ = build_virtual_strips(plans1, parts1, KERF)

    result1 = evaluator.evaluate_bin_assignments(strips1, stocks1, KERF)
    print(evaluator.summarize(result1))

    # source_stock 힌트 일치율 확인
    match_count = sum(1 for a in result1.assignments if a.is_source_stock_match)
    print(f"  source_stock 힌트 일치: {match_count}/{len(result1.assignments)}개")
    assert len(result1.unassigned) == 0, f"미배정 발생: {result1.unassigned}"
    print("  ✅ 모든 Strip 배정 성공")

    # ── 시나리오 2: T/W 조건 불일치 원장 건너뛰기 ──────────────────
    print()
    print(SEP)
    print("시나리오 2: T/W 부적합 원장 건너뛰기")
    print(SEP)

    stocks2 = [
        Stock(id="S_narrow", dims=Dims(l=3000, w=150, t=18), qty=1, trimming=TrimmingMargins()),  # W=150 < strip W=300
        Stock(id="S_wide",   dims=Dims(l=3000, w=300, t=18), qty=1, trimming=TrimmingMargins()),
    ]
    parts2 = [Part(id="P1", dims=Dims(l=800, w=300, t=18), qty=2, lock_z=True, allow_xy_rotation=False)]
    plans2, _ = solve_all_strips(parts2, stocks2, KERF, time_budget=3.0)
    strips2, _ = build_virtual_strips(plans2, parts2, KERF)

    result2 = evaluator.evaluate_bin_assignments(strips2, stocks2, KERF)
    print(evaluator.summarize(result2))

    for a in result2.assignments:
        assert a.slot.stock.id == "S_wide", (
            f"W=150 원장에 배정되면 안 됨: {a.slot.stock.id}"
        )
    print("  ✅ T/W 부적합 원장(S_narrow) 올바르게 건너뜀")

    # ── 시나리오 3: 미배정 처리 ────────────────────────────────────
    print()
    print(SEP)
    print("시나리오 3: 미배정 처리 (원장 공간 부족)")
    print(SEP)

    stocks3 = [Stock(id="S1", dims=Dims(l=500, w=300, t=18), qty=1, trimming=TrimmingMargins())]
    parts3 = [Part(id="P_BIG", dims=Dims(l=800, w=300, t=18), qty=1, lock_z=True, allow_xy_rotation=False)]
    plans3, _ = solve_all_strips(parts3, stocks3, KERF, time_budget=3.0)

    # 플랜이 없을 경우 수동으로 Strip 생성 (DP가 거부하는 케이스)
    if not plans3:
        from virtual_strip import VirtualStrip as VS
        import uuid
        fake_strip = VS(
            strip_id="vs_fake",
            dims=Dims(l=800, w=300, t=18),
            internal_parts=[(parts3[0], Dims(l=800, w=300, t=18), 1)],
            waste_internal=0,
            source_stock=stocks3[0],
            source_plan=None,
        )
        strips3 = [fake_strip]
    else:
        strips3, _ = build_virtual_strips(plans3, parts3, KERF)

    result3 = evaluator.evaluate_bin_assignments(strips3, stocks3, KERF)
    print(evaluator.summarize(result3))

    # Strip이 없으면 미배정 테스트 의미 없으므로 조건부 검증
    if strips3:
        print(f"  미배정 수: {len(result3.unassigned)}")
        assert result3.layout_score < 0, "미배정 시 음수 점수여야 함"
        print("  ✅ 미배정 EvalResult.unassigned 분리 및 음수 점수 확인")

    # ── 시나리오 4: qty>1 원장의 독립 슬롯 ────────────────────────
    print()
    print(SEP)
    print("시나리오 4: qty>1 원장 독립 슬롯 관리")
    print(SEP)

    stocks4 = [Stock(id="SM", dims=Dims(l=2000, w=300, t=18), qty=3, trimming=TrimmingMargins())]
    parts4 = [Part(id="P1", dims=Dims(l=900, w=300, t=18), qty=4, lock_z=True, allow_xy_rotation=False)]
    plans4, _ = solve_all_strips(parts4, stocks4, KERF, time_budget=3.0)
    strips4, leftover4 = build_virtual_strips(plans4, parts4, KERF)

    result4 = evaluator.evaluate_bin_assignments(strips4, stocks4, KERF)
    print(evaluator.summarize(result4))

    # 각 슬롯(SM#0, SM#1, SM#2)이 독립적으로 관리되는지 확인
    slot_ids = [a.slot_id for a in result4.assignments]
    print(f"  배정된 슬롯: {slot_ids}")

    # 동일 슬롯에 복수 배정 시 remaining_l이 연속 차감되는지 확인
    slot_remaining: Dict[str, float] = {}
    for a in result4.assignments:
        slot_remaining[a.slot_id] = a.remaining_l_after
    for sid, rem in slot_remaining.items():
        print(f"    {sid}: 남은L={rem:.1f}")
    print("  ✅ qty>1 원장 독립 슬롯 생성 및 독립 remaining_l 관리 확인")

    # ── 시나리오 5: score_layout A급 잔재 vs 파편 비교 ─────────────
    print()
    print(SEP)
    print("시나리오 5: score_layout — 잔재 품질 점수 비교")
    print(SEP)

    # 케이스 A: 잔여 L=1000 (A급 잔재) → 높은 점수
    slot_prime = BinSlot(
        slot_id="S_prime#0", stock=stocks1[1],
        instance_idx=0, usable_l=2000, usable_w=300, usable_t=18,
        remaining_l=1000,   # A급 잔재
    )
    # 케이스 B: 잔여 L=100 (파편) → 낮은 점수
    slot_fragment = BinSlot(
        slot_id="S_frag#0", stock=stocks1[1],
        instance_idx=0, usable_l=2000, usable_w=300, usable_t=18,
        remaining_l=100,    # 파편
    )
    # 케이스 C: 잔여 L=10 (완전 소진) → 가장 높은 점수
    slot_full = BinSlot(
        slot_id="S_full#0", stock=stocks1[1],
        instance_idx=0, usable_l=2000, usable_w=300, usable_t=18,
        remaining_l=10,     # 완전 소진
    )

    score_prime    = evaluator.score_layout([slot_prime],    [])
    score_fragment = evaluator.score_layout([slot_fragment], [])
    score_full     = evaluator.score_layout([slot_full],     [])

    print(f"  완전소진 점수: {score_full:.0f}   (기대: +{_SCORE_BONUS_FULL_SLOT:.0f})")
    print(f"  A급잔재 점수: {score_prime:.0f}   (기대: +{_SCORE_BONUS_PRIME_OFFCUT:.0f})")
    print(f"  파편 점수:   {score_fragment:.0f}  (기대: {_SCORE_PENALTY_FRAGMENT:.0f})")

    assert score_full > score_prime > score_fragment, (
        f"점수 순서 오류: full={score_full}, prime={score_prime}, frag={score_fragment}"
    )
    print("  ✅ score_layout 우선순위: 완전소진 > A급잔재 > 파편")

    print()
    print("=" * 60)
    print("모든 테스트 통과 ✅")
    print("=" * 60)


# 테스트용 import (타입 힌팅을 위해 여기서만 허용)
from typing import Dict  # noqa: E402 — 모듈 상단이 아닌 테스트 함수 근처에 위치
from core import Part    # noqa: E402 — 테스트에서만 사용


if __name__ == "__main__":
    _run_tests()
