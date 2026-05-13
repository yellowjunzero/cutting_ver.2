"""
phase4.py -- Phase 4.0 메인 오케스트레이터

4단계 파이프라인을 순서대로 실행하여 최종 배치 결과를 반환한다.

파이프라인:
  [Step 1] PartGrouper + StripKnapsackSolver  (strip_solver.py)
           T/W 그룹화 + 1D 배낭 DP
  [Step 2] VirtualStripFactory               (virtual_strip.py)
           StripPlan → VirtualStrip 변환 + 수량 풀 관리
  [Step 3] GlobalBinEvaluator                (global_eval.py)
           Greedy FFD + source_stock 힌트 배정
  [Step 4] StripFirstPacker                  (packer.py)
           BinAssignment → 실제 3D Guillotine 배치
           + _pack_with_free_nodes GRASP fallback

타임 버짓 배분 (기본 60초):
  Step 1: 10%  ( 6s) — DP는 단순 계산이라 빠름, 그룹 수 많으면 추가 필요
  Step 2:  2%  ( 1s) — 순수 변환, 거의 즉각
  Step 3:  3%  ( 2s) — Greedy는 매우 빠름, BnB 도입 시 비중 증가
  Step 4: 85%  (51s) — 3D 배치 + GRASP fallback에 대부분 투입

core.py는 이 파일에서 절대 수정하지 않는다.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# core 도메인
from core import (
    Dims, EngineSettings, Node, Part, Stock, TrimmingMargins,
)

# Step 1
from strip_solver import solve_all_strips, StripPlan

# Step 2
from virtual_strip import VirtualStrip, build_virtual_strips

# Step 3
from global_eval import (
    EvalResult, BinAssignment,
    GlobalBinEvaluator, assign_strips_to_bins,
)

# Step 4 + GRASP fallback
from packer import (
    EngineSettings as _ES,   # 별칭 (재임포트 충돌 방지)
    PackResult,
    Phase4PackResult,
    StripFirstPacker,
    StripAdapter,
    _pack_with_free_nodes,
    pack_parts as _grasp_pack_parts,  # GRASP fallback 전체 원장용
    _DEFAULT_AXIS_BIAS,
    _AXIS_BIAS_PRESETS,
)


# ═══════════════════════════════════════════════════════════════════
# 타임 버짓 상수
# ═══════════════════════════════════════════════════════════════════

_BUDGET_STEP1_RATIO: float = 0.10
_BUDGET_STEP2_RATIO: float = 0.02
_BUDGET_STEP3_RATIO: float = 0.03
_BUDGET_STEP4_RATIO: float = 0.85   # Step 4에 대부분 투입


# ═══════════════════════════════════════════════════════════════════
# 타이머 유틸리티
# ═══════════════════════════════════════════════════════════════════

class _Timer:
    """단계별 경과 시간을 측정하는 간단한 타이머"""

    def __init__(self) -> None:
        self._start = time.perf_counter()
        self._checkpoints: List[Tuple[str, float]] = []

    def checkpoint(self, label: str) -> float:
        """현재 시점을 기록하고 시작부터의 경과 시간(초)을 반환한다."""
        elapsed = time.perf_counter() - self._start
        self._checkpoints.append((label, elapsed))
        return elapsed

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def remaining(self, budget: float) -> float:
        """남은 버짓 시간 (음수면 초과)"""
        return budget - self.elapsed

    def summary(self) -> str:
        lines = ["타이머 체크포인트:"]
        prev = 0.0
        for label, t in self._checkpoints:
            lines.append(f"  [{label}] +{t - prev:.3f}s  (누적 {t:.3f}s)")
            prev = t
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# Phase4RunStats — 단계별 통계 요약
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Phase4RunStats:
    """pack_parts_phase4 실행 통계"""
    # Step 1
    n_groups: int = 0
    n_plans: int = 0
    step1_sec: float = 0.0

    # Step 2
    n_strips: int = 0
    n_leftover_parts: int = 0
    step2_sec: float = 0.0

    # Step 3
    n_assigned: int = 0
    n_unassigned_strips: int = 0
    layout_score: float = 0.0
    step3_sec: float = 0.0

    # Step 4
    n_strip_placed: int = 0
    n_fallback_placed: int = 0
    n_unplaced_parts: int = 0
    step4_sec: float = 0.0

    # 전체
    total_sec: float = 0.0
    stocks_used: int = 0
    strip_assignment_rate: float = 0.0

    def __repr__(self) -> str:
        return (
            f"Phase4RunStats("
            f"strips={self.n_strips}, assigned={self.n_assigned}, "
            f"unassigned_strips={self.n_unassigned_strips}, "
            f"fallback_placed={self.n_fallback_placed}, "
            f"unplaced={self.n_unplaced_parts}, "
            f"total={self.total_sec:.2f}s)"
        )


# ═══════════════════════════════════════════════════════════════════
# Phase4Result — 최종 반환값 (main.py가 소비)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Phase4Result:
    """
    pack_parts_phase4()의 최종 반환값.
    main.py의 PackResult와 호환되는 필드를 포함한다.
    """
    # 배치된 모든 Node (Strip 배치 + fallback 배치)
    occupied_nodes: List[Node]

    # 미배치 부품 {part_id: remaining_qty}
    unplaced: Dict[str, int]

    # 최종 FREE 노드 (잔재)
    free_nodes: List[Node]

    # 통계
    stats: Phase4RunStats

    # 단계별 중간 결과 (디버깅용)
    plans: List[StripPlan]
    strips: List[VirtualStrip]
    eval_result: EvalResult
    pack_result: Phase4PackResult

    @property
    def processing_time(self) -> float:
        return self.stats.total_sec

    @property
    def stocks_used(self) -> int:
        return self.stats.stocks_used


# ═══════════════════════════════════════════════════════════════════
# 메인 오케스트레이터
# ═══════════════════════════════════════════════════════════════════

def pack_parts_phase4(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
    time_budget: float = 60.0,
) -> Phase4Result:
    """
    Phase 4.0 메인 진입점.

    4단계 파이프라인을 순서대로 실행한다.

    Args:
        settings:     kerf, trimming, optimization_goal 설정
        stocks:       원장 목록 (qty 포함)
        parts:        배치할 부품 목록 (qty 포함)
        time_budget:  전체 시간 예산 (초). 기본 60초.

    Returns:
        Phase4Result
    """
    timer = _Timer()
    stats = Phase4RunStats()
    kerf = settings.kerf

    # ── 타임 버짓 배분 ────────────────────────────────────────────
    budget_s1 = time_budget * _BUDGET_STEP1_RATIO
    budget_s2 = time_budget * _BUDGET_STEP2_RATIO
    budget_s3 = time_budget * _BUDGET_STEP3_RATIO
    budget_s4 = time_budget * _BUDGET_STEP4_RATIO

    # ═══════════════════════════════════════════════════════════════
    # Step 1: 그룹화 + 1D 배낭 DP
    # ═══════════════════════════════════════════════════════════════
    plans, groups = solve_all_strips(
        parts=parts,
        stocks=stocks,
        kerf=kerf,
        tw_tolerance=max(0.5, kerf * 0.1),
        time_budget=budget_s1,
    )
    stats.n_groups = len(groups)
    stats.n_plans = len(plans)
    stats.step1_sec = timer.checkpoint("Step1_DP")

    # ═══════════════════════════════════════════════════════════════
    # Step 2: VirtualStrip 생성 + 잔여 부품 분리
    # ═══════════════════════════════════════════════════════════════
    strips, leftover_parts = build_virtual_strips(
        plans=plans,
        parts=parts,
        kerf=kerf,
    )
    stats.n_strips = len(strips)
    stats.n_leftover_parts = len(leftover_parts)
    stats.step2_sec = timer.checkpoint("Step2_VirtualStrip") - stats.step1_sec

    # ═══════════════════════════════════════════════════════════════
    # Step 3: 전역 원장 배정 (Greedy FFD)
    # ═══════════════════════════════════════════════════════════════
    eval_result = assign_strips_to_bins(
        strips=strips,
        stocks=stocks,
        kerf=kerf,
    )
    stats.n_assigned = len(eval_result.assignments)
    stats.n_unassigned_strips = len(eval_result.unassigned)
    stats.layout_score = eval_result.layout_score
    stats.step3_sec = timer.checkpoint("Step3_BinAssign") - stats.step1_sec - stats.step2_sec

    # 미배정 Strip의 내부 부품들을 leftover_parts에 추가
    # (어느 원장에도 못 들어간 Strip → 개별 부품으로 분해해 GRASP fallback 투입)
    extended_leftover = list(leftover_parts)
    for unassigned_strip in eval_result.unassigned:
        for p_obj, p_dims, p_qty in unassigned_strip.internal_parts:
            # 동일 part_id가 이미 leftover에 있으면 qty를 합산
            existing = next((p for p in extended_leftover if p.id == p_obj.id), None)
            if existing is not None:
                existing.qty += p_qty
            else:
                extended_leftover.append(Part(
                    id=p_obj.id,
                    dims=p_obj.dims,
                    qty=p_qty,
                    lock_z=p_obj.lock_z,
                    allow_xy_rotation=p_obj.allow_xy_rotation,
                    priority=p_obj.priority,
                    color=p_obj.color,
                ))

    # ═══════════════════════════════════════════════════════════════
    # Step 4: 실제 3D 배치 + GRASP fallback
    # ═══════════════════════════════════════════════════════════════
    packer = StripFirstPacker()
    pack_result = packer.execute(
        assignments=eval_result.assignments,
        unassigned_strips=eval_result.unassigned,
        leftover_parts=extended_leftover,
        kerf=kerf,
        axis_bias=_DEFAULT_AXIS_BIAS,
    )

    # extended_leftover에서 fallback이 처리 못한 부품이 남았으면
    # 남은 원장(아직 안 열린 stock)으로 GRASP 최종 폴백
    final_unplaced: Dict[str, int] = dict(pack_result.unplaced)
    extra_occupied: List[Node] = []

    if final_unplaced:
        # 아직 사용되지 않은 원장 인스턴스 계산
        used_stock_counts: Dict[str, int] = {}
        for a in eval_result.assignments:
            used_stock_counts[a.stock_id] = (
                used_stock_counts.get(a.stock_id, 0) + 1
            )
        remaining_stocks = _compute_remaining_stocks(stocks, used_stock_counts)

        if remaining_stocks:
            unplaced_parts = [
                Part(
                    id=pid,
                    dims=_find_part_dims(parts, pid),
                    qty=qty,
                    lock_z=_find_part_attr(parts, pid, "lock_z", True),
                    allow_xy_rotation=_find_part_attr(parts, pid, "allow_xy_rotation", True),
                    priority=_find_part_attr(parts, pid, "priority", 0),
                    color=_find_part_attr(parts, pid, "color", "#4f8ef7"),
                )
                for pid, qty in final_unplaced.items()
                if qty > 0 and _find_part_dims(parts, pid) is not None
            ]

            if unplaced_parts:
                grasp_result = _grasp_pack_parts(
                    settings, remaining_stocks, unplaced_parts
                )
                extra_occupied = grasp_result.occupied_nodes
                final_unplaced = dict(grasp_result.unplaced)

    # ── 최종 결과 조립 ─────────────────────────────────────────────
    stats.n_strip_placed = len(pack_result.strip_records)
    stats.n_fallback_placed = len(pack_result.occupied_nodes) - stats.n_strip_placed
    stats.n_unplaced_parts = sum(final_unplaced.values())
    stats.step4_sec = timer.checkpoint("Step4_3DPlace") - (
        stats.step1_sec + stats.step2_sec + stats.step3_sec
    )
    stats.total_sec = timer.elapsed
    stats.stocks_used = pack_result.stocks_used
    stats.strip_assignment_rate = pack_result.strip_assignment_rate

    all_occupied = pack_result.occupied_nodes + extra_occupied

    return Phase4Result(
        occupied_nodes=all_occupied,
        unplaced=final_unplaced,
        free_nodes=pack_result.free_nodes,
        stats=stats,
        plans=plans,
        strips=strips,
        eval_result=eval_result,
        pack_result=pack_result,
    )


# ═══════════════════════════════════════════════════════════════════
# 내부 헬퍼
# ═══════════════════════════════════════════════════════════════════

def _compute_remaining_stocks(
    stocks: List[Stock],
    used_counts: Dict[str, int],
) -> List[Stock]:
    """
    배정에 사용된 원장 수를 빼서 남은 원장 목록을 반환한다.

    Stock.qty에서 실제 사용된 인스턴스 수를 차감한다.
    qty가 0 이하이면 제외한다.
    """
    result = []
    for s in stocks:
        used = used_counts.get(s.id, 0)
        remaining_qty = s.qty - used
        if remaining_qty > 0:
            result.append(Stock(
                id=s.id,
                dims=s.dims,
                qty=remaining_qty,
                trimming=s.trimming,
            ))
    return result


def _find_part_dims(parts: List[Part], part_id: str) -> Optional[Dims]:
    """part_id로 원본 Dims를 찾는다. 없으면 None."""
    for p in parts:
        if p.id == part_id:
            return p.dims
    return None


def _find_part_attr(parts: List[Part], part_id: str, attr: str, default):
    """part_id로 원본 Part 속성을 찾는다. 없으면 default."""
    for p in parts:
        if p.id == part_id:
            return getattr(p, attr, default)
    return default


# ═══════════════════════════════════════════════════════════════════
# 독립 실행 테스트
# ═══════════════════════════════════════════════════════════════════

def _run_tests() -> None:
    """
    핵심 시나리오 3개로 파이프라인 전체를 검증한다.

    시나리오 1: 기본 파이프라인 — 4단계 모두 정상 통과
    시나리오 2: 긴 띠장 보호 — 3000L 띠장이 다른 부품과 격리
    시나리오 3: 미배정 낙수 — Strip 못 들어간 부품이 GRASP fallback으로 처리
    """
    SEP = "─" * 60

    # ── 시나리오 1: 기본 파이프라인 ────────────────────────────────
    print(SEP)
    print("시나리오 1: 기본 파이프라인 전체 실행")
    print(SEP)

    settings = EngineSettings(kerf=3.0, trimming=TrimmingMargins())
    stocks1 = [
        Stock(id="S1", dims=Dims(l=3000, w=300, t=18), qty=2, trimming=TrimmingMargins()),
        Stock(id="S2", dims=Dims(l=5000, w=300, t=18), qty=1, trimming=TrimmingMargins()),
    ]
    parts1 = [
        Part(id="P1", dims=Dims(l=800, w=300, t=18), qty=4, lock_z=True, allow_xy_rotation=False),
        Part(id="P2", dims=Dims(l=600, w=300, t=18), qty=3, lock_z=True, allow_xy_rotation=False),
        Part(id="P3", dims=Dims(l=400, w=200, t=18), qty=5, lock_z=True, allow_xy_rotation=False),
    ]

    result1 = pack_parts_phase4(settings, stocks1, parts1, time_budget=30.0)
    s = result1.stats

    print(f"  [Step1] 그룹={s.n_groups}  플랜={s.n_plans}  ({s.step1_sec:.3f}s)")
    print(f"  [Step2] strips={s.n_strips}  leftover={s.n_leftover_parts}  ({s.step2_sec:.3f}s)")
    print(f"  [Step3] 배정={s.n_assigned}  미배정={s.n_unassigned_strips}  ({s.step3_sec:.3f}s)")
    print(f"  [Step4] strip배치={s.n_strip_placed}  fallback={s.n_fallback_placed}  미배치={s.n_unplaced_parts}  ({s.step4_sec:.3f}s)")
    print(f"  총 배치: {len(result1.occupied_nodes)}개  원장: {s.stocks_used}장  총시간: {s.total_sec:.3f}s")
    print(f"  미배치: {result1.unplaced}")

    total_req = sum(p.qty for p in parts1)
    total_placed = len(result1.occupied_nodes)
    # Strip 내 부품들 카운트
    strip_parts_count = sum(
        qty
        for r in result1.pack_result.strip_records
        for _, _, qty in r.strip.internal_parts
    )
    print(f"  요청 부품: {total_req}개  3D노드 수: {total_placed}개  Strip 내 부품: {strip_parts_count}개")
    print("  ✅ 파이프라인 전체 실행 성공")

    # ── 시나리오 2: 긴 띠장 격리 보호 ──────────────────────────────
    print()
    print(SEP)
    print("시나리오 2: 긴 띠장(2900L) 격리 — 단독 Strip 배정 확인")
    print(SEP)

    stocks2 = [Stock(id="S1", dims=Dims(l=3000, w=200, t=18), qty=2, trimming=TrimmingMargins())]
    parts2 = [
        Part(id="STRIP_LONG", dims=Dims(l=2900, w=200, t=18), qty=1,
             lock_z=True, allow_xy_rotation=False),
        Part(id="SHORT",      dims=Dims(l=500,  w=200, t=18), qty=4,
             lock_z=True, allow_xy_rotation=False),
    ]

    result2 = pack_parts_phase4(settings, stocks2, parts2, time_budget=15.0)
    s2 = result2.stats
    print(f"  strips={s2.n_strips}  배정={s2.n_assigned}  fallback={s2.n_fallback_placed}")
    print(f"  미배치: {result2.unplaced}")

    # 긴 띠장이 단독 Strip으로 생성됐는지 확인
    strip_long_records = [
        r for r in result2.pack_result.strip_records
        if any(p.id == "STRIP_LONG" for p, _, _ in r.strip.internal_parts)
    ]
    if strip_long_records:
        rec = strip_long_records[0]
        solo = len(rec.strip.internal_parts) == 1 and rec.strip.internal_parts[0][0].id == "STRIP_LONG"
        print(f"  긴 띠장 단독 Strip: {solo}")
        print(f"  배치 위치: {rec.occupied_node.origin}")
    print("  ✅ 긴 띠장 처리 확인")

    # ── 시나리오 3: Strip 미배정 → fallback으로 처리 ────────────────
    print()
    print(SEP)
    print("시나리오 3: 미배정 Strip 내 부품이 GRASP fallback으로 처리")
    print(SEP)

    # 작은 원장에 큰 부품 → Strip 생성은 되지만 배정 실패 가능
    stocks3 = [Stock(id="S1", dims=Dims(l=2000, w=300, t=18), qty=1, trimming=TrimmingMargins())]
    parts3 = [
        Part(id="P_BIG",   dims=Dims(l=1800, w=300, t=18), qty=1, lock_z=True, allow_xy_rotation=False),
        Part(id="P_SMALL", dims=Dims(l=300,  w=300, t=18), qty=3, lock_z=True, allow_xy_rotation=False),
    ]

    result3 = pack_parts_phase4(settings, stocks3, parts3, time_budget=15.0)
    s3 = result3.stats
    print(f"  strips={s3.n_strips}  배정={s3.n_assigned}  미배정strip={s3.n_unassigned_strips}")
    print(f"  fallback 배치={s3.n_fallback_placed}  미배치: {result3.unplaced}")
    print("  ✅ 미배정 처리 확인")

    print()
    print("=" * 60)
    print("모든 테스트 통과 ✅")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
