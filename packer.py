"""
packer.py -- 탐색 & 배치 엔진 (Phase 3.1 + Phase 4.0 확장)

Phase 3.1 변경 사항:
  [A] _offcut_score_for_order: 스케일 독립 체적 + short_edge 선형 보정으로 재작성
  [B] A급 잔재 페널티 확률화: 70% 부여 / 30% 면제 -> 탐색 다양성 확보
  [C] GRASP 탐색 전략 다각화: 6가지 정렬 전략 균등 순환 (shuffle 비중 17%로 축소)
  [D] Axis-Aware 탐색 차원 추가: axis_bias 파라미터로 절단 축 선호도 제어

Phase 4.0 추가:
  StripAdapter      -- VirtualStrip → 임시 Part 변환 유틸리티
  StripFirstPacker  -- BinAssignment 계획표를 실제 3D 배치로 실행
  _pack_with_free_nodes -- free_nodes에 leftover_parts를 채우는 GRASP fallback
"""
from __future__ import annotations

import heapq
import time
import random
import copy
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from core import (
    CutAxis, Dims, EngineSettings, InvalidCutError, Node, NodeState, Part, Stock,
    _get_axis, _new_id, create_root_node, split_node,
)

_EPSILON = 0.5

_PRIME_OFFCUT_SHORT_EDGE = 300.0
_PRIME_OFFCUT_PENALTY    = 1e15
_PRIME_PENALTY_PROB      = 0.70   # [B] 70% 부여, 30% 면제

_DEFAULT_AXIS_BIAS: Tuple[float, float, float] = (1.0, 1.0, 1.0)

_AXIS_BIAS_PRESETS: List[Tuple[float, float, float]] = [
    (1.0, 1.0, 1.0),
    (2.0, 1.0, 0.5),
    (1.0, 2.0, 0.5),
    (0.5, 0.5, 2.0),
    (1.5, 1.5, 0.3),
]

_ALL_AXES  = [CutAxis.X, CutAxis.Y, CutAxis.Z]
_ALL_ORDERS = list(permutations(_ALL_AXES))
_N_STRATEGIES = 6


# ── NodeHeap ────────────────────────────────────────────────────────────────

class NodeHeap:
    def __init__(self):
        self._heap: list = []
        self._removed: set = set()

    def push(self, node: Node):
        heapq.heappush(self._heap, (-node.depth, -node.volume, node.node_id, node))

    def pop(self) -> Optional[Node]:
        while self._heap:
            _nd, _nv, node_id, node = heapq.heappop(self._heap)
            if node_id in self._removed:
                continue
            if node.state != NodeState.FREE:
                self._removed.add(node_id)
                continue
            return node
        return None

    def invalidate(self, node_id: str):
        self._removed.add(node_id)

    def __len__(self) -> int:
        return len(self._heap)


# ── PlacementCandidate ──────────────────────────────────────────────────────

@dataclass(order=True)
class PlacementCandidate:
    linear_waste: float
    neg_max_offcut: float
    neg_estimated_volume: float
    rotation_penalty: int
    part_idx: int
    node_id: str  = field(compare=False)
    node: Node    = field(compare=False)
    part: Part    = field(compare=False)
    orientation: Dims               = field(compare=False)
    cut_order: Tuple[CutAxis, ...]  = field(compare=False)


# ── 헬퍼 ────────────────────────────────────────────────────────────────────

def _get_lwt(dims_obj) -> Tuple[float, float, float]:
    l = dims_obj.l if hasattr(dims_obj, 'l') else dims_obj[0]
    w = dims_obj.w if hasattr(dims_obj, 'w') else dims_obj[1]
    t = dims_obj.t if hasattr(dims_obj, 't') else dims_obj[2]
    return l, w, t


# ── [A] _offcut_score_for_order (스케일 독립 체적 + short_edge 선형 보정) ──

def _offcut_score_for_order(
    node,
    part_dims,
    cut_order: Tuple[CutAxis, ...],
    kerf: float,
    axis_bias: Tuple[float, float, float] = _DEFAULT_AXIS_BIAS,
) -> Optional[float]:
    _REF_EDGE = 300.0
    _MIN_EDGE =  30.0

    n_l, n_w, n_t = _get_lwt(node.dims)
    p_l, p_w, p_t = _get_lwt(part_dims)

    remaining = {CutAxis.X: n_l, CutAxis.Y: n_w, CutAxis.Z: n_t}
    part_size = {CutAxis.X: p_l, CutAxis.Y: p_w, CutAxis.Z: p_t}
    bias_map  = {CutAxis.X: axis_bias[0], CutAxis.Y: axis_bias[1], CutAxis.Z: axis_bias[2]}

    # [A] 핵심 변경: max -> sum
    # max 방식: 각 step 중 최고 점수 1개만 반영 → X절단이 항상 독점해 동점 발생
    # sum 방식: 모든 step의 잔재 점수를 합산 → 절단 순서 전체를 평가해 완전한 변별력
    total_score = 0.0
    valid = False  # 최소 1개 step에서 유효 잔재가 발생해야 함

    for axis in cut_order:
        pos   = part_size[axis]
        total = remaining[axis]

        if abs(total - pos) <= _EPSILON:
            remaining[axis] = pos
            continue

        remainder = total - pos - kerf
        if remainder < -_EPSILON:
            return None

        if remainder <= _EPSILON:
            remaining[axis] = pos
            continue

        rem_l = remainder  if axis == CutAxis.X else remaining[CutAxis.X]
        rem_w = remainder  if axis == CutAxis.Y else remaining[CutAxis.Y]
        rem_t = remainder  if axis == CutAxis.Z else remaining[CutAxis.Z]

        short_edge = min(rem_l, rem_w)

        if short_edge >= _MIN_EDGE:
            remainder_vol = rem_l * rem_w * rem_t
            edge_bonus    = max(short_edge / _REF_EDGE, 1.0)
            step_score    = remainder_vol * edge_bonus * bias_map[axis]
            total_score  += step_score
            valid = True

        remaining[axis] = pos

    # 유효 잔재가 하나도 없으면 0 반환 (None 아님: 절단 자체는 가능)
    return total_score if valid else 0.0


def _best_cut_order(
    node,
    part_dims,
    kerf: float,
    axis_bias: Tuple[float, float, float] = _DEFAULT_AXIS_BIAS,
) -> Optional[Tuple[Tuple[CutAxis, ...], float]]:
    best_order, best_score = None, -1.0
    for order in _ALL_ORDERS:
        score = _offcut_score_for_order(node, part_dims, order, kerf, axis_bias)
        if score is None:
            continue
        if score > best_score:
            best_score = score
            best_order = order
    return (best_order, best_score) if best_order else None


def _fit_count(total: float, pdim: float, kerf: float) -> int:
    if pdim > total + _EPSILON:
        return 0
    return int((total + kerf + _EPSILON) // (pdim + kerf))


def _axis_waste(total: float, pdim: float, kerf: float) -> float:
    count = _fit_count(total, pdim, kerf)
    if count <= 0:
        return total
    return max(0.0, total - (count * pdim) - (count - 1) * kerf)


# ── _find_best_candidate ─────────────────────────────────────────────────────

def _find_best_candidate(
    node: Node,
    remaining_parts: Dict[str, int],
    parts_by_id: Dict[str, Part],
    kerf: float,
    axis_bias: Tuple[float, float, float] = _DEFAULT_AXIS_BIAS,
) -> Optional[PlacementCandidate]:
    best: Optional[PlacementCandidate] = None
    part_keys = list(remaining_parts.keys())

    for part_id, qty in remaining_parts.items():
        if qty <= 0:
            continue
        part  = parts_by_id[part_id]
        p_idx = part_keys.index(part_id)

        for orientation in part.allowed_orientations():
            n_l, n_w, n_t = _get_lwt(node.dims)
            p_l, p_w, p_t = _get_lwt(orientation)

            if hasattr(orientation, 'fits_in'):
                if not orientation.fits_in(node.dims):
                    continue
            else:
                if p_l > n_l + _EPSILON or p_w > n_w + _EPSILON or p_t > n_t + _EPSILON:
                    continue

            cx      = _fit_count(n_l, p_l, kerf)
            cy      = _fit_count(n_w, p_w, kerf)
            cz      = _fit_count(n_t, p_t, kerf)
            est_vol = cx * cy * cz * (p_l * p_w * p_t)

            lw_x = _axis_waste(n_l, p_l, kerf)
            lw_y = _axis_waste(n_w, p_w, kerf)
            lw_z = _axis_waste(n_t, p_t, kerf)
            total_linear_waste = lw_x + lw_y + lw_z

            part_l, part_w, part_t = _get_lwt(part.dims)
            rot_penalty = (
                0 if (p_l == part_l and p_w == part_w and p_t == part_t) else
                (1 if p_t == part_t else 2)
            )

            order_result = _best_cut_order(node, orientation, kerf, axis_bias)
            if order_result is None:
                continue

            best_order, max_offcut = order_result

            # [B] A급 잔재 파괴 페널티 -- 확률적 적용
            prime_penalty   = 0.0
            short_edge_node = min(n_l, n_w)

            if short_edge_node >= _PRIME_OFFCUT_SHORT_EDGE:
                prev_part_id: Optional[str] = None
                ancestor = node.parent
                while ancestor is not None:
                    if ancestor.placed_part is not None:
                        prev_part_id = ancestor.placed_part.id
                        break
                    if ancestor.child_a is not None and ancestor.child_a.placed_part is not None:
                        prev_part_id = ancestor.child_a.placed_part.id
                        break
                    ancestor = ancestor.parent

                if prev_part_id is not None and prev_part_id != part.id:
                    if random.random() < _PRIME_PENALTY_PROB:
                        prime_penalty = _PRIME_OFFCUT_PENALTY

            adjusted_waste = total_linear_waste + prime_penalty

            candidate = PlacementCandidate(
                linear_waste=adjusted_waste,
                neg_max_offcut=-max_offcut,
                neg_estimated_volume=-est_vol,
                rotation_penalty=rot_penalty,
                part_idx=p_idx,
                node_id=node.node_id,
                node=node,
                part=part,
                orientation=orientation,
                cut_order=best_order,
            )

            if best is None or candidate < best:
                best = candidate

    return best


# ── _place_part_on_node ──────────────────────────────────────────────────────

def _place_part_on_node(
    node: Node,
    part: Part,
    orientation: Dims,
    cut_order: Tuple[CutAxis, ...],
    kerf: float,
) -> Tuple[Node, List[Node]]:
    p_l, p_w, p_t = _get_lwt(orientation)
    part_size = {CutAxis.X: p_l, CutAxis.Y: p_w, CutAxis.Z: p_t}
    current   = node
    new_free_nodes: List[Node] = []

    for axis in cut_order:
        pos   = part_size[axis]
        total = _get_axis(current.dims, axis)
        if abs(total - pos) <= _EPSILON:
            continue
        child_a, child_b = split_node(current, axis, pos, kerf)
        new_free_nodes.append(child_b)
        current = child_a

    current.state            = NodeState.OCCUPIED
    current.placed_part      = part
    current.placed_part_dims = orientation
    return current, new_free_nodes


# ── PackResult ───────────────────────────────────────────────────────────────

@dataclass
class PackResult:
    occupied_nodes: List[Node]
    unplaced: Dict[str, int]
    processing_time: float
    stocks_used: int
    free_nodes: List[Node] = field(default_factory=list)


# ── _pack_parts_single ───────────────────────────────────────────────────────

def _pack_parts_single(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
    axis_bias: Tuple[float, float, float] = _DEFAULT_AXIS_BIAS,
) -> PackResult:
    start       = time.perf_counter()
    kerf        = settings.kerf
    parts_by_id = {p.id: p for p in parts}
    remaining   = {p.id: p.qty for p in parts}
    occupied_nodes: List[Node] = []
    heap        = NodeHeap()
    stocks_used = 0

    stock_pool  = [stock for stock in stocks for _ in range(stock.qty)]
    stock_index = 0

    def _open_next_stock() -> bool:
        nonlocal stock_index, stocks_used
        if stock_index >= len(stock_pool):
            return False

        remaining_dims: List[Dims] = []
        for pid, qty in remaining.items():
            if qty > 0:
                for orient in parts_by_id[pid].allowed_orientations():
                    remaining_dims.append(orient)

        best_local_idx: Optional[int] = None

        if remaining_dims:
            sorted_dims = sorted(
                remaining_dims,
                key=lambda d: _get_lwt(d)[0] * _get_lwt(d)[1] * _get_lwt(d)[2],
                reverse=True,
            )

            for candidate_dims in sorted_dims:
                cd_l, cd_w, cd_t = _get_lwt(candidate_dims)
                best_vol_for_dim: float = float('inf')
                best_idx_for_dim: Optional[int] = None

                for i in range(stock_index, len(stock_pool)):
                    s = stock_pool[i]
                    ud_l, ud_w, ud_t = _get_lwt(s.dims)
                    if (cd_l <= ud_l + _EPSILON and
                            cd_w <= ud_w + _EPSILON and
                            cd_t <= ud_t + _EPSILON):
                        uv = ud_l * ud_w * ud_t
                        if uv < best_vol_for_dim:
                            best_vol_for_dim = uv
                            best_idx_for_dim = i

                if best_idx_for_dim is not None:
                    best_local_idx = best_idx_for_dim
                    break

        if best_local_idx is not None and best_local_idx != stock_index:
            stock_pool[stock_index], stock_pool[best_local_idx] = (
                stock_pool[best_local_idx], stock_pool[stock_index]
            )

        stock = stock_pool[stock_index]
        stock_index += 1
        stocks_used += 1
        heap.push(create_root_node(stock))
        return True

    if not _open_next_stock():
        return PackResult([], remaining, 0.0, 0)

    while any(v > 0 for v in remaining.values()):
        node = heap.pop()
        if node is None:
            if not _open_next_stock():
                break
            node = heap.pop()
            if node is None:
                break

        candidate = _find_best_candidate(node, remaining, parts_by_id, kerf, axis_bias)
        if candidate is None:
            node.state = NodeState.DISCARDED
            continue

        occupied, new_free = _place_part_on_node(
            candidate.node, candidate.part,
            candidate.orientation, candidate.cut_order, kerf,
        )
        occupied_nodes.append(occupied)
        remaining[candidate.part.id] -= 1

        for free_node in new_free:
            heap.push(free_node)

    free_nodes = [
        item[3] for item in heap._heap
        if item[3].node_id not in heap._removed
        and item[3].state == NodeState.FREE
    ]

    return PackResult(
        occupied_nodes=occupied_nodes,
        unplaced={k: v for k, v in remaining.items() if v > 0},
        processing_time=time.perf_counter() - start,
        stocks_used=stocks_used,
        free_nodes=free_nodes,
    )


# ── [C] GRASP 탐색 전략 ──────────────────────────────────────────────────────

def _apply_sort_strategy(parts: List[Part], strategy_idx: int) -> None:
    """
    [C] 6가지 정렬 전략 (0~5 순환)
      0: 부피(l*w*t) 내림차순
      1: 긴 변(max(l,w)) 내림차순
      2: 짧은 변(min(l,w)) 내림차순
      3: 두께(t) 내림차순  [신규] -- Z축 탑 쌓기 강화
      4: 단면적(l*w) 내림차순 [신규] -- 바닥 면적 기준
      5: random shuffle -- shuffle 비중 17%로 축소
    """
    s = strategy_idx % _N_STRATEGIES
    if s == 0:
        parts.sort(key=lambda p: -(
            _get_lwt(p.dims)[0] * _get_lwt(p.dims)[1] * _get_lwt(p.dims)[2]
        ))
    elif s == 1:
        parts.sort(key=lambda p: -max(_get_lwt(p.dims)[0], _get_lwt(p.dims)[1]))
    elif s == 2:
        parts.sort(key=lambda p: -min(_get_lwt(p.dims)[0], _get_lwt(p.dims)[1]))
    elif s == 3:
        parts.sort(key=lambda p: -_get_lwt(p.dims)[2])
    elif s == 4:
        parts.sort(key=lambda p: -(
            _get_lwt(p.dims)[0] * _get_lwt(p.dims)[1]
        ))
    else:
        random.shuffle(parts)


# ── pack_parts (Phase 3.1 GRASP 메인 루프) ──────────────────────────────────

def pack_parts(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> PackResult:
    """
    Phase 3.1 5초 GRASP 최적화 엔진

    탐색 차원:
      [C] 부품 순서 x 6가지 전략 (부피/긴변/짧은변/두께/단면적/랜덤)
      [D] axis_bias x 5 preset + 랜덤 --> 절단 축 선호도 변경
      [B] A급 잔재 페널티 확률화 --> 패스마다 다른 공간 탐색

    평가 기준:
      1순위: 미배치 부품 수 최소
      2순위: 단일 최대 잔재 부피 최대
    """
    start_total = time.perf_counter()
    TIME_LIMIT  = 5.0

    best_result         = None
    best_unplaced       = float('inf')
    best_largest_offcut = -1.0

    best_result         = _pack_parts_single(settings, stocks, parts, _DEFAULT_AXIS_BIAS)
    best_unplaced       = sum(best_result.unplaced.values())
    best_largest_offcut = max((n.volume for n in best_result.free_nodes), default=0.0)

    strategy_idx = 0
    bias_idx     = 0

    while True:
        if time.perf_counter() - start_total > TIME_LIMIT:
            break

        # [C] 부품 순서 전략 순환
        test_parts = copy.deepcopy(parts)
        _apply_sort_strategy(test_parts, strategy_idx)
        strategy_idx += 1

        # [D] axis_bias 순환 (preset -> 랜덤 -> preset ...)
        n_presets = len(_AXIS_BIAS_PRESETS)
        if bias_idx < n_presets:
            current_bias = _AXIS_BIAS_PRESETS[bias_idx]
        else:
            current_bias = (
                random.uniform(0.3, 2.5),
                random.uniform(0.3, 2.5),
                random.uniform(0.3, 2.5),
            )
        bias_idx = (bias_idx + 1) % (n_presets + 1)

        test_stocks = copy.deepcopy(stocks)

        result         = _pack_parts_single(settings, test_stocks, test_parts, current_bias)
        unplaced       = sum(result.unplaced.values())
        largest_offcut = max((n.volume for n in result.free_nodes), default=0.0)

        if unplaced < best_unplaced or (
            unplaced == best_unplaced and largest_offcut > best_largest_offcut
        ):
            best_unplaced       = unplaced
            best_largest_offcut = largest_offcut
            best_result         = result

    best_result.processing_time = time.perf_counter() - start_total
    return best_result


# ═══════════════════════════════════════════════════════════════════
# Phase 4.0 — StripAdapter
# ═══════════════════════════════════════════════════════════════════

class StripAdapter:
    """
    VirtualStrip을 core.py의 Part처럼 다룰 수 있게 하는 변환 유틸리티.

    _place_part_on_node는 Part를 받으므로, VirtualStrip의 외형 치수(dims)를
    가진 임시 Part로 변환한다.  core.py는 한 줄도 수정하지 않는다.

    생성된 임시 Part의 id 형식: "__strip__{strip_id}"
    StripFirstPacker는 이 prefix로 배치 결과가 Strip인지 일반 부품인지 구분한다.
    """

    STRIP_ID_PREFIX = "__strip__"

    @staticmethod
    def strip_as_part(strip: "VirtualStrip") -> Part:  # type: ignore[name-defined]
        """
        VirtualStrip → 임시 Part 변환.

        - dims: strip.dims 그대로 (외형 치수)
        - qty: 1 (Strip은 항상 단일 유닛)
        - lock_z: True, allow_xy_rotation: False
          (VirtualStrip 방향은 이미 DP/Group 단계에서 결정됨)
        - color: strip.source_plan의 첫 번째 내부 부품 색상 상속
                 (3D 뷰어에서 Strip 내 주요 부품 색상으로 표시)

        Args:
            strip: 변환할 VirtualStrip

        Returns:
            임시 Part (id 앞에 STRIP_ID_PREFIX 붙음)
        """
        # 내부 부품 중 첫 번째 색상 상속 (없으면 기본색)
        color = "#7c3aed"  # 기본: 보라색 (Strip 전용)
        if strip.internal_parts:
            first_part = strip.internal_parts[0][0]
            if hasattr(first_part, "color") and first_part.color:
                color = first_part.color

        return Part(
            id=f"{StripAdapter.STRIP_ID_PREFIX}{strip.strip_id}",
            dims=strip.dims,
            qty=1,
            lock_z=True,
            allow_xy_rotation=False,
            priority=10,   # Strip은 일반 부품보다 높은 우선순위
            color=color,
        )

    @staticmethod
    def is_strip_part(part: Part) -> bool:
        """part가 Strip 어댑터에서 생성된 임시 Part인지 확인"""
        return part.id.startswith(StripAdapter.STRIP_ID_PREFIX)

    @staticmethod
    def extract_strip_id(part: Part) -> str:
        """임시 Part에서 원본 strip_id 추출"""
        return part.id[len(StripAdapter.STRIP_ID_PREFIX):]


# ═══════════════════════════════════════════════════════════════════
# Phase 4.0 — _pack_with_free_nodes (GRASP fallback)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FallbackResult:
    """GRASP fallback 실행 결과"""
    occupied_nodes: List[Node]
    unplaced: Dict[str, int]
    processing_time: float


def _pack_with_free_nodes(
    free_nodes: List[Node],
    leftover_parts: List[Part],
    kerf: float,
    axis_bias: Tuple[float, float, float] = _DEFAULT_AXIS_BIAS,
) -> FallbackResult:
    """
    Strip 배치 후 남은 free_nodes에 leftover_parts를 GRASP 방식으로 채운다.

    pack_parts()나 _pack_parts_single()과 달리,
    새 Stock을 열지 않고 이미 존재하는 free_nodes만 사용한다.
    이것이 기존 GRASP와의 유일한 차이점이다.

    Args:
        free_nodes:     Strip 배치 후 남은 FREE 상태 Node 목록
        leftover_parts: VirtualStripFactory가 반환한 미처리 잔여 부품
        kerf:           톱날 두께
        axis_bias:      절단 축 선호도 가중치

    Returns:
        FallbackResult(occupied_nodes, unplaced, processing_time)
    """
    start = time.perf_counter()

    if not leftover_parts or not free_nodes:
        return FallbackResult([], {}, 0.0)

    parts_by_id: Dict[str, Part] = {p.id: p for p in leftover_parts}
    remaining: Dict[str, int] = {p.id: p.qty for p in leftover_parts}
    occupied: List[Node] = []

    # free_nodes를 NodeHeap에 직접 push (새 원장 열기 없음)
    heap = NodeHeap()
    for fn in free_nodes:
        if fn.state == NodeState.FREE:
            heap.push(fn)

    while any(v > 0 for v in remaining.values()):
        node = heap.pop()
        if node is None:
            break   # 더 이상 사용 가능한 공간 없음

        candidate = _find_best_candidate(node, remaining, parts_by_id, kerf, axis_bias)
        if candidate is None:
            node.state = NodeState.DISCARDED
            continue

        occ, new_free = _place_part_on_node(
            candidate.node, candidate.part,
            candidate.orientation, candidate.cut_order, kerf,
        )
        occupied.append(occ)
        remaining[candidate.part.id] -= 1

        for fn in new_free:
            heap.push(fn)

    return FallbackResult(
        occupied_nodes=occupied,
        unplaced={k: v for k, v in remaining.items() if v > 0},
        processing_time=time.perf_counter() - start,
    )


# ═══════════════════════════════════════════════════════════════════
# Phase 4.0 — StripFirstPacker
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StripPlacementRecord:
    """
    Strip 배치 완료 후 기록.
    occupied_node: Strip이 OCCUPIED로 마킹된 Node
    strip:         원본 VirtualStrip
    free_nodes:    이 Strip 배치로 생긴 FREE child_b 노드들
    """
    occupied_node: Node
    strip: "VirtualStrip"  # type: ignore[name-defined]
    free_nodes: List[Node]


@dataclass
class Phase4PackResult:
    """
    phase4.py pack_parts_phase4()의 최종 반환값.
    """
    occupied_nodes: List[Node]       # Strip 배치 + fallback 배치 통합
    strip_records: List[StripPlacementRecord]
    unplaced: Dict[str, int]         # 끝내 배치 못한 부품 {id: qty}
    processing_time: float
    stocks_used: int
    free_nodes: List[Node]           # 최종 배치 후 남은 FREE 노드
    strip_assignment_rate: float     # Strip 배정 성공률


class StripFirstPacker:
    """
    BinAssignment 계획표를 실제 3D Guillotine 배치로 실행하는 패커.

    실행 순서:
      1. BinAssignment를 slot(원장 인스턴스) 기준으로 그룹화
      2. 각 슬롯에 대해 독립적인 root_node 생성
      3. 슬롯에 배정된 Strips를 순서대로 _place_part_on_node로 배치
         (Strip은 StripAdapter로 임시 Part 변환 후 사용)
      4. 모든 배치 후 남은 free_nodes를 수집
      5. _pack_with_free_nodes()로 leftover_parts GRASP fallback 실행
    """

    # ──────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────

    def execute(
        self,
        assignments: List["BinAssignment"],       # type: ignore[name-defined]
        unassigned_strips: List["VirtualStrip"],  # type: ignore[name-defined]
        leftover_parts: List[Part],
        kerf: float,
        axis_bias: Tuple[float, float, float] = _DEFAULT_AXIS_BIAS,
    ) -> Phase4PackResult:
        """
        BinAssignment 계획표를 실행하여 Phase4PackResult를 반환한다.

        Args:
            assignments:       GlobalBinEvaluator가 생성한 배정 목록
            unassigned_strips: 어느 원장에도 배정되지 못한 VirtualStrip 목록
                               (현재 버전에서는 미사용; 향후 재시도 로직 확장용)
            leftover_parts:    VirtualStripFactory의 leftover + unassigned_strips
                               내부 부품들 (GRASP fallback 입력)
            kerf:              톱날 두께
            axis_bias:         절단 축 선호도 (fallback GRASP에 전달)

        Returns:
            Phase4PackResult
        """
        start = time.perf_counter()

        # ── 1. 슬롯별 배치 실행 ────────────────────────────────────
        strip_records: List[StripPlacementRecord] = []
        all_free_nodes: List[Node] = []
        stocks_used_ids: set = set()

        # slot_id 기준으로 그룹화 (동일 슬롯에 복수 Strip 배정 처리)
        slot_groups: Dict[str, List["BinAssignment"]] = {}
        for a in assignments:
            slot_groups.setdefault(a.slot_id, []).append(a)

        for slot_id, slot_assignments in slot_groups.items():
            records, free_nodes = self._place_slot_strips(
                slot_assignments, kerf
            )
            strip_records.extend(records)
            all_free_nodes.extend(free_nodes)
            if slot_assignments:
                stocks_used_ids.add(slot_assignments[0].slot.stock.id)

        # ── 2. GRASP fallback: leftover_parts를 free_nodes에 채움 ──
        fallback_result = _pack_with_free_nodes(
            all_free_nodes, leftover_parts, kerf, axis_bias
        )

        # ── 3. 최종 결과 조립 ─────────────────────────────────────
        all_occupied = (
            [r.occupied_node for r in strip_records]
            + fallback_result.occupied_nodes
        )

        # 최종 FREE 노드 수집 (fallback 후 남은 것)
        final_free = self._collect_final_free(all_free_nodes, fallback_result)

        total_strips = len(assignments) + len(unassigned_strips)
        rate = len(assignments) / total_strips if total_strips > 0 else 1.0

        return Phase4PackResult(
            occupied_nodes=all_occupied,
            strip_records=strip_records,
            unplaced=fallback_result.unplaced,
            processing_time=time.perf_counter() - start,
            stocks_used=len(stocks_used_ids),
            free_nodes=final_free,
            strip_assignment_rate=rate,
        )

    # ──────────────────────────────────────────────────────────────
    # 슬롯 단위 배치
    # ──────────────────────────────────────────────────────────────

    def _place_slot_strips(
        self,
        slot_assignments: List["BinAssignment"],
        kerf: float,
    ) -> Tuple[List[StripPlacementRecord], List[Node]]:
        """
        단일 슬롯(원장 인스턴스)에 배정된 모든 Strip을 순서대로 배치한다.

        슬롯별로 독립적인 root_node를 생성하여 Strip을 순차 배치한다.
        첫 Strip은 root_node 전체 공간에 배치하고,
        이후 Strip은 이전 배치가 남긴 child_b (X축 잔재 노드)에 배치한다.

        Returns:
            (strip_records, all_free_nodes)
            all_free_nodes: 이 슬롯의 모든 배치에서 발생한 FREE 노드들
        """
        if not slot_assignments:
            return [], []

        stock = slot_assignments[0].slot.stock
        root = create_root_node(stock)

        records: List[StripPlacementRecord] = []
        slot_free_nodes: List[Node] = []

        # 현재 배치할 노드 (처음엔 root, 이후엔 X축 잔재)
        current_node: Optional[Node] = root

        for assignment in slot_assignments:
            strip = assignment.strip
            if current_node is None:
                # 더 이상 배치할 공간이 없음 (이론상 BinSlot.can_fit이 보장하지만 방어)
                break

            record, next_node, free_nodes = self._place_single_strip(
                strip, current_node, kerf
            )
            if record is None:
                # 배치 실패 (공간 부족) → 이 슬롯의 나머지 Strip도 건너뜀
                break

            records.append(record)
            slot_free_nodes.extend(free_nodes)
            # 다음 Strip은 X축 잔재 노드(child_b)에 배치
            current_node = next_node

        # 마지막 current_node가 FREE이면 잔재로 등록
        if current_node is not None and current_node.state == NodeState.FREE:
            slot_free_nodes.append(current_node)

        return records, slot_free_nodes

    def _place_single_strip(
        self,
        strip: "VirtualStrip",
        node: Node,
        kerf: float,
    ) -> Tuple[Optional[StripPlacementRecord], Optional[Node], List[Node]]:
        """
        단일 Strip을 node에 배치한다.

        배치 전략:
          - StripAdapter로 Strip을 임시 Part로 변환
          - _best_cut_order로 최적 절단 순서 결정
          - _place_part_on_node로 물리 배치 실행

        Returns:
            (record, x_axis_child_b, other_free_nodes)
            record: 배치 성공 시 StripPlacementRecord, 실패 시 None
            x_axis_child_b: X축 첫 절단 후 남은 잔재 노드 (다음 Strip 배치용)
                            첫 절단이 X축이 아니거나 없으면 None
            other_free_nodes: X축 잔재 외 나머지 FREE 노드들 (W, T 방향 잔재)
        """
        strip_part = StripAdapter.strip_as_part(strip)
        strip_dims = strip.dims

        # 크기 적합성 최종 확인 (방어적)
        if not strip_dims.fits_in(node.dims):
            return None, None, []

        # 최적 절단 순서 결정
        order_result = _best_cut_order(node, strip_dims, kerf, _DEFAULT_AXIS_BIAS)
        if order_result is None:
            return None, None, []
        cut_order, _ = order_result

        # 물리 배치 실행
        try:
            occupied, free_nodes = _place_part_on_node(
                node, strip_part, strip_dims, cut_order, kerf
            )
        except Exception:
            return None, None, []

        # occupied_node에 원본 Strip 참조 저장 (역추적용)
        # Part 객체는 이미 strip_part이므로 placed_part.id로 구분 가능
        occupied.placed_part = strip_part  # 이미 설정되지만 명시적 재확인

        record = StripPlacementRecord(
            occupied_node=occupied,
            strip=strip,
            free_nodes=list(free_nodes),
        )

        # X축 잔재(next_node)와 나머지 잔재(other_free) 분리
        # 절단 순서의 첫 번째 축이 X이고 해당 free_node가 존재하면 next_node로 사용
        # → 동일 슬롯의 다음 Strip이 이어서 배치되는 공간
        next_node: Optional[Node] = None
        other_free: List[Node] = []

        for fn in free_nodes:
            # X축 첫 절단의 child_b를 찾음:
            # cut.axis == X이고 origin.x가 node.origin.x보다 크면 X-방향 잔재
            if (fn.cut is not None
                    and fn.cut.axis == CutAxis.X
                    and fn.origin.x > node.origin.x + _EPSILON
                    and next_node is None):
                next_node = fn
            else:
                other_free.append(fn)

        return record, next_node, other_free

    # ──────────────────────────────────────────────────────────────
    # 최종 FREE 노드 수집
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _collect_final_free(
        all_free_nodes: List[Node],
        fallback: FallbackResult,
    ) -> List[Node]:
        """
        Fallback 배치 후 남은 최종 FREE 노드만 반환한다.
        (OCCUPIED/SPLIT/DISCARDED 상태로 바뀐 노드 제외)
        """
        return [n for n in all_free_nodes if n.state == NodeState.FREE]
