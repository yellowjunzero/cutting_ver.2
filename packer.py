"""
packer.py -- 탐색 & 배치 엔진 (Phase 3.1: Axis-Aware GRASP + 변별력 개선)

Phase 3.1 변경 사항:
  [A] _offcut_score_for_order: 스케일 독립 체적 + short_edge 선형 보정으로 재작성
  [B] A급 잔재 페널티 확률화: 70% 부여 / 30% 면제 -> 탐색 다양성 확보
  [C] GRASP 탐색 전략 다각화: 6가지 정렬 전략 균등 순환 (shuffle 비중 17%로 축소)
  [D] Axis-Aware 탐색 차원 추가: axis_bias 파라미터로 절단 축 선호도 제어
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
