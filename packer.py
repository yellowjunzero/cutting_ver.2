"""
packer.py — 탐색 & 배치 엔진

공개 API: pack_parts(settings, stocks, parts) → (List[Node], Dict[str, int])
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from core import (
    CutAxis,
    Dims,
    EngineSettings,
    InvalidCutError,
    Node,
    NodeState,
    Part,
    Stock,
    _get_axis,
    _new_id,
    create_root_node,
    split_node,
)

_EPSILON = 0.5  # mm 단위 오차 허용치


# ─────────────────────────────────────────────
# NodeHeap — 지연 삭제 우선순위 큐
# ─────────────────────────────────────────────

class NodeHeap:
    def __init__(self):
        self._heap: list = []
        self._removed: set = set()

    def push(self, node: Node):
        heapq.heappush(self._heap, (-node.depth, node.volume, node.node_id, node))

    def pop(self) -> Optional[Node]:
        while self._heap:
            neg_depth, neg_vol, node_id, node = heapq.heappop(self._heap)
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


# ─────────────────────────────────────────────
# 배치 후보 (부피 중심의 공평한 경쟁)
# ─────────────────────────────────────────────

@dataclass(order=True)
class PlacementCandidate:
    neg_estimated_volume: float  # ✨ 1순위: '개수'가 아닌 '채우는 부피'로 경쟁! (큰 부품도 우선권 획득)
    linear_waste: float          # 2순위: 쓰레기 최소화
    part_idx: int                # 3순위: 종류별 블록화 (섞임 방지)
    rotation_penalty: int        # 4순위: 회전 페널티
    neg_max_offcut: float        # 5순위: 잔재 크게 남기기
    node_id: str = field(compare=False)
    node: Node = field(compare=False)
    part: Part = field(compare=False)
    orientation: Dims = field(compare=False)
    cut_order: Tuple[CutAxis, ...] = field(compare=False)


# ─────────────────────────────────────────────
# Max-Offcut 절단 순서 최적화 (딱 맞는 절단 버그 수정!)
# ─────────────────────────────────────────────

_ALL_AXES = [CutAxis.X, CutAxis.Y, CutAxis.Z]
_ALL_ORDERS = list(permutations(_ALL_AXES))

def _offcut_score_for_order(
    node: Node,
    part_dims: Dims,
    cut_order: Tuple[CutAxis, ...],
    kerf: float,
) -> Optional[float]:
    remaining = { CutAxis.X: node.dims.l, CutAxis.Y: node.dims.w, CutAxis.Z: node.dims.t }
    part_size = { CutAxis.X: part_dims.l, CutAxis.Y: part_dims.w, CutAxis.Z: part_dims.t }
    max_score = -1.0

    for axis in cut_order:
        pos = part_size[axis]
        total = remaining[axis]

        if abs(total - pos) <= _EPSILON:
            remaining[axis] = pos
            continue

        remainder = total - pos - kerf
        
        # ✨ 버그 수정: remainder가 0인 경우(딱 맞음)를 실패로 처리하지 않고 허용합니다.
        if remainder < -_EPSILON:
            return None  

        # 딱 맞게 잘려서 남는 공간이 없을 때
        if remainder <= _EPSILON:
            remaining[axis] = pos
            continue

        rem_l = remaining[CutAxis.X] if axis != CutAxis.X else remainder
        rem_w = remaining[CutAxis.Y] if axis != CutAxis.Y else remainder
        rem_t = remaining[CutAxis.Z] if axis != CutAxis.Z else remainder
        
        short_edge = min(rem_l, rem_w)
        
        if short_edge < 30:
            score = 0.0
        else:
            normalized_l = rem_l / 1000.0
            normalized_w = rem_w / 1000.0
            normalized_t = rem_t / 1000.0
            normalized_short = short_edge / 1000.0
            score = (normalized_short ** 2) * normalized_t * (normalized_l * normalized_w * normalized_t)
            
        if score > max_score:
            max_score = score
            
        remaining[axis] = pos

    return max_score

def _best_cut_order(node: Node, part_dims: Dims, kerf: float) -> Optional[Tuple[Tuple[CutAxis, ...], float]]:
    best_order, best_score = None, -1.0
    for order in _ALL_ORDERS:
        score = _offcut_score_for_order(node, part_dims, order, kerf)
        if score is None: continue
        if score > best_score:
            best_score = score
            best_order = order
    return (best_order, best_score) if best_order else None


# ─────────────────────────────────────────────
# Best-Fit 후보 선택
# ─────────────────────────────────────────────

def _fit_count(total: float, pdim: float, kerf: float) -> int:
    if pdim > total + _EPSILON: return 0
    return int((total + kerf + _EPSILON) // (pdim + kerf))

def _axis_waste(total: float, pdim: float, kerf: float) -> float:
    count = _fit_count(total, pdim, kerf)
    if count <= 0: return total
    return max(0.0, total - (count * pdim) - (count - 1) * kerf)

def _find_best_candidate(
    node: Node, remaining_parts: Dict[str, int], parts_by_id: Dict[str, Part], kerf: float,
) -> Optional[PlacementCandidate]:
    best: Optional[PlacementCandidate] = None
    part_keys = list(remaining_parts.keys())

    for part_id, qty in remaining_parts.items():
        if qty <= 0: continue
        part = parts_by_id[part_id]
        p_idx = part_keys.index(part_id)

        for orientation in part.allowed_orientations():
            if not orientation.fits_in(node.dims): continue

            cx = _fit_count(node.dims.l, orientation.l, kerf)
            cy = _fit_count(node.dims.w, orientation.w, kerf)
            cz = _fit_count(node.dims.t, orientation.t, kerf)
            est_count = cx * cy * cz  
            
            # ✨ 수정: 개수가 아닌 '차지하는 부피'로 계산
            est_vol = est_count * (orientation.l * orientation.w * orientation.t)

            lw_x = _axis_waste(node.dims.l, orientation.l, kerf)
            lw_y = _axis_waste(node.dims.w, orientation.w, kerf)
            lw_z = _axis_waste(node.dims.t, orientation.t, kerf)
            total_linear_waste = lw_x + lw_y + lw_z

            rot_penalty = 0 if (orientation.l == part.dims.l and orientation.w == part.dims.w and orientation.t == part.dims.t) else (1 if orientation.t == part.dims.t else 2)

            order_result = _best_cut_order(node, orientation, kerf)
            if order_result is None: continue

            best_order, max_offcut = order_result

            candidate = PlacementCandidate(
                neg_estimated_volume=-est_vol,  # 부피 경쟁
                linear_waste=total_linear_waste,
                part_idx=p_idx,                 
                rotation_penalty=rot_penalty,   
                neg_max_offcut=-max_offcut,     
                node_id=node.node_id, node=node, part=part,
                orientation=orientation, cut_order=best_order,
            )

            if best is None or candidate < best:
                best = candidate

    return best


# ─────────────────────────────────────────────
# 순차 관통 절단
# ─────────────────────────────────────────────

def _place_part_on_node(
    node: Node, part: Part, orientation: Dims, cut_order: Tuple[CutAxis, ...], kerf: float,
) -> Tuple[Node, List[Node]]:
    part_size = { CutAxis.X: orientation.l, CutAxis.Y: orientation.w, CutAxis.Z: orientation.t }
    current = node
    new_free_nodes = []

    for axis in cut_order:
        pos = part_size[axis]
        total = _get_axis(current.dims, axis)
        if abs(total - pos) <= _EPSILON: continue
        child_a, child_b = split_node(current, axis, pos, kerf)
        new_free_nodes.append(child_b)
        current = child_a

    current.state = NodeState.OCCUPIED
    current.placed_part = part
    current.placed_part_dims = orientation
    return current, new_free_nodes


# ─────────────────────────────────────────────
# ✨ 싱글 패스 엔진 (기존의 pack_parts)
# ─────────────────────────────────────────────

def _pack_parts_single(
    settings: EngineSettings, stocks: List[Stock], parts: List[Part],
) -> PackResult:
    start = time.perf_counter()
    kerf = settings.kerf
    parts_by_id = {p.id: p for p in parts}
    
    remaining = {p.id: p.qty for p in parts}
    occupied_nodes = []
    heap = NodeHeap()
    stocks_used = 0

    stock_pool = [stock for stock in stocks for _ in range(stock.qty)]
    stock_index = 0

    def _open_next_stock() -> bool:
        nonlocal stock_index, stocks_used
        if stock_index >= len(stock_pool): return False
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
            if not _open_next_stock(): break
            node = heap.pop()
            if node is None: break

        candidate = _find_best_candidate(node, remaining, parts_by_id, kerf)
        if candidate is None:
            node.state = NodeState.DISCARDED
            continue

        occupied, new_free = _place_part_on_node(
            candidate.node, candidate.part, candidate.orientation, candidate.cut_order, kerf
        )
        occupied_nodes.append(occupied)
        remaining[candidate.part.id] -= 1

        for free_node in new_free:
            heap.push(free_node)

    free_nodes = [item[-1] for item in heap._heap if item[-1].node_id not in heap._removed and item[-1].state == NodeState.FREE]
    
    return PackResult(
        occupied_nodes=occupied_nodes,
        unplaced={k: v for k, v in remaining.items() if v > 0},
        processing_time=time.perf_counter() - start,
        stocks_used=stocks_used,
        free_nodes=free_nodes,
    )


# ─────────────────────────────────────────────
# ✨ 멀티 패스 래퍼 (최적의 평행우주를 찾아라!)
# ─────────────────────────────────────────────
import copy

@dataclass
class PackResult:
    occupied_nodes: List[Node]
    unplaced: Dict[str, int]
    processing_time: float
    stocks_used: int
    free_nodes: List[Node] = field(default_factory=list)

def pack_parts(
    settings: EngineSettings, stocks: List[Stock], parts: List[Part],
) -> PackResult:
    """
    여러 가지 정렬 방식을 테스트하고 가장 성적이 좋은(미배치가 적은) 결과를 반환합니다.
    """
    best_result = None
    best_unplaced = float('inf')
    
    # 3가지 시나리오로 가상 절단을 진행합니다.
    strategies = [
        sorted(parts, key=lambda p: -(p.l * p.w * p.t)), # 1. 부피가 큰 놈부터 (수율 최강)
        sorted(parts, key=lambda p: -max(p.l, p.w)),     # 2. 길이가 긴 놈부터 (Strip 보존)
        sorted(parts, key=lambda p: -p.priority)         # 3. 사용자 입력 순서
    ]

    start_total = time.perf_counter()

    for strategy_parts in strategies:
        # 원본 데이터 오염 방지
        test_stocks = copy.deepcopy(stocks)
        test_parts = copy.deepcopy(strategy_parts)
        
        result = _pack_parts_single(settings, test_stocks, test_parts)
        unplaced_count = sum(result.unplaced.values())
        
        # 현재 시나리오가 가장 미배치 수량이 적다면 1등 갱신
        if unplaced_count < best_unplaced:
            best_unplaced = unplaced_count
            best_result = result
        
        # 잔재 하나 없이 완벽하게 다 넣었다면 더 이상 계산할 필요 없음
        if best_unplaced == 0:
            break

    # 최종 연산 시간 보정 후 반환
    best_result.processing_time = time.perf_counter() - start_total
    return best_result


# ─────────────────────────────────────────────
# 결과 데이터 클래스 & 메인 API
# ─────────────────────────────────────────────

@dataclass
class PackResult:
    occupied_nodes: List[Node]
    unplaced: Dict[str, int]
    processing_time: float
    stocks_used: int
    free_nodes: List[Node] = field(default_factory=list)

def pack_parts(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> PackResult:
    start = time.perf_counter()

    kerf = settings.kerf
    parts_by_id: Dict[str, Part] = {p.id: p for p in parts}

    remaining: Dict[str, int] = {}
    for p in sorted(parts, key=lambda x: -x.priority):
        remaining[p.id] = p.qty

    occupied_nodes: List[Node] = []
    heap = NodeHeap()
    stocks_used = 0

    stock_pool: List[Stock] = []
    for stock in stocks:
        for _ in range(stock.qty):
            stock_pool.append(stock)

    stock_index = 0

    def _open_next_stock() -> bool:
        nonlocal stock_index, stocks_used
        if stock_index >= len(stock_pool):
            return False
        stock = stock_pool[stock_index]
        stock_index += 1
        stocks_used += 1
        root = create_root_node(stock)
        heap.push(root)
        return True

    if not _open_next_stock():
        return PackResult([], remaining, 0.0, 0)

    while True:
        if not any(v > 0 for v in remaining.values()):
            break

        node = heap.pop()

        if node is None:
            if not _open_next_stock():
                break
            node = heap.pop()
            if node is None:
                break

        candidate = _find_best_candidate(node, remaining, parts_by_id, kerf)

        if candidate is None:
            node.state = NodeState.DISCARDED
            continue

        occupied, new_free = _place_part_on_node(
            candidate.node,
            candidate.part,
            candidate.orientation,
            candidate.cut_order,
            kerf,
        )
        occupied_nodes.append(occupied)
        remaining[candidate.part.id] -= 1

        for free_node in new_free:
            heap.push(free_node)

    free_nodes = []
    for item in heap._heap:
        node = item[-1]
        if node.node_id not in heap._removed and node.state == NodeState.FREE:
            free_nodes.append(node)

    elapsed = time.perf_counter() - start
    return PackResult(
        occupied_nodes=occupied_nodes,
        unplaced={k: v for k, v in remaining.items() if v > 0},
        processing_time=elapsed,
        stocks_used=stocks_used,
        free_nodes=free_nodes,
    )
