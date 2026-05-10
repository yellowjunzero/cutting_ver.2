"""
packer.py — 탐색 & 배치 엔진 (5초 극한 최적화 GRASP)
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


@dataclass(order=True)
class PlacementCandidate:
    neg_estimated_volume: float  # 1순위: '채우는 부피' 중심 (덩치 큰 부품도 공평하게 점수 획득)
    linear_waste: float          # 2순위: 톱날 로스 방어
    part_idx: int                # 3순위: 부품 섞임 방지
    rotation_penalty: int        
    neg_max_offcut: float        
    node_id: str = field(compare=False)
    node: Node = field(compare=False)
    part: Part = field(compare=False)
    orientation: Dims = field(compare=False)
    cut_order: Tuple[CutAxis, ...] = field(compare=False)

_ALL_AXES = [CutAxis.X, CutAxis.Y, CutAxis.Z]
_ALL_ORDERS = list(permutations(_ALL_AXES))

def _offcut_score_for_order(
    node: Node, part_dims: Dims, cut_order: Tuple[CutAxis, ...], kerf: float, randomize: bool
) -> Optional[float]:
    remaining = {CutAxis.X: node.dims.l, CutAxis.Y: node.dims.w, CutAxis.Z: node.dims.t}
    part_size = {CutAxis.X: part_dims.l, CutAxis.Y: part_dims.w, CutAxis.Z: part_dims.t}
    max_score = -1.0

    for axis in cut_order:
        pos = part_size[axis]
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
            
            # ✨ 핵심 업그레이드: 점수 흔들기 (동일한 데이터라도 시도할 때마다 다른 각도로 잘라보게 만듦)
            if randomize:
                score *= random.uniform(0.6, 1.4)
            
        if score > max_score:
            max_score = score
            
        remaining[axis] = pos

    return max_score

def _best_cut_order(node: Node, part_dims: Dims, kerf: float, randomize: bool) -> Optional[Tuple[Tuple[CutAxis, ...], float]]:
    best_order, best_score = None, -1.0
    for order in _ALL_ORDERS:
        score = _offcut_score_for_order(node, part_dims, order, kerf, randomize)
        if score is None: continue
        if score > best_score:
            best_score = score
            best_order = order
    return (best_order, best_score) if best_order else None

def _fit_count(total: float, pdim: float, kerf: float) -> int:
    if pdim > total + _EPSILON: return 0
    return int((total + kerf + _EPSILON) // (pdim + kerf))

def _axis_waste(total: float, pdim: float, kerf: float) -> float:
    count = _fit_count(total, pdim, kerf)
    if count <= 0: return total
    return max(0.0, total - (count * pdim) - (count - 1) * kerf)

def _find_best_candidate(
    node: Node, remaining_parts: Dict[str, int], parts_by_id: Dict[str, Part], kerf: float, randomize: bool
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
            
            est_vol = est_count * (orientation.l * orientation.w * orientation.t)
            if randomize:
                est_vol *= random.uniform(0.8, 1.2) # 부피 평가도 살짝 흔들어줌

            lw_x = _axis_waste(node.dims.l, orientation.l, kerf)
            lw_y = _axis_waste(node.dims.w, orientation.w, kerf)
            lw_z = _axis_waste(node.dims.t, orientation.t, kerf)
            total_linear_waste = lw_x + lw_y + lw_z

            rot_penalty = 0 if (orientation.l == part.dims.l and orientation.w == part.dims.w and orientation.t == part.dims.t) else (1 if orientation.t == part.dims.t else 2)

            order_result = _best_cut_order(node, orientation, kerf, randomize)
            if order_result is None: continue

            best_order, max_offcut = order_result

            candidate = PlacementCandidate(
                neg_estimated_volume=-est_vol,
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

@dataclass
class PackResult:
    occupied_nodes: List[Node]
    unplaced: Dict[str, int]
    processing_time: float
    stocks_used: int
    free_nodes: List[Node] = field(default_factory=list)

def _pack_parts_single(
    settings: EngineSettings, stocks: List[Stock], parts: List[Part], randomize: bool
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

        candidate = _find_best_candidate(node, remaining, parts_by_id, kerf, randomize)
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

def pack_parts(
    settings: EngineSettings, stocks: List[Stock], parts: List[Part],
) -> PackResult:
    """
    ✨ 5초 극한 최적화 엔진 (GRASP)
    """
    start_total = time.perf_counter()
    TIME_LIMIT = 5.0  # 대표님이 허락하신 5초의 최적화 시간
    
    best_result = None
    best_unplaced = float('inf')
    best_waste = float('inf')
    
    # 1. 무작위성 없는 '정석' 모드로 1회 실행
    best_result = _pack_parts_single(settings, stocks, parts, randomize=False)
    best_unplaced = sum(best_result.unplaced.values())
    best_waste = sum(n.volume for n in best_result.free_nodes)
    
    if best_unplaced == 0:
        best_result.processing_time = time.perf_counter() - start_total
        return best_result

    # 2. 정석으로 안 된다면 5초 동안 수백 번 무작위 탐색 시도
    while True:
        if time.perf_counter() - start_total > TIME_LIMIT:
            break
            
        # 부품 우선순위를 매번 다르게 섞습니다.
        test_parts = copy.deepcopy(parts)
        strategy = random.random()
        if strategy < 0.3:
            test_parts.sort(key=lambda p: -(p.l * p.w * p.t))
        elif strategy < 0.6:
            test_parts.sort(key=lambda p: -max(p.l, p.w))
        else:
            random.shuffle(test_parts)
            
        test_stocks = copy.deepcopy(stocks)
        
        # 무작위성을 부여하여 시뮬레이션
        result = _pack_parts_single(settings, test_stocks, test_parts, randomize=True)
        unplaced = sum(result.unplaced.values())
        waste = sum(n.volume for n in result.free_nodes)
        
        # 더 나은 결과를 찾으면 갱신 (미배치가 적거나, 미배치가 같은데 쓰레기가 적은 경우)
        if unplaced < best_unplaced or (unplaced == best_unplaced and waste < best_waste):
            best_unplaced = unplaced
            best_waste = waste
            best_result = result
            
            if best_unplaced == 0:
                break

    best_result.processing_time = time.perf_counter() - start_total
    return best_result
