"""
packer.py — 탐색 & 배치 엔진

공개 API: pack_parts(settings, stocks, parts) → (List[Node], Dict[str, int])

설계 원칙:
  - Node-Centric 탐색: 공간(Node) 중심으로 순회하여 자연스러운 혼합 배치 달성
  - Best-Fit: 낭비(node.volume - part.volume)가 가장 작은 매칭 선택
  - Max-Offcut: 6가지 절단 순서 중 가장 큰 단일 잔재를 남기는 순서 선택
  - NodeHeap: 지연 삭제 우선순위 큐 (부피 내림차순)
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
    """부피 내림차순 min-heap (지연 삭제 방식)"""

    def __init__(self):
        self._heap: list = []
        self._removed: set = set()

    def push(self, node: Node):
        # 2순위의 마이너스(-)를 뺍니다!
        # 이제 깊이가 같다면 '부피가 작은 공간(방금 자른 슬라이스)'부터 완벽히 채웁니다.
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
# 배치 후보 (유연함과 정돈됨의 황금 밸런스)
# ─────────────────────────────────────────────

@dataclass(order=True)
class PlacementCandidate:
    neg_estimated_count: int  # ✨ 1순위: 가장 많이 들어가는 덩어리 찾기 (기둥/띠 형성의 핵심)
    linear_waste: float       # ✨ 2순위: 덩어리 안에서 발생하는 1차원 쓰레기 최소화 (로스율 방어!)
    rotation_penalty: int     # ✨ 3순위: 동점이라면, 원래 자르던 방향(회전) 유지 (작업성 고려)
    neg_max_offcut: float     # ✨ 4순위: 자투리 공간을 잘게 쪼개지 말고 한 덩어리로 크게 남기기
    node_id: str = field(compare=False)
    node: Node = field(compare=False)
    part: Part = field(compare=False)
    orientation: Dims = field(compare=False)
    cut_order: Tuple[CutAxis, ...] = field(compare=False)


# ─────────────────────────────────────────────
# Max-Offcut 절단 순서 (모든 방향의 유연성 부활!)
# ─────────────────────────────────────────────

_ALL_AXES = [CutAxis.X, CutAxis.Y, CutAxis.Z]
_ALL_ORDERS = list(permutations(_ALL_AXES))  # 6가지 절단 순서를 다시 모두 허용합니다!

def _offcut_volumes_for_order(
    node: Node, part_dims: Dims, cut_order: Tuple[CutAxis, ...], kerf: float,
) -> Optional[Tuple[float, ...]]:
    remaining = {CutAxis.X: node.dims.l, CutAxis.Y: node.dims.w, CutAxis.Z: node.dims.t}
    part_size = {CutAxis.X: part_dims.l, CutAxis.Y: part_dims.w, CutAxis.Z: part_dims.t}
    offcut_vols = []

    for axis in cut_order:
        pos = part_size[axis]
        total = remaining[axis]
        if abs(total - pos) <= _EPSILON:
            remaining[axis] = pos
            offcut_vols.append(0.0)
            continue
        remainder = total - pos - kerf
        if remainder <= 0: return None  
        
        b_dims_map = {**remaining, axis: remainder}
        b_vol = b_dims_map[CutAxis.X] * b_dims_map[CutAxis.Y] * b_dims_map[CutAxis.Z]
        offcut_vols.append(b_vol)
        remaining[axis] = pos

    return tuple(offcut_vols)

def _best_cut_order(node: Node, part_dims: Dims, kerf: float) -> Optional[Tuple[Tuple[CutAxis, ...], float]]:
    best_order = None
    best_max_offcut = -1.0

    # 유연하게 6가지를 다 찔러보고, 가장 자투리가 크게 남는 칼질 순서를 지능적으로 선택합니다.
    for order in _ALL_ORDERS:
        result = _offcut_volumes_for_order(node, part_dims, order, kerf)
        if result is None: continue
        max_offcut = max(result) if result else 0.0
        if max_offcut > best_max_offcut:
            best_max_offcut = max_offcut
            best_order = order

    if best_order is None: return None
    return best_order, best_max_offcut


# ─────────────────────────────────────────────
# Best-Fit 후보 선택 (억지 규칙 철폐)
# ─────────────────────────────────────────────

def _fit_count(total: float, pdim: float, kerf: float) -> int:
    if pdim > total + _EPSILON: return 0
    return int((total + kerf + _EPSILON) // (pdim + kerf))

def _axis_waste(total: float, pdim: float, kerf: float) -> float:
    count = _fit_count(total, pdim, kerf)
    if count <= 0: return total
    waste = total - (count * pdim) - (count - 1) * kerf
    return max(0.0, waste)

def _find_best_candidate(
    node: Node, remaining_parts: Dict[str, int], parts_by_id: Dict[str, Part], kerf: float,
) -> Optional[PlacementCandidate]:
    best: Optional[PlacementCandidate] = None

    for part_id, qty in remaining_parts.items():
        if qty <= 0: continue
        part = parts_by_id[part_id]

        for orientation in part.allowed_orientations():
            if not orientation.fits_in(node.dims): continue

            # 1. 덩어리 잠재력 (우선순위 1)
            cx = _fit_count(node.dims.l, orientation.l, kerf)
            cy = _fit_count(node.dims.w, orientation.w, kerf)
            cz = _fit_count(node.dims.t, orientation.t, kerf)
            est_count = cx * cy * cz  

            # 2. 로스율 방어 (우선순위 2)
            lw_x = _axis_waste(node.dims.l, orientation.l, kerf)
            lw_y = _axis_waste(node.dims.w, orientation.w, kerf)
            lw_z = _axis_waste(node.dims.t, orientation.t, kerf)
            total_linear_waste = lw_x + lw_y + lw_z

            # 3. 회전 페널티 (우선순위 3 - 벌점만 부여)
            if orientation.l == part.dims.l and orientation.w == part.dims.w and orientation.t == part.dims.t:
                rot_penalty = 0
            elif orientation.t == part.dims.t:
                rot_penalty = 1
            else:
                rot_penalty = 2

            order_result = _best_cut_order(node, orientation, kerf)
            if order_result is None: continue
            best_order, max_offcut = order_result

            candidate = PlacementCandidate(
                neg_estimated_count=-est_count,   
                linear_waste=total_linear_waste,  
                rotation_penalty=rot_penalty,     
                neg_max_offcut=-max_offcut,       
                node_id=node.node_id,
                node=node,
                part=part,
                orientation=orientation,
                cut_order=best_order,
            )

            if best is None or candidate < best:
                best = candidate

    return best


# ─────────────────────────────────────────────
# 3단계 순차 관통 절단
# ─────────────────────────────────────────────

def _place_part_on_node(
    node: Node,
    part: Part,
    orientation: Dims,
    cut_order: Tuple[CutAxis, ...],
    kerf: float,
) -> Tuple[Node, List[Node]]:
    """
    node에 part를 orientation 방향으로 배치.
    최대 3번의 관통 절단 수행 (이미 딱 맞으면 해당 절단 생략).

    Returns:
        (occupied_node, new_free_nodes)
    """
    part_size = {
        CutAxis.X: orientation.l,
        CutAxis.Y: orientation.w,
        CutAxis.Z: orientation.t,
    }

    current = node
    new_free_nodes: List[Node] = []

    for axis in cut_order:
        pos = part_size[axis]
        total = _get_axis(current.dims, axis)

        # 이미 딱 맞으면 절단 생략
        if abs(total - pos) <= _EPSILON:
            continue

        child_a, child_b = split_node(current, axis, pos, kerf)
        new_free_nodes.append(child_b)
        current = child_a  # 계속 child_a를 파고듦

    # 최종 리프 노드에 부품 배치
    current.state = NodeState.OCCUPIED
    current.placed_part = part
    current.placed_part_dims = orientation

    return current, new_free_nodes


# ─────────────────────────────────────────────
# 결과 데이터 클래스
# ─────────────────────────────────────────────

@dataclass
class PackResult:
    occupied_nodes: List[Node]
    unplaced: Dict[str, int]
    processing_time: float
    stocks_used: int
    free_nodes: List[Node] = field(default_factory=list)  # ✨ 남은 잔재(빈 공간) 리스트 추가


# ─────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────

def pack_parts(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> PackResult:
    """
    메인 패킹 엔진.

    Args:
        settings: kerf, optimization_goal 등 엔진 설정
        stocks: 원장 목록
        parts: 부품 목록

    Returns:
        PackResult(occupied_nodes, unplaced, processing_time, stocks_used, free_nodes)
    """
    start = time.perf_counter()

    kerf = settings.kerf
    parts_by_id: Dict[str, Part] = {p.id: p for p in parts}

    # 남은 부품 수량 (우선순위 역순으로 정렬: priority 높을수록 먼저)
    remaining: Dict[str, int] = {}
    for p in sorted(parts, key=lambda x: -x.priority):
        remaining[p.id] = p.qty

    occupied_nodes: List[Node] = []
    heap = NodeHeap()
    stocks_used = 0

    # 원장을 순차적으로 열어가며 처리
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

    # 첫 번째 원장 열기
    if not _open_next_stock():
        return PackResult([], remaining, 0.0, 0)

    # 메인 루프: heap이 빌 때까지
    while True:
        # 배치할 부품이 남았는지 확인
        if not any(v > 0 for v in remaining.values()):
            break

        node = heap.pop()

        if node is None:
            # 현재 heap이 비었으면 다음 원장 열기
            if not _open_next_stock():
                break
            node = heap.pop()
            if node is None:
                break

        # 현재 node에 가장 잘 맞는 부품 탐색
        candidate = _find_best_candidate(node, remaining, parts_by_id, kerf)

        if candidate is None:
            # 이 공간에 아무것도 안 들어감 → DISCARD
            node.state = NodeState.DISCARDED
            continue

        # 배치 실행
        occupied, new_free = _place_part_on_node(
            candidate.node,
            candidate.part,
            candidate.orientation,
            candidate.cut_order,
            kerf,
        )
        occupied_nodes.append(occupied)
        remaining[candidate.part.id] -= 1

        # 새 FREE 잔재들을 heap에 등록
        for free_node in new_free:
            heap.push(free_node)

    # ✨ 루프 종료 후, 힙에 남아있는 사용되지 않은 잔재(FREE)들을 긁어모읍니다.
    free_nodes = []
    for item in heap._heap:
        node = item[-1]  # 튜플의 마지막 요소가 Node 객체
        if node.node_id not in heap._removed and node.state == NodeState.FREE:
            free_nodes.append(node)

    elapsed = time.perf_counter() - start
    return PackResult(
        occupied_nodes=occupied_nodes,
        unplaced={k: v for k, v in remaining.items() if v > 0},
        processing_time=elapsed,
        stocks_used=stocks_used,
        free_nodes=free_nodes,  # ✨ 수집한 잔재 반환
    )
