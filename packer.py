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
        # 1순위: -node.depth (가장 깊은 노드 = 방금 자르고 남은 직속 잔재 우선)
        # 2순위: -node.volume (깊이가 같다면 그중에서 가장 큰 공간 우선)
        heapq.heappush(self._heap, (-node.depth, -node.volume, node.node_id, node))

    def pop(self) -> Optional[Node]:
        while self._heap:
            neg_vol, node_id, node = heapq.heappop(self._heap)
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
# 배치 후보 (정렬 가능)
# ─────────────────────────────────────────────

@dataclass(order=True)
class PlacementCandidate:
    score: float          # node.volume - part.volume (작을수록 좋음)
    neg_max_offcut: float # 음수 max_offcut (작을수록 잔재 큼)
    node_id: str = field(compare=False)
    node: Node = field(compare=False)
    part: Part = field(compare=False)
    orientation: Dims = field(compare=False)
    cut_order: Tuple[CutAxis, ...] = field(compare=False)


# ─────────────────────────────────────────────
# Max-Offcut 절단 순서 최적화
# ─────────────────────────────────────────────

_ALL_AXES = [CutAxis.X, CutAxis.Y, CutAxis.Z]
_ALL_ORDERS = list(permutations(_ALL_AXES))  # 6가지


def _offcut_volumes_for_order(
    node: Node,
    part_dims: Dims,
    cut_order: Tuple[CutAxis, ...],
    kerf: float,
) -> Optional[Tuple[float, ...]]:
    """
    실제 노드를 변경하지 않고 절단 순서를 시뮬레이션.
    반환: 각 단계의 child_b 부피 튜플, 절단 불가 시 None
    """
    remaining = {
        CutAxis.X: node.dims.l,
        CutAxis.Y: node.dims.w,
        CutAxis.Z: node.dims.t,
    }
    part_size = {
        CutAxis.X: part_dims.l,
        CutAxis.Y: part_dims.w,
        CutAxis.Z: part_dims.t,
    }
    offcut_vols = []

    for axis in cut_order:
        pos = part_size[axis]
        total = remaining[axis]

        # 이미 딱 맞으면 이 축 절단 불필요 (잔재=0)
        if abs(total - pos) <= _EPSILON:
            remaining[axis] = pos
            offcut_vols.append(0.0)
            continue

        remainder = total - pos - kerf
        if remainder <= 0:
            return None  # 이 순서로는 절단 불가

        # child_b 부피 = remainder × 나머지 두 축의 현재 크기
        b_dims_map = {**remaining, axis: remainder}
        b_vol = (
            b_dims_map[CutAxis.X]
            * b_dims_map[CutAxis.Y]
            * b_dims_map[CutAxis.Z]
        )
        offcut_vols.append(b_vol)
        remaining[axis] = pos

    return tuple(offcut_vols)


def _best_cut_order(
    node: Node,
    part_dims: Dims,
    kerf: float,
) -> Optional[Tuple[Tuple[CutAxis, ...], float]]:
    """
    6가지 절단 순서 중 max_offcut이 가장 큰 순서 선택.
    반환: (best_order, max_offcut_volume) 또는 None
    """
    best_order = None
    best_max_offcut = -1.0

    for order in _ALL_ORDERS:
        result = _offcut_volumes_for_order(node, part_dims, order, kerf)
        if result is None:
            continue
        max_offcut = max(result) if result else 0.0
        if max_offcut > best_max_offcut:
            best_max_offcut = max_offcut
            best_order = order

    if best_order is None:
        return None
    return best_order, best_max_offcut


# ─────────────────────────────────────────────
# Best-Fit 후보 선택
# ─────────────────────────────────────────────

def _find_best_candidate(
    node: Node,
    remaining_parts: Dict[str, int],
    parts_by_id: Dict[str, Part],
    kerf: float,
) -> Optional[PlacementCandidate]:
    """
    현재 node에 배치 가능한 가장 최적의 (part, orientation, cut_order) 조합 탐색.
    Dynamic Mixed Nesting: remaining_parts 전체를 순회하므로 혼합 배치 자동 달성.
    """
    best: Optional[PlacementCandidate] = None

    for part_id, qty in remaining_parts.items():
        if qty <= 0:
            continue
        part = parts_by_id[part_id]

        for orientation in part.allowed_orientations():
            # 기본 크기 체크
            if not orientation.fits_in(node.dims):
                continue

            # Max-Offcut 절단 순서 계산
            order_result = _best_cut_order(node, orientation, kerf)
            if order_result is None:
                continue

            best_order, max_offcut = order_result
            score = node.volume - orientation.volume

            candidate = PlacementCandidate(
                score=score,
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
        PackResult(occupied_nodes, unplaced, processing_time, stocks_used)
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

    elapsed = time.perf_counter() - start
    return PackResult(
        occupied_nodes=occupied_nodes,
        unplaced={k: v for k, v in remaining.items() if v > 0},
        processing_time=elapsed,
        stocks_used=stocks_used,
    )
