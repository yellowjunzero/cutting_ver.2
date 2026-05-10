"""
core.py — 도메인 모델 + 물리 법칙 레이어 (절대 수정 금지)

4대 물리 제약:
  C1: Strict Guillotine Cut  — 완전 관통 평면, 정확히 2자식 생성
  C2: Kerf Thickness         — 절단 시 kerf만큼 부피 소멸
  C3: Initial Trimming       — 원장 가장자리 여백 자동 반영
  C4: Orientation Mapping    — lock_z/allow_xy_rotation 기반 회전 제한
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────
# 열거형
# ─────────────────────────────────────────────

class CutAxis(str, Enum):
    X = "X"
    Y = "Y"
    Z = "Z"


class NodeState(str, Enum):
    FREE = "FREE"
    OCCUPIED = "OCCUPIED"
    SPLIT = "SPLIT"
    DISCARDED = "DISCARDED"


class OptimizationGoal(str, Enum):
    MINIMIZE_WASTE = "MINIMIZE_WASTE"


# ─────────────────────────────────────────────
# 예외
# ─────────────────────────────────────────────

class CuttingError(Exception):
    """절단 엔진 기본 예외"""


class InvalidCutError(CuttingError):
    """물리 제약 위반 절단 시도"""


# ─────────────────────────────────────────────
# 값 객체 (Immutable)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Dims:
    l: float  # X축 길이 (Length)
    w: float  # Y축 너비 (Width)
    t: float  # Z축 두께 (Thickness)

    def __post_init__(self):
        if self.l <= 0 or self.w <= 0 or self.t <= 0:
            raise ValueError(f"Dims는 모두 양수여야 합니다: l={self.l}, w={self.w}, t={self.t}")

    @property
    def volume(self) -> float:
        return self.l * self.w * self.t

    def fits_in(self, other: "Dims") -> bool:
        """self가 other 공간 안에 들어갈 수 있는지"""
        return self.l <= other.l and self.w <= other.w and self.t <= other.t

    def __le__(self, other: "Dims") -> bool:
        return self.fits_in(other)

    def __gt__(self, other: "Dims") -> bool:
        return not self.fits_in(other)


@dataclass(frozen=True)
class Point3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class TrimmingMargins:
    x: float = 0.0  # X축 양단 합산 여백
    y: float = 0.0
    z: float = 0.0


# ─────────────────────────────────────────────
# 입력 엔티티
# ─────────────────────────────────────────────

@dataclass
class Stock:
    id: str
    dims: Dims
    qty: int
    trimming: TrimmingMargins = field(default_factory=TrimmingMargins)

    @property
    def usable_dims(self) -> Dims:
        """[C3] 트리밍 적용 후 실제 사용 가능 치수"""
        # 기존: ul = self.dims.l - 2 * self.trimming.x (아래처럼 2 * 삭제)
        ul = self.dims.l - self.trimming.x
        uw = self.dims.w - self.trimming.y
        ut = self.dims.t - self.trimming.z
        if ul <= 0 or uw <= 0 or ut <= 0:
            raise ValueError(
                f"Stock '{self.id}'의 트리밍이 너무 커서 사용 가능 치수가 0 이하입니다."
            )
        return Dims(l=ul, w=uw, t=ut)

    @property
    def usable_volume(self) -> float:
        return self.usable_dims.volume


@dataclass
class Part:
    id: str
    dims: Dims
    qty: int
    lock_z: bool = True          # [C4] 두께 방향 회전 잠금
    allow_xy_rotation: bool = True  # [C4] L↔W 교환 허용
    priority: int = 0
    color: str = "#4f8ef7"       # UI 표시용 색상

    def allowed_orientations(self) -> List[Dims]:
        """[C4] 허용된 배치 방향 목록 반환"""
        l, w, t = self.dims.l, self.dims.w, self.dims.t
        if self.lock_z:
            # t 고정, l↔w만 교환 가능
            orientations = [Dims(l=l, w=w, t=t)]
            if self.allow_xy_rotation and l != w:
                orientations.append(Dims(l=w, w=l, t=t))
        else:
            # 6가지 순열
            perms = set()
            from itertools import permutations
            for perm in permutations([l, w, t]):
                perms.add(perm)
            orientations = [Dims(l=p[0], w=p[1], t=p[2]) for p in perms]
        return orientations


@dataclass(frozen=True)
class EngineSettings:
    kerf: float = 3.0
    trimming: TrimmingMargins = field(default_factory=TrimmingMargins)
    optimization_goal: OptimizationGoal = OptimizationGoal.MINIMIZE_WASTE


# ─────────────────────────────────────────────
# 절단 기록 (Immutable)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class Cut:
    cut_id: str
    axis: CutAxis
    position: float   # Node 로컬 좌표 기준 — child_a의 해당 축 크기
    kerf: float
    parent_node_id: str


# ─────────────────────────────────────────────
# Node — 트리의 핵심 단위
# ─────────────────────────────────────────────

@dataclass
class Node:
    node_id: str
    dims: Dims
    origin: Point3D
    state: NodeState = NodeState.FREE

    # 절단 이력
    cut: Optional[Cut] = None

    # 배치 정보
    placed_part: Optional[Part] = None
    placed_part_dims: Optional[Dims] = None

    # 트리 구조
    child_a: Optional["Node"] = None   # 앞쪽 (position 크기)
    child_b: Optional["Node"] = None   # 뒤쪽 잔재 (Offcut)
    parent: Optional["Node"] = None
    depth: int = 0
    stock_id: Optional[str] = None

    @property
    def volume(self) -> float:
        return self.dims.volume

    @property
    def is_leaf(self) -> bool:
        return self.child_a is None and self.child_b is None

    def collect_cut_history(self) -> List[Cut]:
        """루트까지 역추적하여 절단 이력 반환 (루트→리프 순서)"""
        history = []
        node = self
        while node is not None:
            if node.cut is not None:
                history.append(node.cut)
            node = node.parent
        history.reverse()
        return history


# ─────────────────────────────────────────────
# 팩토리 함수
# ─────────────────────────────────────────────

def create_root_node(stock: Stock) -> Node:
    """[C3] 트리밍이 반영된 루트 노드 생성"""
    usable = stock.usable_dims
    trim = stock.trimming
    # ✨ 트리밍(양단 합산)을 절반으로 나누어 양쪽 여백으로 균일하게 분배합니다!
    origin = Point3D(x=trim.x / 2.0, y=trim.y / 2.0, z=trim.z / 2.0)
    return Node(
        node_id=_new_id(),
        dims=usable,
        origin=origin,
        state=NodeState.FREE,
        stock_id=stock.id,
        depth=0,
    )


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


# ─────────────────────────────────────────────
# split_node() — C1·C2 물리 제약의 유일한 게이트키퍼
# ─────────────────────────────────────────────

def split_node(
    node: Node,
    axis: CutAxis,
    position: float,
    kerf: float,
) -> Tuple[Node, Node]:
    """
    node를 axis 방향 position 지점에서 절단한다.

    [C1] 완전 관통: 정확히 2개의 자식 노드 생성
    [C2] Kerf:     child_b의 해당 축 크기 = total - position - kerf

    Returns:
        (child_a, child_b)
        child_a: position 크기 (앞쪽)
        child_b: 잔재 (뒤쪽)

    Raises:
        InvalidCutError: 물리 제약 위반
    """
    # 사전 조건 검증
    if not node.is_leaf:
        raise InvalidCutError(f"Node '{node.node_id}'는 리프 노드가 아닙니다.")
    if node.state != NodeState.FREE:
        raise InvalidCutError(f"Node '{node.node_id}'의 상태가 FREE가 아닙니다: {node.state}")
    if kerf < 0:
        raise InvalidCutError(f"Kerf는 0 이상이어야 합니다: {kerf}")

    # 해당 축의 총 길이
    total_dim = _get_axis(node.dims, axis)

    if position <= 0:
        raise InvalidCutError(f"절단 위치는 0보다 커야 합니다: position={position}")
    if position >= total_dim:
        raise InvalidCutError(
            f"절단 위치({position})가 노드 크기({total_dim}) 이상입니다. 축={axis}"
        )

    remainder = total_dim - position - kerf
    if remainder <= 0:
        raise InvalidCutError(
            f"Kerf({kerf}) 차감 후 잔재 크기({remainder:.2f})가 0 이하입니다. "
            f"total={total_dim}, position={position}, axis={axis}"
        )

    # Cut 기록
    cut = Cut(
        cut_id=_new_id(),
        axis=axis,
        position=position,
        kerf=kerf,
        parent_node_id=node.node_id,
    )

    # child_a: position 크기
    a_dims = _replace_axis(node.dims, axis, position)
    a_origin = node.origin

    # child_b: 잔재 (remainder 크기, origin 이동) [C2]
    b_dims = _replace_axis(node.dims, axis, remainder)
    b_origin = _shift_origin(node.origin, axis, position + kerf)

    child_a = Node(
        node_id=_new_id(),
        dims=a_dims,
        origin=a_origin,
        state=NodeState.FREE,
        cut=cut,
        parent=node,
        depth=node.depth + 1,
        stock_id=node.stock_id,
    )
    child_b = Node(
        node_id=_new_id(),
        dims=b_dims,
        origin=b_origin,
        state=NodeState.FREE,
        cut=cut,
        parent=node,
        depth=node.depth + 1,
        stock_id=node.stock_id,
    )

    # 노드 상태 업데이트
    node.state = NodeState.SPLIT
    node.child_a = child_a
    node.child_b = child_b

    return child_a, child_b


# ─────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────

def _get_axis(dims: Dims, axis: CutAxis) -> float:
    if axis == CutAxis.X:
        return dims.l
    elif axis == CutAxis.Y:
        return dims.w
    else:
        return dims.t


def _replace_axis(dims: Dims, axis: CutAxis, value: float) -> Dims:
    if axis == CutAxis.X:
        return Dims(l=value, w=dims.w, t=dims.t)
    elif axis == CutAxis.Y:
        return Dims(l=dims.l, w=value, t=dims.t)
    else:
        return Dims(l=dims.l, w=dims.w, t=value)


def _shift_origin(origin: Point3D, axis: CutAxis, delta: float) -> Point3D:
    if axis == CutAxis.X:
        return Point3D(x=origin.x + delta, y=origin.y, z=origin.z)
    elif axis == CutAxis.Y:
        return Point3D(x=origin.x, y=origin.y + delta, z=origin.z)
    else:
        return Point3D(x=origin.x, y=origin.y, z=origin.z + delta)
