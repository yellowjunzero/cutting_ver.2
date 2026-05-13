"""
main.py — FastAPI HTTP 레이어 (Phase 4.0)

책임:
  - HTTP 요청/응답 직렬화 (Pydantic)
  - 입력 검증 + 의미론적 사전 검사 (model_validator)
  - CPU 작업 격리 (run_in_threadpool)
  - 예외 → HTTP 상태코드 변환
  - Phase 4.0: VirtualStrip 해체 → 개별 PlacedPartOut 변환

배포 명령:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import functools
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

import phase4 as _phase4
from core import (
    CuttingError,
    Cut,
    CutAxis,
    Dims,
    EngineSettings,
    InvalidCutError,
    Node,
    OptimizationGoal,
    Part,
    Stock,
    TrimmingMargins,
    _new_id,
)
from packer import StripAdapter
from virtual_strip import VirtualStrip

# Phase 4.0 time_budget (초)
_PHASE4_TIME_BUDGET: float = 30.0

app = FastAPI(
    title="3D Guillotine Cut Optimizer",
    version="4.0.0",
    description="목재·철강·스폰지 등 판재 최적 재단 API — Phase 4.0 Strip Engine",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic 입력 모델  (기존 골격 유지)
# ─────────────────────────────────────────────

class TrimmingIn(BaseModel):
    x: float = Field(0.0, ge=0, description="X축 양단 여백 합산 (mm)")
    y: float = Field(0.0, ge=0, description="Y축 양단 여백 합산 (mm)")
    z: float = Field(0.0, ge=0, description="Z축 양단 여백 합산 (mm)")


class SettingsIn(BaseModel):
    kerf: float = Field(3.0, ge=0, le=50, description="톱날 두께 손실 (mm)")
    trimming: TrimmingIn = Field(default_factory=TrimmingIn)
    optimization_goal: str = Field("MINIMIZE_WASTE")


class StockIn(BaseModel):
    id: str = Field(..., min_length=1)
    l: float = Field(..., gt=0, description="길이 (mm)")
    w: float = Field(..., gt=0, description="너비 (mm)")
    t: float = Field(..., gt=0, description="두께 (mm)")
    qty: int = Field(..., ge=1, le=1000)


class PartIn(BaseModel):
    id: str = Field(..., min_length=1)
    l: float = Field(..., gt=0)
    w: float = Field(..., gt=0)
    t: float = Field(..., gt=0)
    qty: int = Field(..., ge=1, le=10000)
    lock_z: bool = True
    allow_xy_rotation: bool = True
    priority: int = Field(0, ge=0, le=100)
    color: str = Field("#4f8ef7")


class OptimizeRequest(BaseModel):
    settings: SettingsIn
    stocks: List[StockIn] = Field(..., min_length=1)
    parts: List[PartIn] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_parts_fit_in_stocks(self) -> "OptimizeRequest":
        """각 Part가 최소 1개 Stock에 들어갈 수 있는지 사전 검증"""
        kerf = self.settings.kerf
        trim = self.settings.trimming

        for part in self.parts:
            part_orientations = [(part.l, part.w, part.t)]
            if part.allow_xy_rotation:
                part_orientations.append((part.w, part.l, part.t))
            if not part.lock_z:
                from itertools import permutations
                part_orientations = list(set(permutations([part.l, part.w, part.t])))

            fits = False
            for stock in self.stocks:
                usable_l = stock.l - trim.x
                usable_w = stock.w - trim.y
                usable_t = stock.t - trim.z
                if usable_l <= 0 or usable_w <= 0 or usable_t <= 0:
                    continue
                for pl, pw, pt in part_orientations:
                    if pl <= usable_l and pw <= usable_w and pt <= usable_t:
                        fits = True
                        break
                if fits:
                    break

            if not fits:
                raise ValueError(
                    f"부품 '{part.id}' ({part.l}×{part.w}×{part.t}mm)은 "
                    f"어떤 원장에도 들어가지 않습니다. "
                    f"원장 크기 또는 Trimming을 확인하세요."
                )

        return self


# ─────────────────────────────────────────────
# Pydantic 응답 모델  (기존 골격 유지)
# ─────────────────────────────────────────────

class DimsOut(BaseModel):
    l: float
    w: float
    t: float
    volume: float


class OriginOut(BaseModel):
    x: float
    y: float
    z: float


class OffcutOut(BaseModel):
    node_id: str
    stock_id: str
    dims: DimsOut
    origin: OriginOut


class CutRecordOut(BaseModel):
    cut_id: str
    axis: str
    position: float
    kerf: float
    parent_node_id: str


class PlacedPartOut(BaseModel):
    node_id: str
    stock_id: str
    part_id: str
    color: str
    placed_dims: DimsOut
    origin: OriginOut
    cut_history: List[CutRecordOut]
    depth: int
    # Phase 4.0 추가: Strip 해체 정보
    from_strip: bool = False        # True이면 VirtualStrip에서 해체된 부품
    strip_id: Optional[str] = None  # 원본 Strip ID


class StockSummaryOut(BaseModel):
    stock_id: str
    original_dims: DimsOut
    usable_dims: DimsOut
    placed_count: int
    placed_volume: float    # 실제 부품 volume 합 (kerf 낭비 제외)
    usable_volume: float
    efficiency_pct: float


class OptimizeResponse(BaseModel):
    placements: List[PlacedPartOut]
    offcuts: List[OffcutOut]
    unplaced: Dict[str, int]
    stock_summaries: List[StockSummaryOut]
    failures: List[str]
    stats: Dict[str, Any]


# ─────────────────────────────────────────────
# 도메인 변환 헬퍼  (기존 유지)
# ─────────────────────────────────────────────

def _build_engine_settings(s: SettingsIn) -> EngineSettings:
    return EngineSettings(
        kerf=s.kerf,
        trimming=TrimmingMargins(x=s.trimming.x, y=s.trimming.y, z=s.trimming.z),
        optimization_goal=OptimizationGoal.MINIMIZE_WASTE,
    )


def _build_stocks(raw: List[StockIn], trim: TrimmingIn) -> List[Stock]:
    return [
        Stock(
            id=s.id,
            dims=Dims(l=s.l, w=s.w, t=s.t),
            qty=s.qty,
            trimming=TrimmingMargins(x=trim.x, y=trim.y, z=trim.z),
        )
        for s in raw
    ]


def _build_parts(raw: List[PartIn]) -> List[Part]:
    return [
        Part(
            id=p.id,
            dims=Dims(l=p.l, w=p.w, t=p.t),
            qty=p.qty,
            lock_z=p.lock_z,
            allow_xy_rotation=p.allow_xy_rotation,
            priority=p.priority,
            color=p.color,
        )
        for p in raw
    ]


# ─────────────────────────────────────────────
# 응답 조립 헬퍼
# ─────────────────────────────────────────────

def _node_to_placed_out(node: Node) -> PlacedPartOut:
    """일반 부품 Node → PlacedPartOut (기존 로직 유지)"""
    dims   = node.placed_part_dims
    origin = node.origin
    history = node.collect_cut_history()

    return PlacedPartOut(
        node_id=node.node_id,
        stock_id=node.stock_id or "",
        part_id=node.placed_part.id,
        color=node.placed_part.color,
        placed_dims=DimsOut(l=dims.l, w=dims.w, t=dims.t, volume=dims.volume),
        origin=OriginOut(x=origin.x, y=origin.y, z=origin.z),
        cut_history=[
            CutRecordOut(
                cut_id=c.cut_id,
                axis=c.axis.value,
                position=c.position,
                kerf=c.kerf,
                parent_node_id=c.parent_node_id,
            )
            for c in history
        ],
        depth=node.depth,
        from_strip=False,
        strip_id=None,
    )


def _explode_strip_node(
    node: Node,
    strip: VirtualStrip,
    kerf: float,
) -> List[PlacedPartOut]:
    """
    VirtualStrip이 배치된 Node를 해체하여 개별 PlacedPartOut 목록 반환.

    내부 부품들이 X축 방향으로 순차 배열되어 있음을 이용해
    각 부품의 origin.x를 누적 계산한다.

    좌표 계산:
        current_x = node.origin.x   (Strip 전체 시작 X)
        부품 i 배치 후: current_x += part_dims.l + kerf

    Y축(W), Z축(T)은 Strip의 origin.y, origin.z를 모든 부품이 공유한다.
    (VirtualStrip은 T/W가 동일한 부품들의 묶음이므로 Y·Z는 이동 없음)

    cut_history는 Strip 노드의 절단 이력을 모든 내부 부품이 공유한다.
    (물리적으로 동일한 절단 공정을 거쳤기 때문)
    """
    results: List[PlacedPartOut] = []

    # Strip 전체의 시작 좌표
    base_x = node.origin.x
    base_y = node.origin.y
    base_z = node.origin.z

    # Strip 노드의 절단 이력 (모든 내부 부품이 공유)
    shared_history = node.collect_cut_history()
    cut_records = [
        CutRecordOut(
            cut_id=c.cut_id,
            axis=c.axis.value,
            position=c.position,
            kerf=c.kerf,
            parent_node_id=c.parent_node_id,
        )
        for c in shared_history
    ]

    current_x = base_x
    part_seq = 0   # 스트립 내 순번 (node_id 고유성 보장용)

    # internal_parts: [(Part, Dims, qty), ...] 길이 내림차순 정렬
    for p_obj, p_dims, qty in strip.internal_parts:
        for _ in range(qty):
            # 고유 node_id 생성 (Strip 노드 ID + 순번)
            synthetic_node_id = f"{node.node_id}_x{part_seq:03d}"

            results.append(PlacedPartOut(
                node_id=synthetic_node_id,
                stock_id=node.stock_id or "",
                part_id=p_obj.id,
                color=p_obj.color,
                placed_dims=DimsOut(
                    l=p_dims.l, w=p_dims.w, t=p_dims.t,
                    volume=p_dims.volume,
                ),
                origin=OriginOut(x=current_x, y=base_y, z=base_z),
                cut_history=cut_records,
                depth=node.depth,
                from_strip=True,
                strip_id=strip.strip_id,
            ))

            # 다음 부품 X 시작 위치 (부품 길이 + kerf)
            current_x += p_dims.l + kerf
            part_seq += 1

    return results


def _build_placements(
    result: "_phase4.Phase4Result",
    kerf: float,
) -> List[PlacedPartOut]:
    """
    Phase4Result.occupied_nodes를 순회하여 PlacedPartOut 목록 생성.

    처리 분기:
      - Strip 노드 (placed_part.id가 __strip__으로 시작):
            _explode_strip_node()로 내부 부품 개별 해체
      - 일반 노드:
            기존 _node_to_placed_out() 사용

    Returns:
        모든 개별 부품의 PlacedPartOut 목록 (Strip 해체 포함)
    """
    # strip_id → VirtualStrip 빠른 조회용 맵
    strip_map: Dict[str, VirtualStrip] = {
        s.strip_id: s for s in result.strips
    }

    placements: List[PlacedPartOut] = []

    for node in result.occupied_nodes:
        placed = node.placed_part
        if placed is None:
            continue

        if StripAdapter.is_strip_part(placed):
            # ── Strip 노드 해체 ──────────────────────────────────
            strip_id = StripAdapter.extract_strip_id(placed)
            strip = strip_map.get(strip_id)
            if strip is None:
                # 맵에 없는 경우 방어: 원본 노드 그대로 출력
                placements.append(_node_to_placed_out(node))
                continue
            placements.extend(_explode_strip_node(node, strip, kerf))
        else:
            # ── 일반 노드 ────────────────────────────────────────
            placements.append(_node_to_placed_out(node))

    return placements


def _build_stock_summaries_phase4(
    placements: List[PlacedPartOut],
    stocks: List[Stock],
) -> List[StockSummaryOut]:
    """
    해체된 PlacedPartOut 목록 기준으로 stock_summaries 집계.

    Strip 해체 후 개별 부품 volume을 집계하므로
    kerf 낭비가 제외된 실제 수율이 계산된다.

    usable_volume: 해당 stock_id가 실제로 사용된 슬롯 수 × usable_dims.volume
    (qty=2 원장 중 1장만 사용됐으면 1장 기준으로 집계)
    """
    # 배치 집계: stock_id → (placed_count, placed_volume)
    counts:  Dict[str, int]   = {}
    volumes: Dict[str, float] = {}

    for p in placements:
        sid = p.stock_id or "unknown"
        counts[sid]  = counts.get(sid, 0) + 1
        volumes[sid] = volumes.get(sid, 0.0) + p.placed_dims.volume

    summaries: List[StockSummaryOut] = []
    stocks_map = {s.id: s for s in stocks}

    for sid in counts:
        stock = stocks_map.get(sid)
        if stock is None:
            continue

        orig   = stock.dims
        usable = stock.usable_dims

        placed_vol = volumes.get(sid, 0.0)
        # usable_volume: placed_vol을 담기 위해 실제로 열린 원장 슬롯 수 추정
        # placed_vol / single_usable_vol을 올림하면 실제 사용 슬롯 수
        single_usable = usable.volume
        import math
        slots_used = max(1, math.ceil(placed_vol / single_usable)) if single_usable > 0 else 1
        usable_vol = single_usable * slots_used

        eff = round((placed_vol / usable_vol * 100) if usable_vol > 0 else 0.0, 2)

        summaries.append(StockSummaryOut(
            stock_id=sid,
            original_dims=DimsOut(l=orig.l,   w=orig.w,   t=orig.t,   volume=orig.volume),
            usable_dims=DimsOut(  l=usable.l, w=usable.w, t=usable.t, volume=usable.volume),
            placed_count=counts[sid],
            placed_volume=placed_vol,
            usable_volume=usable_vol,
            efficiency_pct=eff,
        ))

    return summaries


# ─────────────────────────────────────────────
# 동기 실행 래퍼 (ThreadPool에서 호출)
# ─────────────────────────────────────────────

def _run_packing(
    engine_settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> "_phase4.Phase4Result":
    """Phase 4.0 엔진 호출 (30초 타임 버짓)"""
    return _phase4.pack_parts_phase4(
        settings=engine_settings,
        stocks=stocks,
        parts=parts,
        time_budget=_PHASE4_TIME_BUDGET,
    )


# ─────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "4.0.0",
        "engine": "Phase 4.0 Strip Engine",
    }


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(body: OptimizeRequest) -> OptimizeResponse:
    engine_settings = _build_engine_settings(body.settings)
    stocks  = _build_stocks(body.stocks, body.settings.trimming)
    parts   = _build_parts(body.parts)
    kerf    = engine_settings.kerf

    try:
        result: _phase4.Phase4Result = await run_in_threadpool(
            functools.partial(_run_packing, engine_settings, stocks, parts)
        )
    except InvalidCutError as e:
        raise HTTPException(status_code=400, detail={
            "error": "물리 제약 위반",
            "detail": str(e),
            "error_code": "INVALID_CUT",
        })
    except CuttingError as e:
        raise HTTPException(status_code=422, detail={
            "error": "절단 엔진 오류",
            "detail": str(e),
            "error_code": "CUTTING_ERROR",
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "입력값 오류",
            "detail": str(e),
            "error_code": "VALUE_ERROR",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "서버 오류",
            "detail": str(e),
            "error_code": "INTERNAL_ERROR",
        })

    # ── Strip 해체 포함 PlacedPartOut 생성 ──────────────────────────
    placements = _build_placements(result, kerf)

    # ── stock_summaries (해체된 개별 부품 volume 기준 집계) ──────────
    summaries = _build_stock_summaries_phase4(placements, stocks)

    # ── 잔재(offcut) 데이터 ──────────────────────────────────────────
    offcuts = [
        OffcutOut(
            node_id=n.node_id,
            stock_id=n.stock_id or "",
            dims=DimsOut(l=n.dims.l, w=n.dims.w, t=n.dims.t, volume=n.dims.volume),
            origin=OriginOut(x=n.origin.x, y=n.origin.y, z=n.origin.z),
        )
        for n in result.free_nodes
    ]

    # ── 미배치 사유 ──────────────────────────────────────────────────
    failures = [
        f"부품 '{pid}' {qty}개를 배치하지 못했습니다. 원장 공간이 부족합니다."
        for pid, qty in result.unplaced.items()
    ]

    # ── 통계 (Phase 4.0 확장) ────────────────────────────────────────
    # total_placed_vol: 해체된 개별 부품 volume 합 (kerf 낭비 제외)
    # total_usable_vol: 실제 사용된 원장 슬롯의 usable_volume 합
    # overall_eff:      사용 원장 대비 실제 배치 부품 수율
    # yield_rate_pct:   전체 투입 원장(qty 합산) 대비 배치 부품 수율
    total_placed_vol = sum(s.placed_volume for s in summaries)
    total_usable_vol = sum(s.usable_volume for s in summaries)   # 사용된 원장만

    overall_eff = round(
        (total_placed_vol / total_usable_vol * 100) if total_usable_vol > 0 else 0.0, 2
    )

    # yield_rate: 전체 원장(qty 포함) usable_volume 대비 실제 배치 vol
    total_stock_usable = sum(s.usable_dims.volume * s.qty for s in stocks)
    yield_rate_pct = round(
        (total_placed_vol / total_stock_usable * 100) if total_stock_usable > 0 else 0.0, 2
    )

    s = result.stats
    stats: Dict[str, Any] = {
        # 기존 키 (프론트엔드 호환)
        "total_placed":          len(placements),
        "total_unplaced_types":  len(result.unplaced),
        "total_placed_volume":   total_placed_vol,
        "total_usable_volume":   total_usable_vol,
        "overall_efficiency_pct": overall_eff,
        "stocks_used":           result.stocks_used,
        "processing_time_sec":   round(result.processing_time, 4),

        # Phase 4.0 신규 키
        "yield_rate_pct":        yield_rate_pct,
        "n_strips":              s.n_strips,
        "n_groups":              s.n_groups,
        "strip_assigned":        s.n_assigned,
        "strip_unassigned":      s.n_unassigned_strips,
        "strip_assignment_rate": round(s.strip_assignment_rate * 100, 1),
        "fallback_placed":       s.n_fallback_placed,
        "step_times": {
            "step1_dp_sec":      round(s.step1_sec, 4),
            "step2_strip_sec":   round(s.step2_sec, 4),
            "step3_assign_sec":  round(s.step3_sec, 4),
            "step4_place_sec":   round(s.step4_sec, 4),
        },
    }

    return OptimizeResponse(
        placements=placements,
        offcuts=offcuts,
        unplaced=result.unplaced,
        stock_summaries=summaries,
        failures=failures,
        stats=stats,
    )