"""
main.py — FastAPI HTTP 레이어

책임:
  - HTTP 요청/응답 직렬화 (Pydantic)
  - 입력 검증 + 의미론적 사전 검사 (model_validator)
  - CPU 작업 격리 (run_in_threadpool)
  - 예외 → HTTP 상태코드 변환

배포 명령:
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

import packer
from core import (
    CuttingError,
    Dims,
    EngineSettings,
    InvalidCutError,
    OptimizationGoal,
    Part,
    Stock,
    TrimmingMargins,
)

app = FastAPI(
    title="3D Guillotine Cut Optimizer",
    version="1.0.0",
    description="목재·철강·스폰지 등 판재 최적 재단 API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic 입력 모델
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
            part_orientations = [
                (part.l, part.w, part.t),
            ]
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
# Pydantic 응답 모델
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


class StockSummaryOut(BaseModel):
    stock_id: str
    original_dims: DimsOut
    usable_dims: DimsOut
    placed_count: int
    placed_volume: float
    usable_volume: float
    efficiency_pct: float


class OptimizeResponse(BaseModel):
    placements: List[PlacedPartOut]
    offcuts: List[OffcutOut]
    unplaced: Dict[str, int]
    stock_summaries: List[StockSummaryOut]
    failures: List[str]   # 사용자 친화적 실패 사유
    stats: Dict[str, Any]


# ─────────────────────────────────────────────
# 도메인 변환 헬퍼
# ─────────────────────────────────────────────

def _build_engine_settings(s: SettingsIn) -> EngineSettings:
    return EngineSettings(
        kerf=s.kerf,
        trimming=TrimmingMargins(x=s.trimming.x, y=s.trimming.y, z=s.trimming.z),
        optimization_goal=OptimizationGoal.MINIMIZE_WASTE,
    )


def _build_stocks(raw: List[StockIn], trim: TrimmingIn) -> List[Stock]:
    result = []
    for s in raw:
        result.append(
            Stock(
                id=s.id,
                dims=Dims(l=s.l, w=s.w, t=s.t),
                qty=s.qty,
                trimming=TrimmingMargins(x=trim.x, y=trim.y, z=trim.z),
            )
        )
    return result


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

def _node_to_placed_out(node) -> PlacedPartOut:
    dims = node.placed_part_dims
    origin = node.origin
    history = node.collect_cut_history()

    return PlacedPartOut(
        node_id=node.node_id,
        stock_id=node.stock_id or "",
        part_id=node.placed_part.id,
        color=node.placed_part.color,
        placed_dims=DimsOut(
            l=dims.l, w=dims.w, t=dims.t, volume=dims.volume
        ),
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
    )


def _build_stock_summaries(
    occupied_nodes: list,
    stocks: List[Stock],
) -> List[StockSummaryOut]:
    # stock_id별 집계
    counts: Dict[str, int] = {}
    volumes: Dict[str, float] = {}

    for node in occupied_nodes:
        sid = node.stock_id or "unknown"
        counts[sid] = counts.get(sid, 0) + 1
        volumes[sid] = volumes.get(sid, 0.0) + (
            node.placed_part_dims.volume if node.placed_part_dims else 0.0
        )

    summaries = []
    stocks_map = {s.id: s for s in stocks}

    for sid in counts:
        stock = stocks_map.get(sid)
        if stock is None:
            continue

        orig = stock.dims
        usable = stock.usable_dims
        placed_vol = volumes.get(sid, 0.0)
        usable_vol = usable.volume
        eff = round((placed_vol / usable_vol * 100) if usable_vol > 0 else 0.0, 2)

        summaries.append(
            StockSummaryOut(
                stock_id=sid,
                original_dims=DimsOut(l=orig.l, w=orig.w, t=orig.t, volume=orig.volume),
                usable_dims=DimsOut(l=usable.l, w=usable.w, t=usable.t, volume=usable.volume),
                placed_count=counts[sid],
                placed_volume=placed_vol,
                usable_volume=usable_vol,
                efficiency_pct=eff,
            )
        )

    return summaries


# ─────────────────────────────────────────────
# 동기 실행 래퍼 (ThreadPool에서 호출)
# ─────────────────────────────────────────────

def _run_packing(
    engine_settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> packer.PackResult:
    return packer.pack_parts(engine_settings, stocks, parts)


# ─────────────────────────────────────────────
# 라우터
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(body: OptimizeRequest) -> OptimizeResponse:
    engine_settings = _build_engine_settings(body.settings)
    stocks = _build_stocks(body.stocks, body.settings.trimming)
    parts = _build_parts(body.parts)

    try:
        result: packer.PackResult = await run_in_threadpool(
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

    # 응답 조립
    placements = [_node_to_placed_out(n) for n in result.occupied_nodes]
    summaries = _build_stock_summaries(result.occupied_nodes, stocks)

    # ✨ 프론트엔드 투명 박스 렌더링을 위한 잔재 데이터 매핑
    offcuts = []
    for n in result.free_nodes:
        offcuts.append(OffcutOut(
            node_id=n.node_id,
            stock_id=n.stock_id or "",
            dims=DimsOut(l=n.dims.l, w=n.dims.w, t=n.dims.t, volume=n.dims.volume),
            origin=OriginOut(x=n.origin.x, y=n.origin.y, z=n.origin.z)
        ))

    # 미배치 사유 생성
    failures = []
    for part_id, qty in result.unplaced.items():
        failures.append(f"부품 '{part_id}' {qty}개를 배치하지 못했습니다. 원장 공간이 부족합니다.")

    # 통합 통계
    total_placed_vol = sum(s.placed_volume for s in summaries)
    total_usable_vol = sum(s.usable_volume for s in summaries)
    overall_eff = round(
        (total_placed_vol / total_usable_vol * 100) if total_usable_vol > 0 else 0.0, 2
    )

    stats = {
        "total_placed": len(placements),
        "total_unplaced_types": len(result.unplaced),
        "total_placed_volume": total_placed_vol,
        "total_usable_volume": total_usable_vol,
        "overall_efficiency_pct": overall_eff,
        "stocks_used": result.stocks_used,
        "processing_time_sec": round(result.processing_time, 4),
    }

    return OptimizeResponse(
        placements=placements,
        offcuts=offcuts,
        unplaced=result.unplaced,
        stock_summaries=summaries,
        failures=failures,
        stats=stats,
    )
