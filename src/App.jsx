import { useState, useMemo, useRef, useEffect, useCallback } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import * as THREE from "three";

// ─────────────────────────────────────────────
// 상수
// ─────────────────────────────────────────────
const API_URL = "https://cutting-ver-2.onrender.com";
const SCALE = 0.001;       // 1mm → 0.001 three.js unit
const STOCK_GAP = 0.5;     // 원장 간격 (three.js unit)
const CAM_X_OFFSET = -0.73; // 사이드바 292px 보정

const PART_COLORS = [
  "#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6",
  "#06b6d4","#f97316","#ec4899","#84cc16","#14b8a6",
  "#6366f1","#e11d48","#0ea5e9","#d97706","#7c3aed",
];

let _colorIdx = 0;
const nextColor = () => PART_COLORS[_colorIdx++ % PART_COLORS.length];

const uid = () => Math.random().toString(36).slice(2, 9);

const DEFAULT_SETTINGS = { kerf: 3, trimming: { x: 10, y: 10, z: 0 } };

const DEFAULT_STOCKS = [
  { _uid: uid(), id: "S1", l: 2440, w: 1220, t: 18, qty: 2 },
];

const DEFAULT_PARTS = [
  { _uid: uid(), id: "P1", l: 600, w: 400, t: 18, qty: 8, lock_z: true, allow_xy_rotation: true, priority: 0, color: PART_COLORS[0] },
  { _uid: uid(), id: "P2", l: 300, w: 250, t: 18, qty: 6, lock_z: true, allow_xy_rotation: true, priority: 0, color: PART_COLORS[1] },
];

// ─────────────────────────────────────────────
// 3D 컴포넌트: 배치된 부품 박스
// ─────────────────────────────────────────────
function PlacedBox({ placement, zOffset }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const { l, w, t } = placement.placed_dims;
  const { x, y, z } = placement.origin;

  const cx = (x + l / 2) * SCALE;
  const cy = (y + w / 2) * SCALE;
  const cz = (z + t / 2) * SCALE + zOffset;

  const color = new THREE.Color(placement.color);
  const emissive = hovered ? new THREE.Color(0xffffff) : new THREE.Color(0x000000);

  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.material.emissiveIntensity = hovered ? 0.15 : 0;
    }
  });

  return (
    <group>
      <mesh
        ref={meshRef}
        position={[cx, cy, cz]}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
        onPointerOut={() => setHovered(false)}
      >
        <boxGeometry args={[l * SCALE, w * SCALE, t * SCALE]} />
        <meshStandardMaterial
          color={color}
          emissive={emissive}
          emissiveIntensity={0}
          transparent
          opacity={0.82}
          roughness={0.4}
          metalness={0.1}
        />
      </mesh>

      {/* 엣지 라인 */}
      <BoxEdges
        cx={cx} cy={cy} cz={cz}
        l={l * SCALE} w={w * SCALE} t={t * SCALE}
        hovered={hovered}
      />

      {/* 호버 툴팁 */}
      {hovered && (
        <Html position={[cx, cy + w * SCALE * 0.5 + 0.05, cz]} center>
          <div style={{
            background: "rgba(10,10,20,0.92)",
            border: "1px solid rgba(255,255,255,0.15)",
            borderRadius: 8,
            padding: "8px 12px",
            color: "#fff",
            fontSize: 12,
            fontFamily: "'DM Mono', monospace",
            whiteSpace: "nowrap",
            pointerEvents: "none",
          }}>
            <div style={{ fontWeight: 700, marginBottom: 4, color: placement.color }}>
              {placement.part_id}
            </div>
            <div style={{ color: "#aaa" }}>
              {placement.placed_dims.l} × {placement.placed_dims.w} × {placement.placed_dims.t} mm
            </div>
            <div style={{ color: "#666", fontSize: 10, marginTop: 3 }}>
              x:{placement.origin.x.toFixed(1)} y:{placement.origin.y.toFixed(1)} z:{placement.origin.z.toFixed(1)}
            </div>
            <div style={{ color: "#666", fontSize: 10 }}>
              절단 {placement.cut_history.length}회
            </div>
          </div>
        </Html>
      )}
    </group>
  );
}

function BoxEdges({ cx, cy, cz, l, w, t, hovered }) {
  const hl = l / 2, hw = w / 2, ht = t / 2;
  const corners = [
    [-hl,-hw,-ht],[hl,-hw,-ht],[hl,hw,-ht],[-hl,hw,-ht],
    [-hl,-hw,ht],[hl,-hw,ht],[hl,hw,ht],[-hl,hw,ht],
  ].map(([x,y,z]) => new THREE.Vector3(cx+x, cy+y, cz+z));

  const edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
  ];

  const color = hovered ? "#ffffff" : "rgba(255,255,255,0.25)";

  return (
    <>
      {edges.map(([a, b], i) => (
        <Line
          key={i}
          points={[corners[a], corners[b]]}
          color={color}
          lineWidth={hovered ? 1.5 : 0.8}
        />
      ))}
    </>
  );
}

// ─────────────────────────────────────────────
// 3D 컴포넌트: 원장 아웃라인 (usable_dims 기준)
// ─────────────────────────────────────────────
function StockOutline({ summary, zOffset }) {
  const { l, w, t } = summary.usable_dims;
  // usable 영역의 origin = trimming 오프셋
  // original_dims - usable_dims 차이의 절반이 trim
  const trimX = (summary.original_dims.l - l) / 2;
  const trimY = (summary.original_dims.w - w) / 2;

  const cx = (trimX + l / 2) * SCALE;
  const cy = (trimY + w / 2) * SCALE;
  const cz = (t / 2) * SCALE + zOffset;

  const hl = l * SCALE / 2, hw = w * SCALE / 2, ht = t * SCALE / 2;
  const corners = [
    [-hl,-hw,-ht],[hl,-hw,-ht],[hl,hw,-ht],[-hl,hw,-ht],
    [-hl,-hw,ht],[hl,-hw,ht],[hl,hw,ht],[-hl,hw,ht],
  ].map(([x,y,z]) => new THREE.Vector3(cx+x, cy+y, cz+z));

  const edges = [
    [0,1],[1,2],[2,3],[3,0],
    [4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7],
  ];

  return (
    <>
      {edges.map(([a, b], i) => (
        <Line key={i} points={[corners[a], corners[b]]} color="#3b82f6" lineWidth={1.2} />
      ))}
    </>
  );
}

// ─────────────────────────────────────────────
// 카메라 자동 프레이밍
// ─────────────────────────────────────────────
function CameraController({ response, orbitRef }) {
  const { camera } = useThree();

  useEffect(() => {
    if (!response || !response.placements.length) return;

    // 전체 바운딩 박스 계산
    let maxZ = 0;
    let maxX = 0;
    let maxY = 0;

    let zOffset = 0;
    response.stock_summaries.forEach((s) => {
      maxZ = Math.max(maxZ, (s.usable_dims.t * SCALE + zOffset));
      maxX = Math.max(maxX, s.usable_dims.l * SCALE);
      maxY = Math.max(maxY, s.usable_dims.w * SCALE);
      zOffset += s.original_dims.t * SCALE + STOCK_GAP;
    });

    const cx = maxX / 2 + CAM_X_OFFSET;
    const cy = maxY / 2;
    const cz = maxZ / 2;

    const size = Math.max(maxX, maxY, maxZ);

    if (orbitRef.current) {
      orbitRef.current.target.set(cx, cy, cz);
    }

    camera.position.set(cx + size * 1.2, cy + size * 0.8, cz + size * 1.5);
    camera.lookAt(cx, cy, cz);
  }, [response]);

  return null;
}

// ─────────────────────────────────────────────
// 씬 전체
// ─────────────────────────────────────────────
function Scene({ response }) {
  const orbitRef = useRef();

  // 원장별 Z 오프셋 계산
  const stockZOffsets = useMemo(() => {
    if (!response) return {};
    const offsets = {};
    let z = 0;
    response.stock_summaries.forEach((s) => {
      offsets[s.stock_id] = z;
      z += s.original_dims.t * SCALE + STOCK_GAP;
    });
    return offsets;
  }, [response]);

  // stock_id 중복 없이 처리 (복수 원장)
  const stockList = useMemo(() => {
    if (!response) return [];
    const seen = new Set();
    return response.stock_summaries.filter((s) => {
      if (seen.has(s.stock_id)) return false;
      seen.add(s.stock_id);
      return true;
    });
  }, [response]);

  return (
    <>
      <CameraController response={response} orbitRef={orbitRef} />
      <OrbitControls ref={orbitRef} makeDefault />

      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 8, 5]} intensity={1.0} castShadow />
      <directionalLight position={[-3, -2, -3]} intensity={0.3} />

      <gridHelper
        args={[20, 40, "#1e293b", "#1e293b"]}
        position={[0, -0.01, 0]}
        rotation={[0, 0, 0]}
      />

      {/* 원장 아웃라인 */}
      {response && stockList.map((s) => (
        <StockOutline
          key={s.stock_id}
          summary={s}
          zOffset={stockZOffsets[s.stock_id] ?? 0}
        />
      ))}

      {/* 배치된 부품들 */}
      {response && response.placements.map((p) => (
        <PlacedBox
          key={p.node_id}
          placement={p}
          zOffset={stockZOffsets[p.stock_id] ?? 0}
        />
      ))}
    </>
  );
}

// ─────────────────────────────────────────────
// 작업 지시서 모달
// ─────────────────────────────────────────────
function buildCutList(response) {
  const byStock = {};

  response.placements.forEach((p) => {
    const sid = p.stock_id;
    if (!byStock[sid]) byStock[sid] = new Map();

    p.cut_history.forEach((cut) => {
      if (!byStock[sid].has(cut.cut_id)) {
        byStock[sid].set(cut.cut_id, cut);
      }
    });
  });

  return Object.entries(byStock).map(([stockId, cutMap]) => {
    const cuts = Array.from(cutMap.values()).map((c, i) => ({ ...c, step: i + 1 }));
    return { stockId, cuts };
  });
}

function CutListModal({ response, onClose }) {
  const [activeStock, setActiveStock] = useState(0);
  const cutList = useMemo(() => buildCutList(response), [response]);

  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  const axisLabel = (axis) => ({ X: "← L →", Y: "← W →", Z: "← T →" }[axis] || axis);
  const axisDesc = (cut) => {
    const label = { X: "길이(L)축", Y: "너비(W)축", Z: "두께(T)축" }[cut.axis] || cut.axis;
    return `${label} ${cut.position.toFixed(1)}mm 지점에서 Kerf ${cut.kerf}mm 관통 절단`;
  };

  return (
    <div style={{
      position: "fixed", inset: 0,
      background: "rgba(0,0,0,0.75)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 1000,
      backdropFilter: "blur(4px)",
    }} onClick={onClose}>
      <div style={{
        background: "#0f172a",
        border: "1px solid #1e3a5f",
        borderRadius: 16,
        width: "min(90vw, 760px)",
        maxHeight: "80vh",
        display: "flex", flexDirection: "column",
        overflow: "hidden",
      }} onClick={(e) => e.stopPropagation()}>

        {/* 헤더 */}
        <div style={{
          padding: "20px 24px 16px",
          borderBottom: "1px solid #1e293b",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div>
            <div style={{ color: "#fff", fontWeight: 700, fontSize: 18, fontFamily: "'DM Mono', monospace" }}>
              📋 작업 지시서
            </div>
            <div style={{ color: "#64748b", fontSize: 12, marginTop: 2 }}>
              원장별 순차 절단 지시 — 동일 cut_id = 1회 물리적 절단
            </div>
          </div>
          <button onClick={onClose} style={{
            background: "none", border: "none",
            color: "#64748b", fontSize: 20, cursor: "pointer",
            lineHeight: 1,
          }}>✕</button>
        </div>

        {/* 원장 탭 */}
        <div style={{
          display: "flex", gap: 4, padding: "12px 24px 0",
          borderBottom: "1px solid #1e293b",
        }}>
          {cutList.map((s, i) => (
            <button key={s.stockId} onClick={() => setActiveStock(i)} style={{
              padding: "6px 14px", borderRadius: "6px 6px 0 0",
              border: "1px solid",
              borderColor: activeStock === i ? "#3b82f6" : "#1e293b",
              borderBottom: "none",
              background: activeStock === i ? "#1e3a5f" : "transparent",
              color: activeStock === i ? "#60a5fa" : "#64748b",
              fontSize: 12, cursor: "pointer",
              fontFamily: "'DM Mono', monospace",
            }}>
              {s.stockId}
            </button>
          ))}
        </div>

        {/* 절단 테이블 */}
        <div style={{ overflowY: "auto", padding: "16px 24px 24px" }}>
          {cutList[activeStock] && (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ color: "#475569" }}>
                  {["Step", "축", "위치(mm)", "Kerf(mm)", "작업 지시"].map((h) => (
                    <th key={h} style={{
                      padding: "8px 10px", textAlign: "left",
                      borderBottom: "1px solid #1e293b",
                      fontFamily: "'DM Mono', monospace",
                      fontWeight: 500,
                    }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cutList[activeStock].cuts.map((cut) => (
                  <tr key={cut.cut_id} style={{ borderBottom: "1px solid #0f172a" }}
                    onMouseEnter={(e) => e.currentTarget.style.background = "#1e293b"}
                    onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                  >
                    <td style={{ padding: "10px 10px", color: "#94a3b8", fontFamily: "'DM Mono', monospace" }}>
                      {String(cut.step).padStart(2, "0")}
                    </td>
                    <td style={{ padding: "10px 10px" }}>
                      <span style={{
                        background: { X: "#1e3a5f", Y: "#1a3320", Z: "#2d1b69" }[cut.axis],
                        color: { X: "#60a5fa", Y: "#4ade80", Z: "#a78bfa" }[cut.axis],
                        padding: "2px 8px", borderRadius: 4,
                        fontSize: 11, fontFamily: "'DM Mono', monospace",
                      }}>{axisLabel(cut.axis)}</span>
                    </td>
                    <td style={{ padding: "10px 10px", color: "#e2e8f0", fontFamily: "'DM Mono', monospace" }}>
                      {cut.position.toFixed(1)}
                    </td>
                    <td style={{ padding: "10px 10px", color: "#f59e0b", fontFamily: "'DM Mono', monospace" }}>
                      {cut.kerf}
                    </td>
                    <td style={{ padding: "10px 10px", color: "#94a3b8", fontSize: 12 }}>
                      {axisDesc(cut)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// 입력 폼 컴포넌트
// ─────────────────────────────────────────────
function NumberInput({ value, onChange, min = 0, label, unit = "mm" }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      {label && (
        <label style={{ fontSize: 10, color: "#64748b", fontFamily: "'DM Mono', monospace", letterSpacing: 1 }}>
          {label}
        </label>
      )}
      <div style={{ position: "relative" }}>
        <input
          type="number"
          value={value}
          min={min}
          onChange={(e) => onChange(Number(e.target.value))}
          style={{
            width: "100%", boxSizing: "border-box",
            background: "#0f172a",
            border: "1px solid #1e293b",
            borderRadius: 6,
            color: "#e2e8f0",
            padding: "7px 28px 7px 10px",
            fontSize: 13,
            fontFamily: "'DM Mono', monospace",
            outline: "none",
          }}
          onFocus={(e) => e.target.style.borderColor = "#3b82f6"}
          onBlur={(e) => e.target.style.borderColor = "#1e293b"}
        />
        <span style={{
          position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)",
          fontSize: 10, color: "#475569",
        }}>{unit}</span>
      </div>
    </div>
  );
}

function CheckBox({ checked, onChange, label }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", fontSize: 12, color: "#94a3b8" }}>
      <div
        onClick={() => onChange(!checked)}
        style={{
          width: 16, height: 16, borderRadius: 4,
          border: `2px solid ${checked ? "#3b82f6" : "#334155"}`,
          background: checked ? "#3b82f6" : "transparent",
          display: "flex", alignItems: "center", justifyContent: "center",
          flexShrink: 0,
        }}
      >
        {checked && <span style={{ color: "#fff", fontSize: 10, fontWeight: 900 }}>✓</span>}
      </div>
      {label}
    </label>
  );
}

// ─────────────────────────────────────────────
// 메인 App 컴포넌트
// ─────────────────────────────────────────────
export default function App() {
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [stocks, setStocks] = useState(DEFAULT_STOCKS);
  const [parts, setParts] = useState(DEFAULT_PARTS);
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);

  // API 전송용 body (_uid 제거)
  const requestBody = useMemo(() => ({
    settings: {
      kerf: settings.kerf,
      trimming: settings.trimming,
      optimization_goal: "MINIMIZE_WASTE",
    },
    stocks: stocks.map(({ _uid, ...s }) => s),
    parts: parts.map(({ _uid, ...p }) => p),
  }), [settings, stocks, parts]);

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_URL}/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const data = await res.json();
      if (!res.ok) {
        const msg = data.detail?.detail || data.detail || JSON.stringify(data);
        throw new Error(msg);
      }
      setResponse(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // ── Stock CRUD ──
  const addStock = () => setStocks((prev) => [
    ...prev,
    { _uid: uid(), id: `S${prev.length + 1}`, l: 2440, w: 1220, t: 18, qty: 1 }
  ]);
  const updateStock = (idx, key, val) => setStocks((prev) =>
    prev.map((s, i) => i === idx ? { ...s, [key]: val } : s)
  );
  const removeStock = (idx) => setStocks((prev) => prev.filter((_, i) => i !== idx));

  // ── Part CRUD ──
  const addPart = () => setParts((prev) => [
    ...prev,
    { _uid: uid(), id: `P${prev.length + 1}`, l: 400, w: 300, t: 18, qty: 4,
      lock_z: true, allow_xy_rotation: true, priority: 0, color: nextColor() }
  ]);
  const updatePart = (idx, key, val) => setParts((prev) =>
    prev.map((p, i) => i === idx ? { ...p, [key]: val } : p)
  );
  const removePart = (idx) => setParts((prev) => prev.filter((_, i) => i !== idx));

  // ── 스타일 상수 ──
  const sidebarStyle = {
    width: 292,
    minWidth: 292,
    height: "100vh",
    background: "#060d1a",
    borderRight: "1px solid #1e293b",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
  };

  const sectionLabelStyle = {
    fontSize: 10,
    fontWeight: 700,
    color: "#475569",
    letterSpacing: 2,
    textTransform: "uppercase",
    fontFamily: "'DM Mono', monospace",
    marginBottom: 8,
  };

  const cardStyle = {
    background: "#0f172a",
    border: "1px solid #1e293b",
    borderRadius: 10,
    padding: "12px 14px",
    marginBottom: 8,
  };

  const addBtnStyle = {
    width: "100%",
    padding: "8px",
    background: "transparent",
    border: "1px dashed #1e3a5f",
    borderRadius: 8,
    color: "#3b82f6",
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "'DM Mono', monospace",
    marginBottom: 8,
  };

  const removeBtnStyle = {
    background: "none",
    border: "none",
    color: "#475569",
    cursor: "pointer",
    fontSize: 14,
    padding: "0 2px",
    lineHeight: 1,
  };

  const stats = response?.stats;

  return (
    <div style={{ display: "flex", height: "100vh", background: "#060d1a", fontFamily: "system-ui, sans-serif" }}>

      {/* ── 사이드바 ── */}
      <div style={sidebarStyle}>

        {/* 로고 */}
        <div style={{
          padding: "18px 20px 14px",
          borderBottom: "1px solid #1e293b",
        }}>
          <div style={{
            fontSize: 15,
            fontWeight: 800,
            color: "#e2e8f0",
            fontFamily: "'DM Mono', monospace",
            letterSpacing: -0.5,
          }}>
            ✦ CUT OPTIMIZER
          </div>
          <div style={{ fontSize: 10, color: "#475569", marginTop: 2, letterSpacing: 1 }}>
            3D GUILLOTINE ENGINE v1.0
          </div>
        </div>

        {/* 스크롤 영역 */}
        <div style={{ flex: 1, overflowY: "auto", padding: "16px 14px", scrollbarWidth: "thin", scrollbarColor: "#1e293b transparent" }}>

          {/* Settings */}
          <div style={{ marginBottom: 20 }}>
            <div style={sectionLabelStyle}>⚙ Settings</div>
            <div style={cardStyle}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
                <NumberInput label="KERF" value={settings.kerf}
                  onChange={(v) => setSettings((s) => ({ ...s, kerf: v }))} />
                <div />
              </div>
              <div style={{ fontSize: 10, color: "#475569", marginBottom: 6, letterSpacing: 1 }}>TRIMMING (양단 합산)</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 6 }}>
                {["x","y","z"].map((ax) => (
                  <NumberInput key={ax} label={`T-${ax.toUpperCase()}`}
                    value={settings.trimming[ax]}
                    onChange={(v) => setSettings((s) => ({ ...s, trimming: { ...s.trimming, [ax]: v } }))}
                  />
                ))}
              </div>
            </div>
          </div>

          {/* Stocks */}
          <div style={{ marginBottom: 20 }}>
            <div style={sectionLabelStyle}>▣ 원장 (Stocks)</div>
            {stocks.map((s, i) => (
              <div key={s._uid} style={cardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <input
                    value={s.id}
                    onChange={(e) => updateStock(i, "id", e.target.value)}
                    style={{
                      background: "transparent", border: "none",
                      color: "#60a5fa", fontSize: 13, fontWeight: 700,
                      fontFamily: "'DM Mono', monospace", width: 80,
                    }}
                  />
                  <button onClick={() => removeStock(i)} style={removeBtnStyle}>✕</button>
                </div>
                {/* T → W → L → Qty 순서 */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 6 }}>
                  <NumberInput label="T (두께)" value={s.t} onChange={(v) => updateStock(i, "t", v)} />
                  <NumberInput label="W (폭)" value={s.w} onChange={(v) => updateStock(i, "w", v)} />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
                  <NumberInput label="L (길이)" value={s.l} onChange={(v) => updateStock(i, "l", v)} />
                  <NumberInput label="QTY" value={s.qty} min={1} onChange={(v) => updateStock(i, "qty", v)} unit="장" />
                </div>
              </div>
            ))}
            <button onClick={addStock} style={addBtnStyle}>+ 원장 추가</button>
          </div>

          {/* Parts */}
          <div style={{ marginBottom: 20 }}>
            <div style={sectionLabelStyle}>◈ 부품 (Parts)</div>
            {parts.map((p, i) => (
              <div key={p._uid} style={{ ...cardStyle, borderLeft: `3px solid ${p.color}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{
                      width: 10, height: 10, borderRadius: "50%",
                      background: p.color, flexShrink: 0,
                    }} />
                    <input
                      value={p.id}
                      onChange={(e) => updatePart(i, "id", e.target.value)}
                      style={{
                        background: "transparent", border: "none",
                        color: "#e2e8f0", fontSize: 13, fontWeight: 700,
                        fontFamily: "'DM Mono', monospace", width: 70,
                      }}
                    />
                  </div>
                  <button onClick={() => removePart(i)} style={removeBtnStyle}>✕</button>
                </div>
                {/* T → W → L → Qty */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 6 }}>
                  <NumberInput label="T (두께)" value={p.t} onChange={(v) => updatePart(i, "t", v)} />
                  <NumberInput label="W (폭)" value={p.w} onChange={(v) => updatePart(i, "w", v)} />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6, marginBottom: 8 }}>
                  <NumberInput label="L (길이)" value={p.l} onChange={(v) => updatePart(i, "l", v)} />
                  <NumberInput label="QTY" value={p.qty} min={1} onChange={(v) => updatePart(i, "qty", v)} unit="개" />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
                  <CheckBox
                    checked={p.lock_z}
                    onChange={(v) => updatePart(i, "lock_z", v)}
                    label="두께(Z) 고정"
                  />
                  <CheckBox
                    checked={p.allow_xy_rotation}
                    onChange={(v) => updatePart(i, "allow_xy_rotation", v)}
                    label="L↔W 회전 허용"
                  />
                </div>
              </div>
            ))}
            <button onClick={addPart} style={addBtnStyle}>+ 부품 추가</button>
          </div>

          {/* 결과 대시보드 */}
          {stats && (
            <div style={{ marginBottom: 16 }}>
              <div style={sectionLabelStyle}>◎ 결과</div>
              <div style={{ ...cardStyle, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                {[
                  ["효율", `${stats.overall_efficiency_pct}%`, "#4ade80"],
                  ["배치", `${stats.total_placed}개`, "#60a5fa"],
                  ["원장", `${stats.stocks_used}장`, "#a78bfa"],
                  ["시간", `${(stats.processing_time_sec * 1000).toFixed(1)}ms`, "#f59e0b"],
                ].map(([label, value, color]) => (
                  <div key={label}>
                    <div style={{ fontSize: 10, color: "#475569", letterSpacing: 1, fontFamily: "'DM Mono', monospace" }}>{label}</div>
                    <div style={{ fontSize: 20, fontWeight: 800, color, fontFamily: "'DM Mono', monospace", lineHeight: 1.2 }}>{value}</div>
                  </div>
                ))}
              </div>
              {response.failures.length > 0 && (
                <div style={{
                  background: "#1c0a0a", border: "1px solid #7f1d1d",
                  borderRadius: 8, padding: "10px 12px", marginTop: 8,
                }}>
                  {response.failures.map((f, i) => (
                    <div key={i} style={{ color: "#f87171", fontSize: 11, lineHeight: 1.5 }}>⚠ {f}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* 하단 버튼 영역 */}
        <div style={{ padding: "12px 14px", borderTop: "1px solid #1e293b", display: "flex", flexDirection: "column", gap: 8 }}>
          {response && (
            <button
              onClick={() => setShowModal(true)}
              style={{
                width: "100%", padding: "10px",
                background: "transparent",
                border: "1px solid #1e3a5f",
                borderRadius: 10,
                color: "#60a5fa",
                fontSize: 13, fontWeight: 600,
                cursor: "pointer",
                fontFamily: "'DM Mono', monospace",
              }}
            >
              📋 작업 지시서 보기
            </button>
          )}

          <button
            onClick={handleOptimize}
            disabled={loading}
            style={{
              width: "100%", padding: "14px",
              background: loading ? "#1e293b" : "linear-gradient(135deg, #1d4ed8, #3b82f6)",
              border: "none",
              borderRadius: 10,
              color: loading ? "#64748b" : "#fff",
              fontSize: 14, fontWeight: 800,
              cursor: loading ? "not-allowed" : "pointer",
              fontFamily: "'DM Mono', monospace",
              letterSpacing: 1,
              transition: "all 0.2s",
              boxShadow: loading ? "none" : "0 4px 20px rgba(59,130,246,0.3)",
            }}
          >
            {loading ? "계산 중..." : "▶ OPTIMIZE"}
          </button>

          {error && (
            <div style={{
              background: "#1c0a0a", border: "1px solid #7f1d1d",
              borderRadius: 8, padding: "10px 12px",
              color: "#f87171", fontSize: 11, lineHeight: 1.6,
              wordBreak: "break-word",
            }}>
              ⚠ {error}
            </div>
          )}
        </div>
      </div>

      {/* ── 3D 뷰어 ── */}
      <div style={{ flex: 1, position: "relative", background: "#030712" }}>
        <Canvas
          camera={{ fov: 45, near: 0.01, far: 1000, position: [2, 1.5, 3] }}
          style={{ width: "100%", height: "100%" }}
        >
          <Scene response={response} />
        </Canvas>

        {/* 빈 상태 */}
        {!response && !loading && (
          <div style={{
            position: "absolute", inset: 0,
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center",
            pointerEvents: "none",
          }}>
            <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>◈</div>
            <div style={{ color: "#334155", fontSize: 14, fontFamily: "'DM Mono', monospace" }}>
              원장과 부품을 입력하고 OPTIMIZE를 눌러주세요
            </div>
          </div>
        )}

        {/* 로딩 오버레이 */}
        {loading && (
          <div style={{
            position: "absolute", inset: 0,
            display: "flex", flexDirection: "column",
            alignItems: "center", justifyContent: "center",
            background: "rgba(3,7,18,0.7)",
            backdropFilter: "blur(4px)",
          }}>
            <div style={{
              width: 40, height: 40,
              border: "3px solid #1e293b",
              borderTop: "3px solid #3b82f6",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
            }} />
            <div style={{ color: "#64748b", marginTop: 16, fontFamily: "'DM Mono', monospace", fontSize: 12 }}>
              최적화 계산 중...
            </div>
          </div>
        )}

        {/* 뷰어 힌트 */}
        <div style={{
          position: "absolute", bottom: 16, left: 16,
          color: "#1e293b", fontSize: 11,
          fontFamily: "'DM Mono', monospace",
          lineHeight: 1.8,
          pointerEvents: "none",
        }}>
          <div>🖱 드래그: 회전 &nbsp; 우클릭: 이동 &nbsp; 휠: 줌</div>
        </div>

        {/* 범례 */}
        {response && (
          <div style={{
            position: "absolute", bottom: 16, right: 16,
            background: "rgba(6,13,26,0.9)",
            border: "1px solid #1e293b",
            borderRadius: 10,
            padding: "10px 14px",
            maxWidth: 200,
          }}>
            <div style={{ fontSize: 10, color: "#475569", letterSpacing: 1, marginBottom: 6, fontFamily: "'DM Mono', monospace" }}>
              범례
            </div>
            {parts.map((p) => {
              const placed = response.placements.filter((pl) => pl.part_id === p.id).length;
              if (placed === 0) return null;
              return (
                <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                  <div style={{ width: 10, height: 10, borderRadius: 2, background: p.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 11, color: "#94a3b8", fontFamily: "'DM Mono', monospace" }}>
                    {p.id} ({placed}개)
                  </span>
                </div>
              );
            })}
            <div style={{ borderTop: "1px solid #1e293b", marginTop: 6, paddingTop: 6 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 18, height: 2, background: "#3b82f6" }} />
                <span style={{ fontSize: 11, color: "#3b82f6", fontFamily: "'DM Mono', monospace" }}>원장 경계</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 작업 지시서 모달 */}
      {showModal && response && (
        <CutListModal response={response} onClose={() => setShowModal(false)} />
      )}

      {/* CSS 애니메이션 */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; background: #030712; }
        input[type=number]::-webkit-inner-spin-button,
        input[type=number]::-webkit-outer-spin-button { opacity: 0.4; }
        @keyframes spin { to { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 2px; }
      `}</style>
    </div>
  );
}
