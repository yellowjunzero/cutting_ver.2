import { useState, useMemo, useRef, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html, Line } from "@react-three/drei";
import * as THREE from "three";

const API_URL = "https://cutting-ver-2.onrender.com";
const SCALE = 0.001;
const STOCK_GAP = 0.5;
const CAM_X_OFFSET = -0.73;
const STRIP_BBOX_COLOR = "#f59e0b";

const PART_COLORS = [
  "#3b82f6","#ef4444","#10b981","#f59e0b","#8b5cf6",
  "#06b6d4","#f97316","#ec4899","#84cc16","#14b8a6",
  "#6366f1","#e11d48","#0ea5e9","#d97706","#7c3aed",
];

const uid = () => Math.random().toString(36).slice(2, 9);
const DEFAULT_SETTINGS = { kerf: 5, trimming: { x: 0, y: 0, z: 0 } };
const DEFAULT_STOCKS = [{ _uid: uid(), id: "S1", l: 0, w: 0, t: 0, qty: 0 }];
const DEFAULT_PARTS  = [
  { _uid: uid(), id: "P1", l: 0, w: 0, t: 0, qty: 0,
    lock_z: false, allow_xy_rotation: true, priority: 0, color: PART_COLORS[0] },
];

function computeStockZOffsets(response) {
  if (!response) return {};
  const offsets = {}, seen = new Set();
  let z = 0;
  (response.stock_summaries || []).forEach((s) => {
    if (!seen.has(s.stock_id)) {
      offsets[s.stock_id] = z;
      z += s.original_dims.t * SCALE + STOCK_GAP;
      seen.add(s.stock_id);
    }
  });
  return offsets;
}

function computeStripBBoxes(placements, stockZOffsets) {
  const map = {};
  placements.forEach((p) => {
    if (!p.from_strip || !p.strip_id) return;
    const sid = p.strip_id;
    const zOff = stockZOffsets[p.stock_id] ?? 0;
    const { x, y, z } = p.origin;
    const { l, w, t } = p.placed_dims;
    const x1 = x*SCALE, x2 = (x+l)*SCALE;
    const y1 = y*SCALE, y2 = (y+w)*SCALE;
    const z1 = z*SCALE+zOff, z2 = (z+t)*SCALE+zOff;
    if (!map[sid]) {
      map[sid] = { minX:x1,maxX:x2,minY:y1,maxY:y2,minZ:z1,maxZ:z2 };
    } else {
      const b = map[sid];
      b.minX = Math.min(b.minX,x1); b.maxX = Math.max(b.maxX,x2);
      b.minY = Math.min(b.minY,y1); b.maxY = Math.max(b.maxY,y2);
      b.minZ = Math.min(b.minZ,z1); b.maxZ = Math.max(b.maxZ,z2);
    }
  });
  return map;
}

function BoxEdges({ cx, cy, cz, l, w, t, color = "rgba(255,255,255,0.25)", lineWidth = 0.8 }) {
  const hl=l/2, hw=w/2, ht=t/2;
  const corners = [
    [-hl,-hw,-ht],[hl,-hw,-ht],[hl,hw,-ht],[-hl,hw,-ht],
    [-hl,-hw,ht], [hl,-hw,ht], [hl,hw,ht], [-hl,hw,ht],
  ].map(([x,y,z]) => new THREE.Vector3(cx+x,cy+y,cz+z));
  const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
  return (
    <>
      {edges.map(([a,b],i)=>(
        <Line key={i} points={[corners[a],corners[b]]} color={color} lineWidth={lineWidth} />
      ))}
    </>
  );
}

function PlacedBox({ placement, zOffset }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const { l, w, t } = placement.placed_dims;
  const { x, y, z } = placement.origin;
  const cx = (x+l/2)*SCALE, cy = (y+w/2)*SCALE, cz = (z+t/2)*SCALE+zOffset;
  const isStrip = placement.from_strip;
  const baseColor = new THREE.Color(placement.color);

  useFrame(() => {
    if (meshRef.current) meshRef.current.material.emissiveIntensity = hovered ? 0.18 : 0;
  });

  return (
    <group>
      <mesh ref={meshRef} position={[cx,cy,cz]}
        onPointerOver={(e)=>{ e.stopPropagation(); setHovered(true); }}
        onPointerOut={()=>setHovered(false)}>
        <boxGeometry args={[l*SCALE,w*SCALE,t*SCALE]} />
        <meshStandardMaterial color={baseColor} emissive={baseColor} emissiveIntensity={0}
          transparent opacity={0.83} roughness={0.35} metalness={0.12} />
      </mesh>
      <BoxEdges cx={cx} cy={cy} cz={cz} l={l*SCALE} w={w*SCALE} t={t*SCALE}
        color={hovered?"#ffffff":(isStrip?"rgba(255,255,255,0.4)":"rgba(255,255,255,0.2)")}
        lineWidth={hovered?1.5:(isStrip?1.1:0.7)} />
      {hovered && (
        <Html position={[cx,cy+w*SCALE*0.5+0.05,cz]} center zIndexRange={[200,0]}>
          <div style={{
            background:"rgba(10,10,20,0.95)",
            border:`1px solid ${isStrip?STRIP_BBOX_COLOR:"rgba(255,255,255,0.15)"}`,
            borderRadius:8,padding:"8px 12px",color:"#fff",fontSize:12,
            fontFamily:"'DM Mono',monospace",whiteSpace:"nowrap",pointerEvents:"none",
            boxShadow:isStrip?`0 0 12px ${STRIP_BBOX_COLOR}44`:"none",
          }}>
            <div style={{fontWeight:700,marginBottom:4,color:placement.color}}>
              {placement.part_id}
              {isStrip&&<span style={{color:STRIP_BBOX_COLOR,fontSize:10,marginLeft:6}}>✦ STRIP</span>}
            </div>
            <div style={{color:"#aaa"}}>{l}×{w}×{t} mm</div>
            <div style={{color:"#666",fontSize:10,marginTop:3}}>
              x:{x.toFixed(1)} y:{y.toFixed(1)} z:{z.toFixed(1)}
            </div>
            <div style={{color:"#666",fontSize:10}}>절단 {placement.cut_history.length}회</div>
            {isStrip&&<div style={{color:STRIP_BBOX_COLOR,fontSize:10,marginTop:3}}>
              strip: {placement.strip_id?.slice(0,10)}
            </div>}
          </div>
        </Html>
      )}
    </group>
  );
}

function StripBoundingBox({ bbox }) {
  const { minX,maxX,minY,maxY,minZ,maxZ } = bbox;
  const cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
  const lx=maxX-minX, ly=maxY-minY, lz=maxZ-minZ;
  return <BoxEdges cx={cx} cy={cy} cz={cz} l={lx+0.001} w={ly+0.001} t={lz+0.001}
           color={STRIP_BBOX_COLOR} lineWidth={2.0} />;
}

function StockOutline({ summary, zOffset }) {
  const orig=summary.original_dims, usable=summary.usable_dims;
  const ocx=(orig.l/2)*SCALE, ocy=(orig.w/2)*SCALE, ocz=(orig.t/2)*SCALE+zOffset;
  const tx=(orig.l-usable.l)/2, ty=(orig.w-usable.w)/2, tz=(orig.t-usable.t)/2;
  const ucx=(tx+usable.l/2)*SCALE, ucy=(ty+usable.w/2)*SCALE, ucz=(tz+usable.t/2)*SCALE+zOffset;
  return (
    <group>
      <BoxEdges cx={ocx} cy={ocy} cz={ocz} l={orig.l*SCALE} w={orig.w*SCALE} t={orig.t*SCALE} color="#334155" lineWidth={1}/>
      <BoxEdges cx={ucx} cy={ucy} cz={ucz} l={usable.l*SCALE} w={usable.w*SCALE} t={usable.t*SCALE} color="#3b82f6" lineWidth={1.5}/>
    </group>
  );
}

function OffcutBox({ offcut, zOffset }) {
  const {l,w,t}=offcut.dims, {x,y,z}=offcut.origin;
  const [hovered,setHovered]=useState(false);
  if (l<6||w<6||t<6) return null;
  const cx=(x+l/2)*SCALE, cy=(y+w/2)*SCALE, cz=(z+t/2)*SCALE+zOffset;
  return (
    <group>
      <mesh position={[cx,cy,cz]}
        onPointerOver={(e)=>{ e.stopPropagation(); setHovered(true); }}
        onPointerOut={()=>setHovered(false)}>
        <boxGeometry args={[l*SCALE,w*SCALE,t*SCALE]}/>
        <meshPhysicalMaterial color="#38bdf8" transparent opacity={hovered?0.28:0.07}
          roughness={0.1} metalness={0.1} clearcoat={1} depthWrite={false}/>
        <lineSegments>
          <edgesGeometry args={[new THREE.BoxGeometry(l*SCALE,w*SCALE,t*SCALE)]}/>
          <lineBasicMaterial color="#7dd3fc" transparent opacity={hovered?0.75:0.22}/>
        </lineSegments>
      </mesh>
      {hovered&&(
        <Html position={[cx,cy+w*SCALE*0.5+0.05,cz]} center zIndexRange={[100,0]}>
          <div style={{background:"rgba(15,23,42,0.9)",border:"1px solid #38bdf8",borderRadius:6,
            padding:"6px 10px",color:"#e0f2fe",fontSize:11,fontFamily:"'DM Mono',monospace",
            whiteSpace:"nowrap",pointerEvents:"none"}}>
            <div style={{fontWeight:700,color:"#38bdf8",marginBottom:3}}>✂ 잔재 (Offcut)</div>
            <div>{l.toFixed(1)}×{w.toFixed(1)}×{t.toFixed(1)} mm</div>
          </div>
        </Html>
      )}
    </group>
  );
}

function CameraController({ response, orbitRef }) {
  const { camera } = useThree();
  useEffect(() => {
    if (!response?.placements?.length) return;
    let maxZ=0,maxX=0,maxY=0,zOff=0;
    (response.stock_summaries||[]).forEach((s)=>{
      maxZ=Math.max(maxZ,s.usable_dims.t*SCALE+zOff);
      maxX=Math.max(maxX,s.usable_dims.l*SCALE);
      maxY=Math.max(maxY,s.usable_dims.w*SCALE);
      zOff+=s.original_dims.t*SCALE+STOCK_GAP;
    });
    const cx=maxX/2+CAM_X_OFFSET, cy=maxY/2, cz=maxZ/2;
    const size=Math.max(maxX,maxY,maxZ);
    if (orbitRef.current) orbitRef.current.target.set(cx,cy,cz);
    camera.position.set(cx+size*1.2,cy+size*0.8,cz+size*1.5);
    camera.lookAt(cx,cy,cz);
  },[response]);
  return null;
}

function Scene({ response }) {
  const orbitRef = useRef();
  const stockZOffsets = useMemo(()=>computeStockZOffsets(response),[response]);
  const stockList = useMemo(()=>{
    if(!response) return [];
    const seen=new Set();
    return (response.stock_summaries||[]).filter((s)=>{
      if(seen.has(s.stock_id)) return false;
      seen.add(s.stock_id); return true;
    });
  },[response]);
  const stripBBoxes = useMemo(()=>{
    if(!response?.placements) return {};
    return computeStripBBoxes(response.placements, stockZOffsets);
  },[response,stockZOffsets]);

  return (
    <>
      <CameraController response={response} orbitRef={orbitRef}/>
      <OrbitControls ref={orbitRef} makeDefault/>
      <ambientLight intensity={0.6}/>
      <directionalLight position={[5,8,5]} intensity={1.0} castShadow/>
      <directionalLight position={[-3,-2,-3]} intensity={0.3}/>
      <gridHelper args={[20,40,"#1e293b","#1e293b"]} position={[0,-0.01,0]}/>
      {response&&stockList.map((s)=>(
        <StockOutline key={s.stock_id} summary={s} zOffset={stockZOffsets[s.stock_id]??0}/>
      ))}
      {response?.placements?.map((p)=>(
        <PlacedBox key={p.node_id} placement={p} zOffset={stockZOffsets[p.stock_id]??0}/>
      ))}
      {response&&Object.entries(stripBBoxes).map(([sid,bbox])=>(
        <StripBoundingBox key={sid} bbox={bbox}/>
      ))}
      {response?.offcuts?.map((o)=>(
        <OffcutBox key={o.node_id} offcut={o} zOffset={stockZOffsets[o.stock_id]??0}/>
      ))}
    </>
  );
}

function buildCutList(response) {
  const byStock={};
  (response.placements||[]).forEach((p)=>{
    if(!byStock[p.stock_id]) byStock[p.stock_id]=new Map();
    (p.cut_history||[]).forEach((cut)=>{
      if(!byStock[p.stock_id].has(cut.cut_id)) byStock[p.stock_id].set(cut.cut_id,cut);
    });
  });
  return Object.entries(byStock).map(([stockId,cutMap])=>({
    stockId,
    cuts:Array.from(cutMap.values()).map((c,i)=>({...c,step:i+1})),
  }));
}

function CutListModal({ response, onClose }) {
  const [activeStock,setActiveStock]=useState(0);
  const cutList=useMemo(()=>buildCutList(response),[response]);
  useEffect(()=>{
    const h=(e)=>{ if(e.key==="Escape") onClose(); };
    window.addEventListener("keydown",h);
    return ()=>window.removeEventListener("keydown",h);
  },[onClose]);
  const axisLabel=(ax)=>({X:"← L →",Y:"← W →",Z:"← T →"}[ax]||ax);
  const axisDesc=(cut)=>{
    const lb={X:"길이(L)축",Y:"너비(W)축",Z:"두께(T)축"}[cut.axis]||cut.axis;
    return `${lb} ${cut.position.toFixed(1)}mm 지점에서 Kerf ${cut.kerf}mm 관통 절단`;
  };
  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.75)",display:"flex",
      alignItems:"center",justifyContent:"center",zIndex:1000,backdropFilter:"blur(4px)"}}
      onClick={onClose}>
      <div style={{background:"#0f172a",border:"1px solid #1e3a5f",borderRadius:16,
        width:"min(90vw,760px)",maxHeight:"80vh",display:"flex",flexDirection:"column",overflow:"hidden"}}
        onClick={(e)=>e.stopPropagation()}>
        <div style={{padding:"20px 24px 16px",borderBottom:"1px solid #1e293b",
          display:"flex",alignItems:"center",justifyContent:"space-between"}}>
          <div>
            <div style={{color:"#fff",fontWeight:700,fontSize:18,fontFamily:"'DM Mono',monospace"}}>📋 작업 지시서</div>
            <div style={{color:"#64748b",fontSize:12,marginTop:2}}>원장별 순차 절단 지시 — 동일 cut_id = 1회 물리적 절단</div>
          </div>
          <button onClick={onClose} style={{background:"none",border:"none",color:"#64748b",fontSize:20,cursor:"pointer"}}>✕</button>
        </div>
        <div style={{display:"flex",gap:4,padding:"12px 24px 0",borderBottom:"1px solid #1e293b"}}>
          {cutList.map((s,i)=>(
            <button key={s.stockId} onClick={()=>setActiveStock(i)} style={{
              padding:"6px 14px",borderRadius:"6px 6px 0 0",border:"1px solid",
              borderColor:activeStock===i?"#3b82f6":"#1e293b",borderBottom:"none",
              background:activeStock===i?"#1e3a5f":"transparent",
              color:activeStock===i?"#60a5fa":"#64748b",
              fontSize:12,cursor:"pointer",fontFamily:"'DM Mono',monospace",
            }}>{s.stockId}</button>
          ))}
        </div>
        <div style={{overflowY:"auto",padding:"16px 24px 24px"}}>
          {cutList[activeStock]&&(
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
              <thead>
                <tr style={{color:"#475569"}}>
                  {["Step","축","위치(mm)","Kerf(mm)","작업 지시"].map((h)=>(
                    <th key={h} style={{padding:"8px 10px",textAlign:"left",
                      borderBottom:"1px solid #1e293b",fontFamily:"'DM Mono',monospace",fontWeight:500}}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cutList[activeStock].cuts.map((cut)=>(
                  <tr key={cut.cut_id} style={{borderBottom:"1px solid #0f172a"}}
                    onMouseEnter={(e)=>e.currentTarget.style.background="#1e293b"}
                    onMouseLeave={(e)=>e.currentTarget.style.background="transparent"}>
                    <td style={{padding:"10px",color:"#94a3b8",fontFamily:"'DM Mono',monospace"}}>{String(cut.step).padStart(2,"0")}</td>
                    <td style={{padding:"10px"}}>
                      <span style={{
                        background:{X:"#1e3a5f",Y:"#1a3320",Z:"#2d1b69"}[cut.axis],
                        color:{X:"#60a5fa",Y:"#4ade80",Z:"#a78bfa"}[cut.axis],
                        padding:"2px 8px",borderRadius:4,fontSize:11,fontFamily:"'DM Mono',monospace",
                      }}>{axisLabel(cut.axis)}</span>
                    </td>
                    <td style={{padding:"10px",color:"#e2e8f0",fontFamily:"'DM Mono',monospace"}}>{cut.position.toFixed(1)}</td>
                    <td style={{padding:"10px",color:"#f59e0b",fontFamily:"'DM Mono',monospace"}}>{cut.kerf}</td>
                    <td style={{padding:"10px",color:"#94a3b8",fontSize:12}}>{axisDesc(cut)}</td>
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

function ResultDashboard({ stats, failures }) {
  if (!stats) return null;
  const mono="'DM Mono',monospace";
  const stepTimes=stats.step_times||{};
  const steps=[
    {key:"step1_dp_sec",    label:"1. DP 그루핑"},
    {key:"step2_strip_sec", label:"2. Strip 생성"},
    {key:"step3_assign_sec",label:"3. 원장 배정"},
    {key:"step4_place_sec", label:"4. 3D 배치"},
  ];
  const M=({label,value,color,sub})=>(
    <div>
      <div style={{fontSize:9,color:"#475569",letterSpacing:1.5,fontFamily:mono,textTransform:"uppercase"}}>{label}</div>
      <div style={{fontSize:19,fontWeight:800,color,fontFamily:mono,lineHeight:1.15}}>{value}</div>
      {sub&&<div style={{fontSize:9,color:"#475569",fontFamily:mono,marginTop:1}}>{sub}</div>}
    </div>
  );
  const Bar=({pct,color})=>(
    <div style={{height:4,background:"#1e293b",borderRadius:2,overflow:"hidden",marginTop:4}}>
      <div style={{height:"100%",width:`${Math.min(100,pct||0)}%`,background:color,borderRadius:2,transition:"width 0.6s ease"}}/>
    </div>
  );
  const card={background:"#0f172a",border:"1px solid #1e293b",borderRadius:10,padding:"10px 14px",marginBottom:8};
  return (
    <div style={{marginBottom:16}}>
      <div style={{fontSize:10,fontWeight:700,color:"#475569",letterSpacing:2,
        textTransform:"uppercase",fontFamily:mono,marginBottom:8}}>◎ 결과</div>

      <div style={card}>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
          <M label="사용 효율" value={`${stats.overall_efficiency_pct??0}%`} color="#4ade80" sub="(사용 원장 기준)"/>
          <M label="전체 수율" value={`${stats.yield_rate_pct??0}%`}         color="#34d399" sub="(투입 원장 기준)"/>
          <M label="배치"      value={`${stats.total_placed}개`}             color="#60a5fa"/>
          <M label="원장 사용" value={`${stats.stocks_used}장`}              color="#a78bfa"/>
        </div>
        <div style={{marginTop:10}}>
          <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#475569",fontFamily:mono,marginBottom:2}}>
            <span>사용효율 {stats.overall_efficiency_pct??0}%</span>
            <span>수율 {stats.yield_rate_pct??0}%</span>
          </div>
          <Bar pct={stats.overall_efficiency_pct??0} color="#4ade80"/>
          <Bar pct={stats.yield_rate_pct??0} color="#34d399"/>
        </div>
      </div>

      <div style={card}>
        <div style={{fontSize:9,color:STRIP_BBOX_COLOR,letterSpacing:1.5,fontFamily:mono,marginBottom:8,textTransform:"uppercase"}}>
          ✦ Strip 엔진 통계
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
          <M label="그룹 수"   value={`${stats.n_groups??0}개`}               color="#fbbf24"/>
          <M label="Strip 수"  value={`${stats.n_strips??0}개`}               color="#f59e0b"/>
          <M label="배정율"    value={`${stats.strip_assignment_rate??0}%`}   color="#fb923c"/>
          <M label="Fallback" value={`${stats.fallback_placed??0}개`}        color="#94a3b8"/>
        </div>
        <Bar pct={stats.strip_assignment_rate??0} color={STRIP_BBOX_COLOR}/>
      </div>

      <div style={card}>
        <div style={{fontSize:9,color:"#64748b",letterSpacing:1.5,fontFamily:mono,marginBottom:8,textTransform:"uppercase"}}>
          ⏱ 단계별 연산 시간
        </div>
        {steps.map(({key,label})=>{
          const sec=stepTimes[key]??0;
          const ms=(sec*1000).toFixed(1);
          const pct=Math.min(100,(sec/(stats.processing_time_sec||1))*100);
          return (
            <div key={key} style={{marginBottom:6}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:mono}}>
                <span style={{color:"#94a3b8"}}>{label}</span>
                <span style={{color:"#e2e8f0"}}>{ms}ms</span>
              </div>
              <div style={{height:3,background:"#1e293b",borderRadius:2,overflow:"hidden",marginTop:2}}>
                <div style={{height:"100%",width:`${pct}%`,background:"#6366f1",borderRadius:2}}/>
              </div>
            </div>
          );
        })}
        <div style={{display:"flex",justifyContent:"space-between",fontSize:10,fontFamily:mono,
          borderTop:"1px solid #1e293b",paddingTop:6,marginTop:4}}>
          <span style={{color:"#64748b"}}>총 연산</span>
          <span style={{color:"#e2e8f0"}}>{(stats.processing_time_sec*1000).toFixed(0)}ms</span>
        </div>
      </div>

      {failures?.length>0&&(
        <div style={{background:"#1c0a0a",border:"1px solid #7f1d1d",borderRadius:8,padding:"10px 12px"}}>
          {failures.map((f,i)=>(
            <div key={i} style={{color:"#f87171",fontSize:11,lineHeight:1.5}}>⚠ {f}</div>
          ))}
        </div>
      )}
    </div>
  );
}

function NumberInput({ value, onChange, min=0, label, unit="mm" }) {
  return (
    <div style={{display:"flex",flexDirection:"column",gap:3}}>
      {label&&<label style={{fontSize:10,color:"#64748b",fontFamily:"'DM Mono',monospace",letterSpacing:1}}>{label}</label>}
      <div style={{position:"relative"}}>
        <input type="number" value={value} min={min} onChange={(e)=>onChange(Number(e.target.value))}
          style={{width:"100%",boxSizing:"border-box",background:"#0f172a",border:"1px solid #1e293b",
            borderRadius:6,color:"#e2e8f0",padding:"7px 28px 7px 10px",fontSize:13,
            fontFamily:"'DM Mono',monospace",outline:"none"}}
          onFocus={(e)=>e.target.style.borderColor="#3b82f6"}
          onBlur={(e)=>e.target.style.borderColor="#1e293b"}/>
        <span style={{position:"absolute",right:8,top:"50%",transform:"translateY(-50%)",fontSize:10,color:"#475569"}}>{unit}</span>
      </div>
    </div>
  );
}

function CheckBox({ checked, onChange, label }) {
  return (
    <label style={{display:"flex",alignItems:"center",gap:6,cursor:"pointer",fontSize:12,color:"#94a3b8"}}>
      <div onClick={()=>onChange(!checked)} style={{width:16,height:16,borderRadius:4,
        border:`2px solid ${checked?"#3b82f6":"#334155"}`,background:checked?"#3b82f6":"transparent",
        display:"flex",alignItems:"center",justifyContent:"center",flexShrink:0}}>
        {checked&&<span style={{color:"#fff",fontSize:10,fontWeight:900}}>✓</span>}
      </div>
      {label}
    </label>
  );
}

export default function App() {
  const [settings,  setSettings]  = useState(DEFAULT_SETTINGS);
  const [stocks,    setStocks]    = useState(DEFAULT_STOCKS);
  const [parts,     setParts]     = useState(DEFAULT_PARTS);
  const [response,  setResponse]  = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const [showModal, setShowModal] = useState(false);

  const requestBody = useMemo(()=>({
    settings:{ kerf:settings.kerf,trimming:settings.trimming,optimization_goal:"MINIMIZE_WASTE" },
    stocks: stocks.map(({_uid,...s})=>s),
    parts:  parts.map(({_uid,...p})=>p),
  }),[settings,stocks,parts]);

  const handleOptimize = async () => {
    setLoading(true); setError(null);
    try {
      const res = await fetch(`${API_URL}/optimize`,{
        method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(requestBody),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail?.detail||data.detail||JSON.stringify(data));
      setResponse(data);
    } catch(e) { setError(e.message); } finally { setLoading(false); }
  };

  const addStock   = ()=>setStocks((p)=>[...p,{_uid:uid(),id:`S${p.length+1}`,l:0,w:0,t:0,qty:0}]);
  const updateStock = (i,k,v)=>setStocks((p)=>p.map((s,j)=>j===i?{...s,[k]:v}:s));
  const removeStock = (i)=>setStocks((p)=>p.filter((_,j)=>j!==i));

  const addPart = ()=>setParts((p)=>[...p,{_uid:uid(),id:`P${p.length+1}`,l:0,w:0,t:0,qty:0,
    lock_z:false,allow_xy_rotation:true,priority:0,color:PART_COLORS[p.length%PART_COLORS.length]}]);
  const updatePart = (i,k,v)=>setParts((p)=>p.map((s,j)=>j===i?{...s,[k]:v}:s));
  const removePart  = (i)=>setParts((p)=>p.filter((_,j)=>j!==i));

  const mono="'DM Mono',monospace";
  const sidebar={width:292,minWidth:292,height:"100vh",background:"#060d1a",
    borderRight:"1px solid #1e293b",display:"flex",flexDirection:"column",overflow:"hidden"};
  const sLbl={fontSize:10,fontWeight:700,color:"#475569",letterSpacing:2,
    textTransform:"uppercase",fontFamily:mono,marginBottom:8};
  const card={background:"#0f172a",border:"1px solid #1e293b",borderRadius:10,padding:"12px 14px",marginBottom:8};
  const addBtn={width:"100%",padding:"8px",background:"transparent",border:"1px dashed #1e3a5f",
    borderRadius:8,color:"#3b82f6",fontSize:12,cursor:"pointer",fontFamily:mono,marginBottom:8};
  const rmBtn={background:"none",border:"none",color:"#475569",cursor:"pointer",fontSize:14,padding:"0 2px",lineHeight:1};
  const hasStrips = (response?.stats?.n_strips||0) > 0;

  return (
    <div style={{display:"flex",height:"100vh",background:"#060d1a",fontFamily:"system-ui,sans-serif"}}>
      <div style={sidebar}>
        <div style={{padding:"18px 20px 14px",borderBottom:"1px solid #1e293b"}}>
          <div style={{fontSize:15,fontWeight:800,color:"#e2e8f0",fontFamily:mono,letterSpacing:-0.5}}>
            ✦ CUT OPTIMIZER
          </div>
          <div style={{fontSize:10,color:"#475569",marginTop:2,letterSpacing:1}}>3D GUILLOTINE ENGINE v4.0</div>
        </div>

        <div style={{flex:1,overflowY:"auto",padding:"16px 14px",scrollbarWidth:"thin",scrollbarColor:"#1e293b transparent"}}>

          <div style={{marginBottom:20}}>
            <div style={sLbl}>⚙ Settings</div>
            <div style={card}>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:10}}>
                <NumberInput label="KERF" value={settings.kerf} onChange={(v)=>setSettings((s)=>({...s,kerf:v}))}/>
                <div/>
              </div>
              <div style={{fontSize:10,color:"#475569",marginBottom:6,letterSpacing:1}}>TRIMMING (양단 합산)</div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6}}>
                {[{ax:"z",lb:"T-T"},{ax:"y",lb:"T-W"},{ax:"x",lb:"T-L"}].map(({ax,lb})=>(
                  <NumberInput key={ax} label={lb} value={settings.trimming[ax]}
                    onChange={(v)=>setSettings((s)=>({...s,trimming:{...s.trimming,[ax]:v}}))}/>
                ))}
              </div>
            </div>
          </div>

          <div style={{marginBottom:20}}>
            <div style={sLbl}>▣ 원장 (Stocks)</div>
            {stocks.map((s,i)=>(
              <div key={s._uid} style={card}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                  <input value={s.id} onChange={(e)=>updateStock(i,"id",e.target.value)}
                    style={{background:"transparent",border:"none",color:"#60a5fa",fontSize:13,fontWeight:700,fontFamily:mono,width:80}}/>
                  <button onClick={()=>removeStock(i)} style={rmBtn}>✕</button>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginBottom:6}}>
                  <NumberInput label="T (두께)" value={s.t} onChange={(v)=>updateStock(i,"t",v)}/>
                  <NumberInput label="W (폭)"   value={s.w} onChange={(v)=>updateStock(i,"w",v)}/>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
                  <NumberInput label="L (길이)" value={s.l} onChange={(v)=>updateStock(i,"l",v)}/>
                  <NumberInput label="QTY" value={s.qty} min={1} onChange={(v)=>updateStock(i,"qty",v)} unit="장"/>
                </div>
              </div>
            ))}
            <button onClick={addStock} style={addBtn}>+ 원장 추가</button>
          </div>

          <div style={{marginBottom:20}}>
            <div style={sLbl}>◈ 부품 (Parts)</div>
            {parts.map((p,i)=>(
              <div key={p._uid} style={{...card,borderLeft:`3px solid ${p.color}`}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
                  <div style={{display:"flex",alignItems:"center",gap:8}}>
                    <div style={{width:10,height:10,borderRadius:"50%",background:p.color,flexShrink:0}}/>
                    <input value={p.id} onChange={(e)=>updatePart(i,"id",e.target.value)}
                      style={{background:"transparent",border:"none",color:"#e2e8f0",fontSize:13,fontWeight:700,fontFamily:mono,width:70}}/>
                  </div>
                  <button onClick={()=>removePart(i)} style={rmBtn}>✕</button>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginBottom:6}}>
                  <NumberInput label="T (두께)" value={p.t} onChange={(v)=>updatePart(i,"t",v)}/>
                  <NumberInput label="W (폭)"   value={p.w} onChange={(v)=>updatePart(i,"w",v)}/>
                </div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6,marginBottom:8}}>
                  <NumberInput label="L (길이)" value={p.l} onChange={(v)=>updatePart(i,"l",v)}/>
                  <NumberInput label="QTY" value={p.qty} min={1} onChange={(v)=>updatePart(i,"qty",v)} unit="개"/>
                </div>
                <div style={{display:"flex",flexDirection:"column",gap:5}}>
                  <CheckBox checked={p.lock_z}            onChange={(v)=>updatePart(i,"lock_z",v)}            label="두께(Z) 고정"/>
                  <CheckBox checked={p.allow_xy_rotation} onChange={(v)=>updatePart(i,"allow_xy_rotation",v)} label="L↔W 회전 허용"/>
                </div>
              </div>
            ))}
            <button onClick={addPart} style={addBtn}>+ 부품 추가</button>
          </div>

          <ResultDashboard stats={response?.stats} failures={response?.failures}/>
        </div>

        <div style={{padding:"12px 14px",borderTop:"1px solid #1e293b",display:"flex",flexDirection:"column",gap:8}}>
          {response&&(
            <button onClick={()=>setShowModal(true)} style={{width:"100%",padding:"10px",background:"transparent",
              border:"1px solid #1e3a5f",borderRadius:10,color:"#60a5fa",fontSize:13,fontWeight:600,cursor:"pointer",fontFamily:mono}}>
              📋 작업 지시서 보기
            </button>
          )}
          <button onClick={handleOptimize} disabled={loading} style={{
            width:"100%",padding:"14px",
            background:loading?"#1e293b":"linear-gradient(135deg,#1d4ed8,#3b82f6)",
            border:"none",borderRadius:10,color:loading?"#64748b":"#fff",fontSize:14,fontWeight:800,
            cursor:loading?"not-allowed":"pointer",fontFamily:mono,letterSpacing:1,transition:"all 0.2s",
            boxShadow:loading?"none":"0 4px 20px rgba(59,130,246,0.3)",
          }}>
            {loading?"계산 중...":"▶ OPTIMIZE"}
          </button>
          {error&&(
            <div style={{background:"#1c0a0a",border:"1px solid #7f1d1d",borderRadius:8,padding:"10px 12px",
              color:"#f87171",fontSize:11,lineHeight:1.6,wordBreak:"break-word"}}>⚠ {error}</div>
          )}
        </div>
      </div>

      <div style={{flex:1,position:"relative",background:"#030712"}}>
        <Canvas camera={{fov:45,near:0.01,far:1000,position:[2,1.5,3]}} style={{width:"100%",height:"100%"}}>
          <Scene response={response}/>
        </Canvas>

        {!response&&!loading&&(
          <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",
            alignItems:"center",justifyContent:"center",pointerEvents:"none"}}>
            <div style={{fontSize:48,marginBottom:16,opacity:0.3}}>◈</div>
            <div style={{color:"#334155",fontSize:14,fontFamily:mono}}>원장과 부품을 입력하고 OPTIMIZE를 눌러주세요</div>
          </div>
        )}

        {loading&&(
          <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",
            alignItems:"center",justifyContent:"center",background:"rgba(3,7,18,0.7)",backdropFilter:"blur(4px)"}}>
            <div style={{width:40,height:40,border:"3px solid #1e293b",
              borderTop:"3px solid #3b82f6",borderRadius:"50%",animation:"spin 0.8s linear infinite"}}/>
            <div style={{color:"#64748b",marginTop:16,fontFamily:mono,fontSize:12}}>Phase 4.0 최적화 계산 중...</div>
          </div>
        )}

        <div style={{position:"absolute",bottom:16,left:16,color:"#1e293b",
          fontSize:11,fontFamily:mono,lineHeight:1.8,pointerEvents:"none"}}>
          <div>🖱 드래그: 회전 &nbsp; 우클릭: 이동 &nbsp; 휠: 줌</div>
        </div>

        {response&&(
          <div style={{position:"absolute",bottom:16,right:16,background:"rgba(6,13,26,0.92)",
            border:"1px solid #1e293b",borderRadius:10,padding:"10px 14px",maxWidth:210}}>
            <div style={{fontSize:10,color:"#475569",letterSpacing:1,marginBottom:6,fontFamily:mono}}>범례</div>
            {parts.map((p)=>{
              const placed=(response.placements||[]).filter((pl)=>pl.part_id===p.id).length;
              if (!placed) return null;
              return (
                <div key={p.id} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                  <div style={{width:10,height:10,borderRadius:2,background:p.color,flexShrink:0}}/>
                  <span style={{fontSize:11,color:"#94a3b8",fontFamily:mono}}>{p.id} ({placed}개)</span>
                </div>
              );
            })}
            <div style={{borderTop:"1px solid #1e293b",marginTop:6,paddingTop:6}}>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                <div style={{width:18,height:2,background:"#3b82f6"}}/>
                <span style={{fontSize:11,color:"#3b82f6",fontFamily:mono}}>원장 경계</span>
              </div>
              {hasStrips&&(
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                  <div style={{width:18,height:2,background:STRIP_BBOX_COLOR}}/>
                  <span style={{fontSize:11,color:STRIP_BBOX_COLOR,fontFamily:mono}}>
                    Strip 그룹 ({response.stats.n_strips}개)
                  </span>
                </div>
              )}
              <div style={{display:"flex",alignItems:"center",gap:8}}>
                <div style={{width:18,height:10,background:"rgba(56,189,248,0.12)",
                  border:"1px solid rgba(125,211,252,0.4)",borderRadius:2}}/>
                <span style={{fontSize:11,color:"#7dd3fc",fontFamily:mono}}>잔재 (Offcut)</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {showModal&&response&&(
        <CutListModal response={response} onClose={()=>setShowModal(false)}/>
      )}

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
