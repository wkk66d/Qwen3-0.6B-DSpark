/* Shared helpers for the Lossy Draft dashboards: formatting, an EMA smoother,
   and two hand-rolled SVG charts. No CDN, no build step. */

const fmt = (v,d=3) => (v===null||v===undefined||Number.isNaN(v)) ? "–" : (+v).toFixed(d);
const pct = v => (v===null||v===undefined) ? "–" : (100*v).toFixed(1)+"%";

/* Exponential moving average over plotted points; span 1 returns the raw series. */
function ema(pts, span){
  if(!pts || !pts.length) return [];
  if(span <= 1) return pts.slice();
  const a = 2/(span+1);
  let s = null;
  return pts.map(p => { s = (s===null) ? p.y : a*p.y + (1-a)*s; return {x:p.x, y:s}; });
}

/* ---------- tiny SVG line chart ---------- */
function lineChart(series, {w=560,h=210,pad=38,yfmt=fmt,xlabel="epoch",hlines=[]}={}){
  const pts = series.flatMap(s=>s.points||[]);
  if(!pts.length) return `<div class="empty">no data yet</div>`;
  const xs=pts.map(p=>p.x), ys=pts.map(p=>p.y);
  let x0=Math.min(...xs), x1=Math.max(...xs);
  let y0=Math.min(...ys), y1=Math.max(...ys);
  if(x0===x1){x0-=0.5;x1+=0.5}
  const span=(y1-y0)||Math.abs(y1)||1; y0-=span*0.12; y1+=span*0.12;
  const X=v=>pad+(v-x0)/(x1-x0)*(w-pad-14);
  const Y=v=>h-pad-(v-y0)/(y1-y0)*(h-pad-14);
  const ticks=n=>Array.from({length:n},(_,i)=>y0+(y1-y0)*i/(n-1));
  let g="";
  ticks(5).forEach(t=>{ g+=`<line x1="${pad}" x2="${w-14}" y1="${Y(t)}" y2="${Y(t)}"
      stroke="var(--line)" stroke-width="1"/>
      <text x="${pad-7}" y="${Y(t)+4}" text-anchor="end" fill="var(--dim)"
            font-size="10.5">${yfmt(t)}</text>`; });
  hlines.forEach(hl=>{ if(hl.y>=y0&&hl.y<=y1) g+=`<line x1="${pad}" x2="${w-14}"
      y1="${Y(hl.y)}" y2="${Y(hl.y)}" stroke="${hl.color||"var(--warn)"}"
      stroke-width="1" stroke-dasharray="4 3"/>
      <text x="${w-16}" y="${Y(hl.y)-4}" text-anchor="end" fill="${hl.color||"var(--warn)"}"
            font-size="10.5">${hl.label||""}</text>`; });
  const xset=[...new Set(xs)].sort((a,b)=>a-b);
  xset.forEach(x=>{ g+=`<text x="${X(x)}" y="${h-pad+16}" text-anchor="middle"
      fill="var(--dim)" font-size="10.5">${x}</text>`; });
  const paths=series.map(s=>{
    const d=(s.points||[]).map((p,i)=>(i?"L":"M")+X(p.x)+" "+Y(p.y)).join(" ");
    const dots=(s.dots===false)?"":(s.points||[]).map(p=>`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3"
        fill="${s.color}"><title>${s.name} · ${xlabel} ${p.x} · ${yfmt(p.y)}</title></circle>`).join("");
    return `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width||2}"
             stroke-linejoin="round" stroke-linecap="round"
             opacity="${s.opacity==null?1:s.opacity}"/>${dots}`;
  }).join("");
  const leg=series.map(s=>`<span><i style="background:${s.color};opacity:${s.opacity==null?1:s.opacity}"></i>${s.name}</span>`).join("");
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">
      <line x1="${pad}" x2="${pad}" y1="14" y2="${h-pad}" stroke="var(--line)"/>
      <line x1="${pad}" x2="${w-14}" y1="${h-pad}" y2="${h-pad}" stroke="var(--line)"/>
      ${g}${paths}
      <text x="${(w+pad)/2}" y="${h-4}" text-anchor="middle" fill="var(--dim)"
            font-size="10.5">${xlabel}</text></svg>
    <div class="legend">${leg}</div>`;
}

function barChart(vals, xlabel){
  if(!vals.length) return `<div class="empty">no data</div>`;
  const w=560,h=180,pad=38, mx=Math.max(...vals)||1;
  const bw=(w-pad-14)/vals.length;
  let g="";
  for(let i=0;i<=4;i++){ const y=h-pad-(h-pad-14)*i/4;
    g+=`<line x1="${pad}" x2="${w-14}" y1="${y}" y2="${y}" stroke="var(--line)"/>
        <text x="${pad-7}" y="${y+4}" text-anchor="end" fill="var(--dim)"
              font-size="10.5">${(mx*i/4).toFixed(2)}</text>`; }
  vals.forEach((v,i)=>{
    const bh=(h-pad-14)*(v/mx), x=pad+i*bw+bw*0.15, y=h-pad-bh;
    g+=`<rect x="${x}" y="${y}" width="${bw*0.7}" height="${bh}" rx="2"
          fill="#4c9be8"><title>${xlabel} ${i}: ${v.toFixed(3)}</title></rect>
        <text x="${pad+i*bw+bw/2}" y="${h-pad+16}" text-anchor="middle"
              fill="var(--dim)" font-size="10.5">${i}</text>`; });
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet">${g}</svg>
    <div class="legend"><span>${xlabel}</span></div>`;
}

/* Fetch the shared payload. Both pages read the same data.json; each renders
   only its own section, so there is one collector and one source of truth. */
async function fetchData(){
  const r = await fetch("data.json", {cache:"no-store"});
  return await r.json();
}

function stamp(ts){ const el=document.getElementById("ts"); if(el) el.textContent = ts || ""; }

/* Compact duration: "45 s", "12 min", "3 h 20 m". */
function fmtDur(s){
  if(s==null || !isFinite(s)) return "–";
  if(s < 90) return Math.round(s)+" s";
  const m = Math.round(s/60);
  if(m < 90) return m+" min";
  return Math.floor(m/60)+" h "+String(m%60).padStart(2,"0")+" m";
}
