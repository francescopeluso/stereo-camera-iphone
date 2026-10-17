"use client";

import { useEffect, useId, useRef, useState } from "react";
import styles from "./Charts.module.css";

const FB = 470 * 0.01;
const STEPS = [0.1, 0.2, 0.5];
const H = 320,
  L = 56,
  R = 16,
  T = 16,
  B = 40;
const ZMAX = 4;

const theory = (z: number, dd: number) => (z * z * dd) / FB;
const fmt = (m: number) => (m < 0.995 ? `${(m * 100).toFixed(m < 0.1 ? 1 : 0)} cm` : `${m.toFixed(2)} m`);

export default function ErrorExplorer() {
  const [z, setZ] = useState(1);
  const [dd, setDd] = useState(0.2);
  const [hoverZ, setHoverZ] = useState<number | null>(null);
  const id = useId();
  const plot = useRef<HTMLDivElement>(null);
  const [W, setW] = useState(640);

  useEffect(() => {
    const el = plot.current!;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const yMax = Math.ceil((theory(ZMAX, dd) * 100) / 20) * 20;
  const sx = (v: number) => L + (v / ZMAX) * (W - L - R);
  const sy = (cm: number) => H - B - (cm / yMax) * (H - T - B);
  const path = Array.from({ length: 81 }, (_, i) => {
    const v = (i / 80) * ZMAX;
    return `${i ? "L" : "M"}${sx(v).toFixed(1)},${sy(theory(v, dd) * 100).toFixed(1)}`;
  }).join("");
  const ticksY = Array.from({ length: 5 }, (_, i) => (yMax / 4) * i);
  const zh = hoverZ ?? z;

  return (
    <div className={styles.explorer}>
      <div className={styles.readout} aria-live="polite">
        <p className={styles.figureNumber}>± {fmt(theory(z, dd))}</p>
        <p className={styles.figureLabel}>
          expected depth error at {z.toFixed(2)} m with {dd} px matching accuracy
        </p>
      </div>

      <div className={styles.controls}>
        <label htmlFor={`${id}-z`}>Distance</label>
        <input id={`${id}-z`} type="range" min={0.2} max={ZMAX} step={0.05} value={z} onChange={(e) => setZ(Number(e.target.value))} />
        <output htmlFor={`${id}-z`}>{z.toFixed(2)} m</output>
        <span id={`${id}-dd`}>Sub-pixel accuracy</span>
        <div className={styles.segmented} role="radiogroup" aria-labelledby={`${id}-dd`}>
          {STEPS.map((s) => (
            <button key={s} type="button" role="radio" aria-checked={dd === s} onClick={() => setDd(s)}>
              {s} px
            </button>
          ))}
        </div>
      </div>

      <figure className={styles.figure}>
        <figcaption className={styles.title}>
          Expected depth error by distance
          <span className={styles.subtitle}>ΔZ ≈ Z²·Δd / (f·B), with f·B ≈ 4.7 px·m on iPhone 16</span>
        </figcaption>
        <div className={styles.plot} ref={plot}>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            width={W}
            height={H}
            role="img"
            aria-label={`Line chart: expected depth error grows from 0 at the camera to ${fmt(theory(1, dd))} at 1 meter, ${fmt(theory(2, dd))} at 2 meters and ${fmt(theory(ZMAX, dd))} at 4 meters.`}
            onPointerMove={(e) => {
              const r = e.currentTarget.getBoundingClientRect();
              const v = ((((e.clientX - r.left) / r.width) * W - L) / (W - L - R)) * ZMAX;
              setHoverZ(v >= 0 && v <= ZMAX ? v : null);
            }}
            onPointerLeave={() => setHoverZ(null)}
          >
            {ticksY.map((t) => (
              <g key={t}>
                <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} className={styles.grid} />
                <text x={L - 8} y={sy(t) + 4} textAnchor="end" className={styles.tick}>
                  {t} cm
                </text>
              </g>
            ))}
            {[0, 1, 2, 3, 4].map((t) => (
              <text key={t} x={sx(t)} y={H - B + 22} textAnchor="middle" className={styles.tick}>
                {t} m
              </text>
            ))}
            <path d={path} fill="none" stroke="var(--correct)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
            <line x1={sx(zh)} x2={sx(zh)} y1={T} y2={H - B} className={styles.cross} />
            <circle cx={sx(zh)} cy={sy(theory(zh, dd) * 100)} r={5} fill="var(--correct)" stroke="var(--paper)" strokeWidth={2} />
          </svg>
          {hoverZ !== null && (
            <div className={styles.svgTip} style={{ left: `${(sx(hoverZ) / W) * 100}%` }}>
              <strong>{hoverZ.toFixed(2)} m</strong>
              <span>± {fmt(theory(hoverZ, dd))}</span>
            </div>
          )}
        </div>
      </figure>
    </div>
  );
}
