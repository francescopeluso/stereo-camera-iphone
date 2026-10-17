"use client";

import { Box, Camera, Columns2, Contrast, Flame, Glasses, Grid3x3, Layers, Radar, type LucideIcon } from "lucide-react";
import { useState } from "react";
import scene from "@/public/scene.json";
import { asset } from "@/app/lib/asset";
import styles from "./ModeViewer.module.css";

type Overlay = "objects" | "cloud" | "none";

const MODES: { id: string; name: string; icon: LucideIcon; overlay: Overlay; text: string }[] = [
  {
    id: "standard",
    name: "Standard",
    icon: Camera,
    overlay: "objects",
    text: "The camera image with object masks, boxes, labels, distance and approximate size.",
  },
  {
    id: "depth",
    name: "Depth map",
    icon: Flame,
    overlay: "objects",
    text: "Distance as color: red is near, blue is far. Black pixels had no reliable match.",
  },
  {
    id: "overlay",
    name: "Overlay",
    icon: Layers,
    overlay: "objects",
    text: "Camera image and depth blended, so you can tell which object each color belongs to.",
  },
  {
    id: "disparity",
    name: "Disparity",
    icon: Contrast,
    overlay: "none",
    text: "The raw stereo signal in grayscale. Brighter means a bigger shift between the two views, so closer.",
  },
  {
    id: "pair",
    name: "Rectified pair",
    icon: Columns2,
    overlay: "none",
    text: "Both views after rectification. A point on a green line in one view sits on the same line in the other.",
  },
  {
    id: "anaglyph",
    name: "Anaglyph",
    icon: Glasses,
    overlay: "none",
    text: "A red/cyan image for 3D glasses. With a 1 cm baseline the effect is subtle, but it's there.",
  },
  {
    id: "segmentation",
    name: "3D segmentation",
    icon: Grid3x3,
    overlay: "none",
    text: "Pixels grouped into surfaces by depth alone. Objects resting on the desk merge with it; the raised laptop screen stands apart.",
  },
  {
    id: "radar",
    name: "Bird's-eye",
    icon: Radar,
    overlay: "none",
    text: "A top-down map with the field of view, distance rings, the nearest obstacle in each direction and the recognized objects.",
  },
  {
    id: "pointcloud",
    name: "Point cloud",
    icon: Box,
    overlay: "cloud",
    text: "Every valid pixel placed in 3D. In the app you drag to rotate and pinch to zoom.",
  },
];

const color = (hue: number, a = 1) => `hsl(${Math.round(hue * 360)} 80% 52% / ${a})`;

export default function ModeViewer() {
  const [index, setIndex] = useState(0);
  const mode = MODES[index];

  return (
    <div className={styles.viewer}>
      <div className={styles.phone}>
        <div className={styles.island} aria-hidden="true" />
        <div className={styles.screen}>
          <div className={styles.stage}>
            <div className={styles.image}>
              {MODES.map((m, i) => (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  key={m.id}
                  src={asset(`/img/mode-${m.id}.webp`)}
                  alt={i === index ? `${m.name} view of the desk scene` : ""}
                  aria-hidden={i !== index}
                  className={styles.frame}
                  data-active={i === index}
                  loading={i === 0 ? "eager" : "lazy"}
                  width={1280}
                  height={720}
                />
              ))}
              {mode.overlay === "objects" && <Objects />}
              {mode.overlay === "cloud" && <p className={styles.hint}>drag: rotate · pinch: zoom · tap: reset</p>}
              <div className={styles.hud} aria-hidden="true">
                <strong>{mode.name}</strong>
                <span>B 10.0 mm · f 470 px</span>
                <span>SGM-4 · D 64 · LR</span>
              </div>
            </div>
          </div>
          <div className={styles.panel}>
            {MODES.map((m, i) => (
              <button
                key={m.id}
                type="button"
                tabIndex={-1}
                aria-hidden="true"
                className={styles.icon}
                data-active={i === index}
                onClick={() => setIndex(i)}
              >
                <m.icon strokeWidth={2.2} />
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className={styles.controls}>
        <div className={styles.picker} role="radiogroup" aria-label="View mode">
          {MODES.map((m, i) => (
            <button
              key={m.id}
              type="button"
              role="radio"
              aria-checked={i === index}
              tabIndex={i === index ? 0 : -1}
              className={styles.option}
              onClick={() => setIndex(i)}
              onKeyDown={(e) => {
                const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
                if (!d) return;
                e.preventDefault();
                const next = (index + d + MODES.length) % MODES.length;
                setIndex(next);
                (e.currentTarget.parentElement?.children[next] as HTMLElement | undefined)?.focus();
              }}
            >
              <m.icon aria-hidden="true" strokeWidth={2} />
              {m.name}
            </button>
          ))}
        </div>
        <p className={styles.caption} aria-live="polite">
          <strong>{mode.name}.</strong> {mode.text}
        </p>
      </div>
    </div>
  );
}

function Objects() {
  return (
    <div className={styles.objects} aria-hidden="true">
      {scene.objects.map((o, i) => {
        const [x, y, w, h] = o.bbox;
        const c = color(o.hue);
        return (
          <div
            key={o.label}
            className={styles.box}
            style={{ left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%`, borderColor: c }}
          >
            <span className={styles.pill} style={{ background: color(o.hue, 0.88) }}>
              #{i + 1} {o.label}
              {o.distance ? ` · ${o.distance.toFixed(2)} m` : ""}
            </span>
            {o.width && o.height ? (
              <span className={styles.dims}>
                {Math.round(o.width * 100)}×{Math.round(o.height * 100)} cm
              </span>
            ) : null}
          </div>
        );
      })}
      <div className={styles.cross}>
        <span className={styles.crossLabel}>{scene.centerDistance.toFixed(2)} m</span>
      </div>
    </div>
  );
}
