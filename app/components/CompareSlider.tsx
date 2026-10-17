"use client";

import { useState } from "react";
import { asset } from "@/app/lib/asset";
import styles from "./CompareSlider.module.css";

export default function CompareSlider() {
  const [split, setSplit] = useState(50);

  return (
    <figure className={styles.figure}>
      <div className={styles.frame} style={{ "--split": `${split}%` } as React.CSSProperties}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={asset("/img/compare-sgm.webp")}
          alt="Depth from semi-global matching: smooth surfaces with few holes"
          width={1280}
          height={720}
          loading="lazy"
        />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          className={styles.top}
          src={asset("/img/compare-block.webp")}
          alt="Depth from block matching: the same scene with many black holes and speckles"
          width={1280}
          height={720}
          loading="lazy"
        />
        <div className={styles.divider} aria-hidden="true" />
        <span className={`${styles.tag} ${styles.tagLeft}`}>Block matching</span>
        <span className={`${styles.tag} ${styles.tagRight}`}>SGM</span>
        <input
          className={styles.range}
          type="range"
          min={0}
          max={100}
          value={split}
          onChange={(e) => setSplit(Number(e.target.value))}
          aria-label="Compare block matching and SGM: move left to show more SGM"
        />
      </div>
      <figcaption className={styles.caption}>
        Same frame, same census costs. Drag the divider: block matching on the left, semi-global matching on the right.
      </figcaption>
    </figure>
  );
}
