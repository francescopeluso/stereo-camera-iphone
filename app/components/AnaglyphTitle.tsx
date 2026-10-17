"use client";

import { useEffect, useRef } from "react";
import styles from "./AnaglyphTitle.module.css";

export default function AnaglyphTitle({ text }: { text: string }) {
  const ref = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    const el = ref.current!;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const scope = el.closest("header") ?? document.body;
    const move = (e: PointerEvent) => {
      const r = scope.getBoundingClientRect();
      const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
      el.style.setProperty("--d", `${(1 - x) * 0.09 + 0.012}em`);
    };
    const leave = () => el.style.removeProperty("--d");
    scope.addEventListener("pointermove", move as EventListener);
    scope.addEventListener("pointerleave", leave);
    return () => {
      scope.removeEventListener("pointermove", move as EventListener);
      scope.removeEventListener("pointerleave", leave);
    };
  }, []);

  return (
    <h1 ref={ref} className={styles.title}>
      <span className="sr-only">{text}</span>
      <span aria-hidden="true" className={styles.left}>
        {text}
      </span>
      <span aria-hidden="true" className={styles.right}>
        {text}
      </span>
    </h1>
  );
}
