"use client";

import { useEffect, useRef, useState } from "react";
import { asset } from "@/app/lib/asset";
import styles from "./PointCloud.module.css";

const VS = `#version 300 es
in vec3 aPos;
in vec3 aColor;
uniform mat4 uMvp;
uniform float uSize;
uniform float uMode;
out vec3 vColor;
vec3 turbo(float x) {
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(
    dot(v4, vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234)) + dot(v2, vec2(-152.94239396, 59.28637943)),
    dot(v4, vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333)) + dot(v2, vec2(4.27729857, 2.82956604)),
    dot(v4, vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771)) + dot(v2, vec2(-89.90310912, 27.34824973)));
}
void main() {
  vec3 p = aPos * 0.001;
  gl_Position = uMvp * vec4(p.x, -p.y, -p.z, 1.0);
  gl_PointSize = uSize;
  vColor = mix(min(aColor * 1.25, vec3(1.0)), turbo(1.0 - (p.z - 0.3) / 1.9), uMode);
}`;

const FS = `#version 300 es
precision mediump float;
in vec3 vColor;
out vec4 outColor;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  outColor = vec4(vColor, 1.0);
}`;

type Mat = Float32Array;

function mul(a: Mat, b: Mat): Mat {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      o[c * 4 + r] = s;
    }
  return o;
}

function perspective(fovy: number, aspect: number, near: number, far: number): Mat {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
}

function translate(x: number, y: number, z: number): Mat {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
}

function rotX(a: number): Mat {
  const c = Math.cos(a),
    s = Math.sin(a);
  return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]);
}

function rotY(a: number): Mat {
  const c = Math.cos(a),
    s = Math.sin(a);
  return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]);
}

function shader(gl: WebGL2RenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  return s;
}

export default function PointCloud() {
  const canvas = useRef<HTMLCanvasElement>(null);
  const view = useRef({ yaw: 0.42, pitch: 0.42, user: false });
  const [depthColors, setDepthColors] = useState(false);
  const [failed, setFailed] = useState(false);
  const mode = useRef(0);

  useEffect(() => {
    mode.current = depthColors ? 1 : 0;
  }, [depthColors]);

  useEffect(() => {
    const el = canvas.current!;
    const gl = el.getContext("webgl2", { antialias: true, premultipliedAlpha: false });
    if (!gl) {
      setFailed(true);
      return;
    }
    const prog = gl.createProgram()!;
    gl.attachShader(prog, shader(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, shader(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      setFailed(true);
      return;
    }
    const uMvp = gl.getUniformLocation(prog, "uMvp");
    const uSize = gl.getUniformLocation(prog, "uSize");
    const uMode = gl.getUniformLocation(prog, "uMode");
    let count = 0;
    let raf = 0;
    let alive = true;
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const start = performance.now();

    fetch(asset("/cloud.bin"))
      .then((r) => r.arrayBuffer())
      .then((buf) => {
        if (!alive) return;
        count = new Int32Array(buf, 0, 1)[0];
        const pos = new Int16Array(buf, 4, count * 3);
        const col = new Uint8Array(buf, 4 + count * 6, count * 3);
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const pb = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, pb);
        gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
        const aPos = gl.getAttribLocation(prog, "aPos");
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.SHORT, false, 0, 0);
        const cb = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, cb);
        gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
        const aColor = gl.getAttribLocation(prog, "aColor");
        gl.enableVertexAttribArray(aColor);
        gl.vertexAttribPointer(aColor, 3, gl.UNSIGNED_BYTE, true, 0, 0);
      })
      .catch(() => setFailed(true));

    const frame = (now: number) => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = Math.round(el.clientWidth * dpr),
        h = Math.round(el.clientHeight * dpr);
      if (el.width !== w || el.height !== h) {
        el.width = w;
        el.height = h;
      }
      gl.viewport(0, 0, w, h);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      if (count) {
        const v = view.current;
        const t = (now - start) / 1000;
        const yaw = v.user || reduced ? v.yaw : v.yaw + 0.22 * Math.sin(t * 0.22);
        const pitch = v.user || reduced ? v.pitch : v.pitch + 0.05 * Math.sin(t * 0.17);
        const fovy = 2 * Math.atan(180 / 470) * (w / h < 1.2 ? 1.5 : 1);
        const mvp = mul(
          perspective(fovy, w / h, 0.02, 30),
          mul(translate(0.08, 0.06, w / h < 1.2 ? -2.3 : -1.75), mul(rotX(pitch), mul(rotY(yaw), translate(0, 0, 1.2)))),
        );
        gl.useProgram(prog);
        gl.enable(gl.DEPTH_TEST);
        gl.uniformMatrix4fv(uMvp, false, mvp);
        gl.uniform1f(uSize, Math.max(2, 3.4 * dpr * Math.min(1.3, w / (1000 * dpr))));
        gl.uniform1f(uMode, mode.current);
        gl.drawArrays(gl.POINTS, 0, count);
      }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);

    let drag: { x: number; y: number; yaw: number; pitch: number } | null = null;
    const down = (e: PointerEvent) => {
      const v = view.current;
      drag = { x: e.clientX, y: e.clientY, yaw: v.yaw, pitch: v.pitch };
      v.user = true;
      el.setPointerCapture(e.pointerId);
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      const v = view.current;
      v.yaw = drag.yaw + (e.clientX - drag.x) * 0.006;
      v.pitch = Math.max(-0.6, Math.min(1.2, drag.pitch + (e.clientY - drag.y) * 0.006));
    };
    const up = () => (drag = null);
    const key = (e: KeyboardEvent) => {
      const v = view.current;
      const d = { ArrowLeft: [-0.08, 0], ArrowRight: [0.08, 0], ArrowUp: [0, -0.06], ArrowDown: [0, 0.06] }[e.key];
      if (!d) return;
      e.preventDefault();
      v.user = true;
      v.yaw += d[0];
      v.pitch = Math.max(-0.6, Math.min(1.2, v.pitch + d[1]));
    };
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("keydown", key);
    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      el.removeEventListener("keydown", key);
    };
  }, []);

  return (
    <figure className={styles.wrap}>
      {failed ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          className={styles.fallback}
          src={asset("/img/mode-pointcloud.webp")}
          alt="Point cloud of a desk scene reconstructed by StereoVision"
        />
      ) : (
        <canvas
          ref={canvas}
          className={styles.canvas}
          tabIndex={0}
          role="img"
          aria-label="Interactive 3D point cloud of a desk with a laptop, a cup, a bottle, an orange and a book, reconstructed by the app's stereo matcher. Drag or use the arrow keys to rotate."
        />
      )}
      <figcaption className={styles.caption}>
        <span>Drag to look around the scene in 3D.</span>
        <button type="button" className={styles.toggle} aria-pressed={depthColors} onClick={() => setDepthColors((d) => !d)}>
          {depthColors ? "Show camera colors" : "Color by distance"}
        </button>
      </figcaption>
    </figure>
  );
}
