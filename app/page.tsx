/* eslint-disable @next/next/no-img-element */
import { Clapperboard, Columns2, GitCompareArrows, Rotate3d, Ruler, SlidersHorizontal } from "lucide-react";
import AnaglyphTitle from "@/app/components/AnaglyphTitle";
import CompareSlider from "@/app/components/CompareSlider";
import ErrorExplorer from "@/app/components/ErrorExplorer";
import Formula from "@/app/components/Formula";
import ModeViewer from "@/app/components/ModeViewer";
import PointCloud from "@/app/components/PointCloud";
import { asset } from "@/app/lib/asset";
import styles from "./page.module.css";

const REPO = "https://github.com/francescopeluso/stereo-camera-iphone";
const SITE = "https://francescopeluso.xyz";

const STEPS = [
  { title: "Capture", text: "Both cameras run in one synchronized session, so the two frames share the same instant.", img: "mode-raw" },
  { title: "Rectify", text: "Both views are re-projected with the iOS calibration until matching points share a row.", img: "use-pair" },
  {
    title: "Census",
    text: "Each pixel becomes a 48-bit fingerprint of its 7×7 neighborhood, immune to exposure differences.",
    img: "step-census",
  },
  {
    title: "Match",
    text: "Every candidate shift gets a cost; semi-global matching smooths it along four directions.",
    img: "step-disparity",
  },
  {
    title: "Check",
    text: "The match is recomputed from the other view. Pixels where the two disagree, in red, are dropped.",
    img: "step-lr",
  },
  { title: "Triangulate", text: "The shift between the views becomes a distance: Z = f·B / d.", img: "mode-depth" },
  { title: "Recognize", text: "YOLO11n-seg outlines objects; each takes the median depth inside its outline.", img: "mode-standard" },
];

const USES = [
  {
    icon: Columns2,
    title: "See epipolar geometry",
    text: "Rectified pair shows both views with the epipolar lines drawn in. Every match lies on the same row, which is why the search is one-dimensional.",
  },
  {
    icon: GitCompareArrows,
    title: "Compare algorithms live",
    text: "Block matching or SGM, left-right check on or off, Apple's own Dual-Wide depth, or LiDAR on Pro models: all on the same frame.",
  },
  {
    icon: SlidersHorizontal,
    title: "Tune it until it breaks",
    text: "Change the disparity range, the smoothness penalties and the uniqueness threshold, and watch exactly where matching starts to fail.",
  },
  {
    icon: Ruler,
    title: "Measure real objects",
    text: "Every recognized object gets a distance, a sideways offset and an approximate size. Tap the image to measure any single point.",
  },
  {
    icon: Rotate3d,
    title: "Walk around the scene",
    text: "The point cloud view turns depth back into geometry you can orbit, like the one at the top of this page.",
  },
  {
    icon: Clapperboard,
    title: "Record a demo",
    text: "Any view records to video with every overlay, for the presentation where a live demo isn't an option.",
  },
];

const FAILURES = [
  [
    "Textureless surfaces",
    "White walls and plain tables give the matcher nothing to compare. SGM fills some in from the edges; the rest stays black.",
  ],
  [
    "Reflections and glass",
    "Mirrors, windows and screens show a reflection, and the app measures the distance to that instead of the surface.",
  ],
  ["Repeating patterns", "Tiles, grids and blinds offer many equally good matches, so the wrong one sometimes wins."],
  ["Occlusion borders", "A strip beside every foreground object is visible to only one camera. The left-right check marks it as invalid."],
  ["Thin structures", "Cables and chair legs are often narrower than the matching window and get smoothed away."],
  ["Low light and motion", "Sensor noise and motion blur erase the fine texture matching relies on."],
] as const;

function Head({ id, title, children }: { id: string; title: string; children: React.ReactNode }) {
  return (
    <div className={styles.head}>
      <h2 id={id}>{title}</h2>
      <p>{children}</p>
    </div>
  );
}

export default function Home() {
  return (
    <>
      <header className={styles.hero} id="top">
        <nav className={styles.nav} aria-label="Main">
          <a href="#top" className={styles.brand}>
            <img src={asset("/logo.png")} alt="" width={36} height={24} />
            StereoVision
          </a>
          <ul className={styles.navLinks}>
            <li>
              <a href="#views">Views</a>
            </li>
            <li>
              <a href="#how">How it works</a>
            </li>
            <li>
              <a href="#matching">Matching</a>
            </li>
            <li>
              <a href="#limits">Limits</a>
            </li>
          </ul>
          <a className={styles.navCta} href={REPO}>
            GitHub
          </a>
        </nav>
        <div className={styles.heroGrid}>
          <div className={styles.heroCloud}>
            <PointCloud />
          </div>
          <div className={styles.heroText}>
            <AnaglyphTitle text="Your iPhone already has two eyes." />
            <p className={styles.lede}>
              StereoVision uses the wide and ultra-wide cameras as a stereo pair. It works out how far away everything is, recognizes the
              objects in front of it and shows the scene nine different ways, live, on the phone.
            </p>
            <div className={styles.actions}>
              <a className={styles.primary} href={REPO}>
                View the source on GitHub
              </a>
              <a className={styles.secondary} href="#views">
                See the nine views
              </a>
            </div>
          </div>
        </div>
      </header>

      <main>
        <section className={`${styles.section} ${styles.night}`} aria-labelledby="views-title">
          <div className={styles.inner} id="views">
            <Head id="views-title" title="Nine ways to look at one scene">
              One tap in the side panel switches what the app draws: the camera image with every object labeled, depth as color, the raw
              stereo signal, a top-down map or the scene in 3D.
            </Head>
            <ModeViewer />
          </div>
        </section>

        <section className={styles.section} aria-labelledby="how-title">
          <div className={styles.inner} id="how">
            <Head id="how-title" title="Two images in, one distance out">
              Seven steps run on the phone&rsquo;s GPU for every frame. All of them are written from scratch in Metal, except object
              recognition.
            </Head>
            <ol className={styles.chain}>
              {STEPS.map((s) => (
                <li key={s.title}>
                  <img src={asset(`/img/${s.img}.webp`)} alt="" width={1280} height={720} loading="lazy" />
                  <h3>{s.title}</h3>
                  <p>{s.text}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section className={styles.section} aria-labelledby="uses-title">
          <div className={styles.inner} id="uses">
            <Head id="uses-title" title="What it's good for">
              I built StereoVision as a personal study project, to get a hands-on feel for what my Artificial Vision lectures covered. These
              are the things it turned out to be good at.
            </Head>
            <ul className={styles.uses}>
              {USES.map((u) => (
                <li key={u.title}>
                  <u.icon aria-hidden="true" strokeWidth={1.75} />
                  <h3>{u.title}</h3>
                  <p>{u.text}</p>
                </li>
              ))}
            </ul>
            <div className={styles.export}>
              <div className={styles.exportText}>
                <h3>Take a frame home</h3>
                <p>
                  Every photo also saves its raw data to the Files app, ready for Python, OpenCV or MeshLab. Videos and photos of the
                  current view, overlays included, go to Photos.
                </p>
              </div>
              <ul className={styles.files} aria-label="Files saved with each photo">
                <li>
                  <code>rgb.png</code>
                  <span>the color image the depth refers to</span>
                </li>
                <li>
                  <code>depth_mm.png</code>
                  <span>16-bit depth in millimeters, 0 where unknown</span>
                </li>
                <li>
                  <code>cloud.ply</code>
                  <span>colored point cloud in meters</span>
                </li>
                <li>
                  <code>meta.json</code>
                  <span>intrinsics, baseline, settings and detected objects</span>
                </li>
                <li>
                  <code>view.png</code>
                  <span>exactly what was on screen</span>
                </li>
              </ul>
              <pre className={styles.code} aria-label="Python example">
                <code>
                  {`import cv2, json
depth = cv2.imread("depth_mm.png", -1) / 1000
K = json.load(open("meta.json"))["intrinsics"]`}
                </code>
              </pre>
            </div>
          </div>
        </section>

        <section className={`${styles.section} ${styles.band}`} aria-labelledby="matching-title">
          <div className={styles.inner} id="matching">
            <Head id="matching-title" title="Why semi-global matching">
              Finding the same point in both images is the hard part of stereo vision. How the app does it decides whether a white wall
              comes out as a smooth surface or as a field of holes.
            </Head>
            <div className={styles.accuracy}>
              <CompareSlider />
              <dl className={styles.points}>
                <div>
                  <dt>Block matching looks at each pixel alone</dt>
                  <dd>
                    It compares a small window around every pixel and keeps the best shift. Where the surface has little texture, many
                    shifts look equally good, and the result is holes and speckles.
                  </dd>
                </div>
                <div>
                  <dt>SGM asks the neighbors</dt>
                  <dd>
                    Semi-global matching adds a cost for changing distance from one pixel to the next, gathered along four directions. Flat
                    areas fill in, while real edges, where the jump is worth paying for, stay sharp.
                  </dd>
                </div>
                <div>
                  <dt>The left-right check removes what can&rsquo;t be trusted</dt>
                  <dd>
                    The match is computed again starting from the other camera. Where the two answers disagree, mostly along the borders of
                    objects that hide what&rsquo;s behind them, the pixel is marked unknown instead of guessed.
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        </section>

        <section className={styles.section} aria-labelledby="limits-title">
          <div className={styles.inner} id="limits">
            <Head id="limits-title" title="Where it stops working">
              The two lenses sit about 1&nbsp;cm apart, against roughly 6.5&nbsp;cm between human eyes. That one number sets most of the
              limits.
            </Head>
            <div className={styles.math}>
              <div className={styles.formula}>
                <Formula tex={String.raw`Z = \frac{f \cdot B}{d}`} label="Z equals f times B divided by d" />
                <p>
                  Distance <em>Z</em> from the focal length <em>f</em> (about 470&nbsp;px), the baseline <em>B</em> (about 1&nbsp;cm) and
                  the disparity <em>d</em>. At <em>d</em>&nbsp;=&nbsp;1&nbsp;px, <em>Z</em>&nbsp;≈&nbsp;4.7&nbsp;m: the farthest the app can
                  tell apart from infinity.
                </p>
              </div>
              <div className={styles.formula}>
                <Formula
                  tex={String.raw`\Delta Z \approx \frac{Z^2}{f \cdot B}\,\Delta d`}
                  label="delta Z is approximately Z squared over f times B, times delta d"
                />
                <p>
                  Differentiate and the error grows with the square of the distance. Twice as far means four times less precise, so beyond 2
                  to 3&nbsp;m the readings are mostly noise.
                </p>
              </div>
            </div>
            <ErrorExplorer />
            <h3 className={styles.subhead}>Scenes that fool stereo vision</h3>
            <dl className={styles.failures}>
              {FAILURES.map(([t, d]) => (
                <div key={t}>
                  <dt>{t}</dt>
                  <dd>{d}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>
      </main>

      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <div className={styles.footerBrand}>
            <img src={asset("/logo.png")} alt="" width={56} height={36} />
            <p>
              StereoVision is a personal study project, built to understand the Artificial Vision lectures better. It isn&rsquo;t tied to
              any course assignment. Tested only on iPhone 16.
            </p>
          </div>
          <ul className={styles.footerLinks}>
            <li>
              <a href={REPO}>Source on GitHub</a>
            </li>
            <li>
              Made by <a href={SITE}>Francesco Peluso</a>
            </li>
          </ul>
        </div>
      </footer>
    </>
  );
}
