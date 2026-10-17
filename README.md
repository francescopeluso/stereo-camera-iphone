# StereoVision landing page

The landing page for [StereoVision](https://github.com/francescopeluso/stereo-camera-iphone), built with Next.js (App Router) and exported as a static site.

## Develop

```sh
npm install
npm run dev
```

## Deploy on Vercel

Import the repository, pick the `landing` branch, and keep the defaults: framework Next.js, build command `next build`. No environment variables are needed; set `NEXT_PUBLIC_SITE_URL` only if you use a custom domain.

## Build

```sh
npm run build        # static site in out/
```

For GitHub Pages under a project path, set the base path and the public URL at build time:

```sh
NEXT_PUBLIC_BASE_PATH=/stereo-camera-iphone \
NEXT_PUBLIC_SITE_URL=https://francescopeluso.github.io/stereo-camera-iphone \
npm run build
```

## Images and data

Every image, the point cloud and the numbers on the page come from a synthetic desk scene, processed by the app's own Metal matcher:

1. `scripts/render_scene.py` ray-traces the scene from two viewpoints 1 cm apart and saves the true depth and object IDs.
2. `scripts/run_sgm.swift` runs the app's `Shaders.metal` kernels (census, SGM with left-right check, block matching) on the rendered pair.
3. `scripts/make_assets.py` builds the view-mode screens, pipeline thumbnails, the point cloud (`public/cloud.bin`), the Open Graph image and `public/scene.json` with the object overlays. It also prints accuracy metrics against the true depth, for reference only: they are not shown on the page.

Regenerate everything with:

```sh
scripts/build_assets.sh ../stereo-camera-iphone/StereoVision/Shaders.metal
```

It needs a Mac with Metal, Swift and [uv](https://github.com/astral-sh/uv).
