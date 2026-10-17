#!/bin/sh
set -e
cd "$(dirname "$0")/.."
SHADERS="${1:-../stereo-camera-iphone/StereoVision/Shaders.metal}"
PY="uv run --python 3.12 --with numpy --with pillow python -I"
$PY scripts/render_scene.py scripts/out
swift scripts/run_sgm.swift "$SHADERS" scripts/out/left_640.png scripts/out/right_640.png scripts/out
$PY scripts/make_assets.py scripts/out public
