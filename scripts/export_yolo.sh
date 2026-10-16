#!/bin/sh
set -e
MODEL="${1:-yolo11n-seg}"
case "$MODEL" in *-seg) NMS="" ;; *) NMS="nms=True" ;; esac
mkdir -p "$(dirname "$0")/../StereoVision/Models" && cd "$(dirname "$0")/../StereoVision/Models"
uv run --python 3.11 --with ultralytics --with "torch==2.5.1" --with "torchvision==0.20.1" --with "numpy<2" --with "coremltools==8.1" \
  yolo export model="$MODEL.pt" format=coreml imgsz=640 $NMS
rm -f "$MODEL.pt"
