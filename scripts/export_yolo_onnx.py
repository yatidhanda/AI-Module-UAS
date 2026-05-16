from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()
    from ultralytics import YOLO

    YOLO(args.model).export(format="onnx")


if __name__ == "__main__":
    main()
