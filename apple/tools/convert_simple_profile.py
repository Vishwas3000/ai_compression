#!/usr/bin/env python3
"""Convert the decoder-only JPEG AI simple-profile model set to Core ML."""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_onnx_to_coreml import convert_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()

    for tool in range(4):
        for component in ("model_y", "model_uv"):
            source = args.onnx_root / f"tools_{tool}" / component
            paths = [
                source / "common_modules" / "hyper_decoder.onnx",
                source / "common_modules" / "hyper_scale_decoder.onnx",
                source / "synthesis.onnx",
            ]
            if component == "model_y":
                paths += [source / "common_modules" / "MCM" / f"stage{stage}.onnx" for stage in range(4)]

            for path in paths:
                relative = path.relative_to(args.onnx_root).with_suffix(".mlpackage")
                output = args.output_root / relative
                if output.exists():
                    continue
                reference = args.references / relative.with_suffix(".npz") if args.references else None
                print(f"Converting {relative}", flush=True)
                convert_model(path, output, reference)


if __name__ == "__main__":
    main()
