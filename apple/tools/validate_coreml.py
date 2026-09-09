#!/usr/bin/env python3
"""Run converted JPEG AI models with Core ML and compare reference outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.proto import FeatureTypes_pb2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--pixel-atol", type=float, default=0.5)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()

    references = sorted(args.references.rglob("*.npz"))
    if not references:
        parser.error(f"no reference files under {args.references}")

    failures = []
    passed = 0
    worst = 0.0
    for reference_path in references:
        relative = reference_path.relative_to(args.references)
        model_path = args.models / relative.with_suffix(".mlpackage")
        if not model_path.exists():
            failures.append(f"missing {model_path}")
            continue

        reference = np.load(reference_path)
        compute_units = ct.ComputeUnit.CPU_ONLY if args.cpu_only else ct.ComputeUnit.ALL
        model = ct.models.MLModel(str(model_path), compute_units=compute_units)
        spec = model.get_spec().description
        inputs = {}
        for index, feature in enumerate(spec.input):
            value = reference[f"input_{index}"]
            if feature.type.multiArrayType.dataType == FeatureTypes_pb2.ArrayFeatureType.INT32:
                value = value.astype(np.int32)
            inputs[feature.name] = value

        actual = model.predict(inputs)
        errors = []
        valid_model = True
        for index, feature in enumerate(spec.output):
            expected = reference[f"output_{index}"]
            value = actual[feature.name]
            error = float(np.max(np.abs(value.astype(np.float64) - expected)))
            errors.append(error)
            if np.issubdtype(expected.dtype, np.integer):
                valid = np.array_equal(value, expected)
            elif reference_path.name == "synthesis.npz":
                valid = error <= args.pixel_atol
            else:
                valid = np.allclose(value, expected, atol=args.atol, rtol=args.rtol)
            if not valid:
                valid_model = False
                failures.append(f"{relative}: output {index} max error {error:g}")
        worst = max(worst, *errors)
        if valid_model:
            passed += 1
        print(f"{'PASS' if valid_model else 'FAIL'} {relative} max_abs_error={max(errors):g}")

    print(f"Validated {passed}/{len(references)} models; worst error={worst:g}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
