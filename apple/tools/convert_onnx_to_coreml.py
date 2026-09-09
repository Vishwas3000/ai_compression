#!/usr/bin/env python3
"""Convert one exported JPEG AI ONNX graph into a flexible-shape Core ML package."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import torch
from onnx import TensorProto
from onnx2torch import convert


DTYPES = {
    TensorProto.FLOAT: (torch.float32, np.float32),
    TensorProto.INT32: (torch.int32, np.int32),
    TensorProto.INT64: (torch.int64, np.int32),
}


def _floor_div(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return torch.div(first, second, rounding_mode="floor")


def convert_model(
    input_path: Path,
    output_path: Path,
    reference_path: Path | None = None,
    sample_spatial_size: int = 8,
) -> None:
    graph = onnx.load(input_path)
    initializers = {item.name for item in graph.graph.initializer}
    graph_inputs = [item for item in graph.graph.input if item.name not in initializers]
    samples = []
    coreml_inputs = []
    rng = np.random.default_rng(0)

    for index, item in enumerate(graph_inputs):
        tensor_type = item.type.tensor_type
        torch_dtype, coreml_dtype = DTYPES[tensor_type.elem_type]
        shape = [dimension.dim_value or sample_spatial_size for dimension in tensor_type.shape.dim]
        if torch_dtype.is_floating_point:
            sample = torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
        else:
            sample = torch.from_numpy(rng.integers(-8, 8, shape, dtype=np.int64))
        samples.append(sample.to(torch_dtype))
        flexible_shape = [
            ct.RangeDim(1, 4096, default=value) if dimension.dim_param else value
            for value, dimension in zip(shape, tensor_type.shape.dim)
        ]
        coreml_inputs.append(
            ct.TensorType(name=f"input_{index}", shape=flexible_shape, dtype=coreml_dtype)
        )

    model = convert(graph).eval()
    if input_path.stem == "hyper_scale_decoder":
        divisions = [module for name, module in model.named_modules() if name.endswith("/Div")]
        if len(divisions) != 3:
            raise ValueError(f"expected three scale-model divisions, found {len(divisions)}")
        for module in divisions:
            module.math_op_function = _floor_div
    traced = torch.jit.trace(model, tuple(samples), strict=False)
    converted = ct.convert(
        traced,
        inputs=coreml_inputs,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT32,
    )
    converted.author = "JPEG AI Apple portability project"
    converted.license = "See the JPEG AI reference software BSD license and patent notice"
    converted.short_description = f"Converted from {input_path.name}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted.save(output_path)
    if reference_path:
        with torch.no_grad():
            result = model(*samples)
        outputs = result if isinstance(result, (tuple, list)) else (result,)
        arrays = {f"input_{index}": value.cpu().numpy() for index, value in enumerate(samples)}
        arrays.update({f"output_{index}": value.cpu().numpy() for index, value in enumerate(outputs)})
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(reference_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--sample-spatial-size", type=int, default=8)
    args = parser.parse_args()
    convert_model(args.input, args.output, args.reference, args.sample_spatial_size)


if __name__ == "__main__":
    main()
