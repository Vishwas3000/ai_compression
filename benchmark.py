#!/usr/bin/env python3
"""Benchmark JPEG against an external JPEG AI encoder/decoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

from PIL import Image, ImageChops, ImageOps, features


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff"}
FIELDS = (
    "source",
    "source_sha256",
    "codec",
    "setting",
    "width",
    "height",
    "encoded_bytes",
    "bits_per_pixel",
    "psnr_db",
    "psnr_y_db",
    "psnr_u_db",
    "psnr_v_db",
    "ssim_y",
    "ms_ssim_y",
    "lpips_alex",
    "encode_total_ms",
    "encode_codec_ms",
    "encode_model_load_ms",
    "encode_overhead_ms",
    "decode_total_ms",
    "decode_codec_ms",
    "decode_model_load_ms",
    "decode_overhead_ms",
)


class Timing(NamedTuple):
    total_ms: float
    codec_ms: float | None
    model_load_ms: float | None
    overhead_ms: float | None


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure rate, image quality, and wall-clock latency for JPEG and JPEG AI."
    )
    parser.add_argument("input", type=Path, help="Image file or directory of source images")
    parser.add_argument(
        "--output", type=Path, default=Path("results/results.csv"), help="Result CSV path"
    )
    parser.add_argument(
        "--work-dir", type=Path, default=Path("artifacts"), help="Encoded/decoded artifact directory"
    )
    parser.add_argument(
        "--jpeg-quality", nargs="+", type=int, default=[30, 50, 70, 90], metavar="Q"
    )
    parser.add_argument(
        "--jpeg-ai-encoder",
        help="Encoder command template using {input}, {output}, {point}, and {index}",
    )
    parser.add_argument(
        "--jpeg-ai-decoder",
        help="Decoder command template using {input}, {output}, {point}, and {index}",
    )
    parser.add_argument(
        "--jpeg-ai-cwd", type=Path, help="Working directory for JPEG AI commands"
    )
    parser.add_argument("--jpeg-ai-points", nargs="+", default=[], metavar="POINT")
    parser.add_argument("--jpeg-ai-extension", default=".bin", help="Encoded bitstream extension")
    parser.add_argument("--jpeg-ai-decoded-extension", default=".png")
    parser.add_argument("--warmup", type=nonnegative_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument(
        "--perceptual-metrics", action="store_true",
        help="also calculate SSIM-Y, MS-SSIM-Y, and LPIPS-Alex (optional PyTorch dependencies)",
    )
    parser.add_argument(
        "--metrics-device", choices=("auto", "cpu", "cuda", "mps"), default="auto",
        help="device for optional perceptual metrics (default: auto)",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise ValueError(f"input does not exist: {args.input}")
    if any(quality < 1 or quality > 100 for quality in args.jpeg_quality):
        raise ValueError("JPEG quality must be between 1 and 100")
    ai_values = (args.jpeg_ai_encoder, args.jpeg_ai_decoder, args.jpeg_ai_points)
    if any(ai_values) and not all(ai_values):
        raise ValueError(
            "JPEG AI requires --jpeg-ai-encoder, --jpeg-ai-decoder, and --jpeg-ai-points"
        )
    for extension in (args.jpeg_ai_extension, args.jpeg_ai_decoded_extension):
        if "/" in extension or "\\" in extension:
            raise ValueError("extensions must not contain a path separator")
    if args.jpeg_ai_cwd is not None and not args.jpeg_ai_cwd.is_dir():
        raise ValueError(f"JPEG AI working directory does not exist: {args.jpeg_ai_cwd}")


def image_paths(source: Path) -> list[Path]:
    paths = [source] if source.is_file() else source.rglob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median_ms(action: Callable[[], None], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        action()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def reference_timing(output: str, total_ms: float) -> Timing:
    total = re.search(r"TOTAL:\s*(\d+):(\d+):([\d.]+)", output)
    loading = re.search(r"Loading models:\s*([\d.]+)\s*second", output)
    if total is None or loading is None:
        return Timing(total_ms, None, None, None)
    codec_ms = (int(total[1]) * 3600 + int(total[2]) * 60 + float(total[3])) * 1000
    load_ms = float(loading[1]) * 1000
    return Timing(total_ms, codec_ms, load_ms, max(0.0, total_ms - codec_ms - load_ms))


def median_command(action: Callable[[], str], warmup: int, repeats: int) -> Timing:
    for _ in range(warmup):
        action()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        output = action()
        samples.append(reference_timing(output, (time.perf_counter() - started) * 1000))
    return Timing(*(statistics.median(values) if None not in values else None for values in zip(*samples)))


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def psnr(reference: Image.Image, decoded: Image.Image) -> float:
    if reference.size != decoded.size:
        raise ValueError(f"decoded size {decoded.size} does not match source size {reference.size}")
    histogram = ImageChops.difference(reference, decoded).histogram()
    squared_error = sum((value % 256) ** 2 * count for value, count in enumerate(histogram))
    mse = squared_error / (reference.width * reference.height * 3)
    return math.inf if mse == 0 else 10 * math.log10(255**2 / mse)


def yuv_psnr(reference: Image.Image, decoded: Image.Image) -> tuple[float, float, float]:
    if reference.size != decoded.size:
        raise ValueError(f"decoded size {decoded.size} does not match source size {reference.size}")
    squared = [0.0, 0.0, 0.0]
    original = reference.tobytes()
    reconstructed = decoded.tobytes()
    for offset in range(0, len(original), 3):
        red = original[offset] - reconstructed[offset]
        green = original[offset + 1] - reconstructed[offset + 1]
        blue = original[offset + 2] - reconstructed[offset + 2]
        y = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        u = (blue - y) / 1.8556
        v = (red - y) / 1.5748
        for index, error in enumerate((y, u, v)):
            squared[index] += error * error
    pixels = reference.width * reference.height
    return tuple(
        math.inf if total == 0 else 10 * math.log10(255**2 / (total / pixels))
        for total in squared
    )


class PerceptualMetrics:
    def __init__(self, enabled: bool, requested_device: str) -> None:
        self.enabled = enabled
        self.device_name: str | None = None
        self.versions: dict[str, str] = {}
        if not enabled:
            return
        try:
            import lpips
            import numpy
            import torch
            from pytorch_msssim import ms_ssim, ssim
        except ImportError as error:
            raise ValueError(
                "--perceptual-metrics requires torch, torchvision, numpy, lpips, and "
                "pytorch-msssim; install requirements-metrics.txt into this Python environment"
            ) from error
        if requested_device == "auto":
            requested_device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise ValueError("--metrics-device cuda requested, but CUDA is unavailable")
        if requested_device == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "--metrics-device mps requested, but Metal Performance Shaders is unavailable"
            )
        self.device_name = requested_device
        self.device = torch.device(requested_device)
        self.numpy = numpy
        self.torch = torch
        self.ssim = ssim
        self.ms_ssim = ms_ssim
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        for package in ("lpips", "numpy", "pytorch-msssim", "torch", "torchvision"):
            self.versions[package] = importlib.metadata.version(package)

    def measure(self, reference: Image.Image, decoded: Image.Image) -> dict[str, float | None]:
        if not self.enabled:
            return {"ssim_y": None, "ms_ssim_y": None, "lpips_alex": None}
        torch = self.torch

        def tensor(image: Image.Image):
            array = self.numpy.asarray(image, dtype=self.numpy.float32).copy()
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device) / 255

        original = tensor(reference)
        reconstructed = tensor(decoded)
        original_y = (
            0.2126 * original[:, 0:1] + 0.7152 * original[:, 1:2]
            + 0.0722 * original[:, 2:3]
        )
        reconstructed_y = (
            0.2126 * reconstructed[:, 0:1] + 0.7152 * reconstructed[:, 1:2]
            + 0.0722 * reconstructed[:, 2:3]
        )
        height, width = original_y.shape[-2:]
        padding = (0, max(0, 164 - width), 0, max(0, 164 - height))
        with torch.inference_mode():
            ssim_y = self.ssim(original_y, reconstructed_y, data_range=1).item()
            ms_ssim_y = self.ms_ssim(
                torch.nn.functional.pad(original_y, padding),
                torch.nn.functional.pad(reconstructed_y, padding),
                data_range=1,
            ).item()
            lpips_alex = self.lpips(original * 2 - 1, reconstructed * 2 - 1).item()
        return {"ssim_y": ssim_y, "ms_ssim_y": ms_ssim_y, "lpips_alex": lpips_alex}


def command(template: str, **values: str) -> list[str]:
    try:
        return [part.format_map(values) for part in shlex.split(template)]
    except KeyError as error:
        raise ValueError(f"unknown command placeholder: {error.args[0]}") from error


def run_command(template: str, *, cwd: Path | None = None, **values: str) -> str:
    argv = command(template, **values)
    completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no command output"
        raise RuntimeError(f"command failed ({completed.returncode}): {shlex.join(argv)}\n{detail}")
    return "\n".join((completed.stdout, completed.stderr))


def extension(value: str) -> str:
    return value if value.startswith(".") else f".{value}"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "point"


def result_row(
    source: Path,
    source_hash: str,
    codec: str,
    setting: str,
    reference: Image.Image,
    decoded: Image.Image,
    encoded: Path,
    encode: Timing,
    decode: Timing,
    perceptual: PerceptualMetrics,
) -> dict[str, object]:
    size = encoded.stat().st_size

    def rounded(value: float | None) -> float | str:
        return "" if value is None else round(value, 3)

    rgb_psnr = psnr(reference, decoded)
    psnr_y, psnr_u, psnr_v = yuv_psnr(reference, decoded)
    quality = perceptual.measure(reference, decoded)

    def quality_value(value: float | None) -> float | str:
        if value is None:
            return ""
        return "inf" if math.isinf(value) else round(value, 6)

    return {
        "source": str(source),
        "source_sha256": source_hash,
        "codec": codec,
        "setting": setting,
        "width": reference.width,
        "height": reference.height,
        "encoded_bytes": size,
        "bits_per_pixel": round(size * 8 / (reference.width * reference.height), 6),
        "psnr_db": quality_value(rgb_psnr),
        "psnr_y_db": quality_value(psnr_y),
        "psnr_u_db": quality_value(psnr_u),
        "psnr_v_db": quality_value(psnr_v),
        "ssim_y": quality_value(quality["ssim_y"]),
        "ms_ssim_y": quality_value(quality["ms_ssim_y"]),
        "lpips_alex": quality_value(quality["lpips_alex"]),
        "encode_total_ms": rounded(encode.total_ms),
        "encode_codec_ms": rounded(encode.codec_ms),
        "encode_model_load_ms": rounded(encode.model_load_ms),
        "encode_overhead_ms": rounded(encode.overhead_ms),
        "decode_total_ms": rounded(decode.total_ms),
        "decode_codec_ms": rounded(decode.codec_ms),
        "decode_model_load_ms": rounded(decode.model_load_ms),
        "decode_overhead_ms": rounded(decode.overhead_ms),
    }


def benchmark_jpeg(
    source: Path,
    source_hash: str,
    reference_path: Path,
    reference: Image.Image,
    quality: int,
    work_dir: Path,
    warmup: int,
    repeats: int,
    perceptual: PerceptualMetrics,
) -> dict[str, object]:
    encoded = work_dir / "jpeg" / f"q{quality}" / f"{reference_path.stem}.jpg"
    encoded.parent.mkdir(parents=True, exist_ok=True)

    def encode() -> None:
        with Image.open(reference_path) as image:
            image.save(encoded, "JPEG", quality=quality, subsampling=2)

    encode_ms = median_ms(encode, warmup, repeats)

    def decode() -> None:
        with Image.open(encoded) as image:
            image.load()

    decode_ms = median_ms(decode, warmup, repeats)
    return result_row(
        source,
        source_hash,
        "jpeg",
        str(quality),
        reference,
        load_rgb(encoded),
        encoded,
        Timing(encode_ms, encode_ms, 0, 0),
        Timing(decode_ms, decode_ms, 0, 0),
        perceptual,
    )


def benchmark_jpeg_ai(
    source: Path,
    source_hash: str,
    reference_path: Path,
    reference: Image.Image,
    point: str,
    point_index: int,
    args: argparse.Namespace,
    perceptual: PerceptualMetrics,
) -> dict[str, object]:
    point_dir = args.work_dir / "jpeg-ai" / slug(point)
    encoded = point_dir / f"{reference_path.stem}{extension(args.jpeg_ai_extension)}"
    decoded_path = point_dir / f"{reference_path.stem}.decoded{extension(args.jpeg_ai_decoded_extension)}"
    point_dir.mkdir(parents=True, exist_ok=True)

    def encode() -> str:
        encoded.unlink(missing_ok=True)
        output = run_command(
            args.jpeg_ai_encoder,
            cwd=args.jpeg_ai_cwd,
            input=str(reference_path.resolve()),
            output=str(encoded.resolve()),
            point=point,
            index=str(point_index),
        )
        if not encoded.is_file():
            raise RuntimeError(f"JPEG AI encoder did not create {encoded}")
        return output

    encode_timing = median_command(encode, args.warmup, args.repeats)

    def decode() -> str:
        decoded_path.unlink(missing_ok=True)
        output = run_command(
            args.jpeg_ai_decoder,
            cwd=args.jpeg_ai_cwd,
            input=str(encoded.resolve()),
            output=str(decoded_path.resolve()),
            point=point,
            index=str(point_index),
        )
        if not decoded_path.is_file():
            raise RuntimeError(f"JPEG AI decoder did not create {decoded_path}")
        return output

    decode_timing = median_command(decode, args.warmup, args.repeats)
    return result_row(
        source,
        source_hash,
        "jpeg-ai",
        point,
        reference,
        load_rgb(decoded_path),
        encoded,
        encode_timing,
        decode_timing,
        perceptual,
    )


def write_results(
    rows: list[dict[str, object]], args: argparse.Namespace, perceptual: PerceptualMetrics
) -> Path:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = args.output.with_name(f"{args.output.stem}.metadata.json")
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "pillow": Image.__version__,
        "libjpeg": features.version_codec("jpg"),
        "jpeg_chroma_subsampling": "4:2:0",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "perceptual_metrics": args.perceptual_metrics,
        "metrics_device_requested": args.metrics_device,
        "metrics_device": perceptual.device_name,
        "metric_versions": perceptual.versions,
        "jpeg_ai_encoder": args.jpeg_ai_encoder,
        "jpeg_ai_decoder": args.jpeg_ai_decoder,
        "jpeg_ai_cwd": str(args.jpeg_ai_cwd) if args.jpeg_ai_cwd else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        sources = image_paths(args.input)
        if not sources:
            raise ValueError(f"no supported images found in {args.input}")

        perceptual = PerceptualMetrics(args.perceptual_metrics, args.metrics_device)
        rows: list[dict[str, object]] = []
        source_dir = args.work_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        for source in sources:
            relative_source = (
                Path(source.name) if args.input.is_file() else source.relative_to(args.input)
            )
            reference_path = (source_dir / relative_source).with_suffix(".png")
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                reference = ImageOps.exif_transpose(image).convert("RGB")
            reference.save(reference_path, "PNG")
            source_hash = sha256(source)

            for quality in args.jpeg_quality:
                rows.append(
                    benchmark_jpeg(
                        source,
                        source_hash,
                        reference_path,
                        reference,
                        quality,
                        args.work_dir,
                        args.warmup,
                        args.repeats,
                        perceptual,
                    )
                )
            for point_index, point in enumerate(args.jpeg_ai_points):
                rows.append(
                    benchmark_jpeg_ai(
                        source,
                        source_hash,
                        reference_path,
                        reference,
                        point,
                        point_index,
                        args,
                        perceptual,
                    )
                )

        metadata_path = write_results(rows, args, perceptual)
        print(f"Wrote {len(rows)} measurements to {args.output}")
        print(f"Wrote run metadata to {metadata_path}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
