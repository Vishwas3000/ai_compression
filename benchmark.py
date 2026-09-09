#!/usr/bin/env python3
"""Benchmark JPEG against an external JPEG AI encoder/decoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from typing import Callable

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
    "encode_ms",
    "decode_ms",
)


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
        description="Measure rate, RGB PSNR, and wall-clock latency for JPEG and JPEG AI."
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


def command(template: str, **values: str) -> list[str]:
    try:
        return [part.format_map(values) for part in shlex.split(template)]
    except KeyError as error:
        raise ValueError(f"unknown command placeholder: {error.args[0]}") from error


def run_command(template: str, *, cwd: Path | None = None, **values: str) -> None:
    argv = command(template, **values)
    completed = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no command output"
        raise RuntimeError(f"command failed ({completed.returncode}): {shlex.join(argv)}\n{detail}")


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
    encode_ms: float,
    decode_ms: float,
) -> dict[str, object]:
    size = encoded.stat().st_size
    return {
        "source": str(source),
        "source_sha256": source_hash,
        "codec": codec,
        "setting": setting,
        "width": reference.width,
        "height": reference.height,
        "encoded_bytes": size,
        "bits_per_pixel": round(size * 8 / (reference.width * reference.height), 6),
        "psnr_db": "inf" if math.isinf(value := psnr(reference, decoded)) else round(value, 6),
        "encode_ms": round(encode_ms, 3),
        "decode_ms": round(decode_ms, 3),
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
        encode_ms,
        decode_ms,
    )


def benchmark_jpeg_ai(
    source: Path,
    source_hash: str,
    reference_path: Path,
    reference: Image.Image,
    point: str,
    point_index: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    point_dir = args.work_dir / "jpeg-ai" / slug(point)
    encoded = point_dir / f"{reference_path.stem}{extension(args.jpeg_ai_extension)}"
    decoded_path = point_dir / f"{reference_path.stem}.decoded{extension(args.jpeg_ai_decoded_extension)}"
    point_dir.mkdir(parents=True, exist_ok=True)

    def encode() -> None:
        encoded.unlink(missing_ok=True)
        run_command(
            args.jpeg_ai_encoder,
            cwd=args.jpeg_ai_cwd,
            input=str(reference_path.resolve()),
            output=str(encoded.resolve()),
            point=point,
            index=str(point_index),
        )
        if not encoded.is_file():
            raise RuntimeError(f"JPEG AI encoder did not create {encoded}")

    encode_ms = median_ms(encode, args.warmup, args.repeats)

    def decode() -> None:
        decoded_path.unlink(missing_ok=True)
        run_command(
            args.jpeg_ai_decoder,
            cwd=args.jpeg_ai_cwd,
            input=str(encoded.resolve()),
            output=str(decoded_path.resolve()),
            point=point,
            index=str(point_index),
        )
        if not decoded_path.is_file():
            raise RuntimeError(f"JPEG AI decoder did not create {decoded_path}")

    decode_ms = median_ms(decode, args.warmup, args.repeats)
    return result_row(
        source,
        source_hash,
        "jpeg-ai",
        point,
        reference,
        load_rgb(decoded_path),
        encoded,
        encode_ms,
        decode_ms,
    )


def write_results(rows: list[dict[str, object]], args: argparse.Namespace) -> Path:
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
                    )
                )

        metadata_path = write_results(rows, args)
        print(f"Wrote {len(rows)} measurements to {args.output}")
        print(f"Wrote run metadata to {metadata_path}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
