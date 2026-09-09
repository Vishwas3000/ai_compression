# JPEG vs JPEG AI Benchmark

A small, reproducible harness for comparing conventional JPEG with a JPEG AI
implementation across rate, distortion, and runtime.

The official [JPEG AI reference software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software)
is source-available under its included BSD license and contains encoder,
decoder, training, and evaluation code. Its README currently targets Ubuntu
with NVIDIA CUDA. The license permits redistribution and modification but
explicitly does not grant patent rights; review that distinction before a
commercial deployment.

## What it measures

- encoded size and bits per pixel (bpp)
- RGB PSNR
- median encode and decode wall time
- source SHA-256 and run metadata for reproducibility

Compare codecs using **PSNR versus bpp curves**, not matching numeric quality
settings: each codec's setting has different semantics.

## Setup

```bash
cd jpeg-vs-jpeg-ai
uv sync
mkdir -p data/originals
```

Put lossless source images (PNG, TIFF, BMP, or PPM) in `data/originals`. Inputs
are normalized to 8-bit RGB PNG so both codecs receive identical pixels.

## JPEG baseline

The built-in baseline uses Pillow's linked libjpeg implementation with 4:2:0
chroma subsampling. Its exact Pillow and libjpeg versions are saved in the run
metadata; replace or extend this anchor if your study follows a specific common
test condition.

```bash
uv run python benchmark.py data/originals \
  --jpeg-quality 30 50 70 90 \
  --output results/jpeg.csv
```

## JPEG AI comparison

Supply command templates matching your JPEG AI implementation. The encoder
template supports `{input}`, `{output}`, and `{point}`; the decoder supports
`{input}`, `{output}`, and `{point}`.

```bash
uv run python benchmark.py data/originals \
  --jpeg-quality 30 50 70 90 \
  --jpeg-ai-points 0.25 0.5 1.0 2.0 \
  --jpeg-ai-encoder '/path/to/encoder --input {input} --output {output} --target {point}' \
  --jpeg-ai-decoder '/path/to/decoder --input {input} --output {output}' \
  --jpeg-ai-extension .bin \
  --jpeg-ai-decoded-extension .png \
  --output results/comparison.csv
```

Command templates are parsed as arguments and run without a shell. Adjust the
flags and operating points to the exact JPEG AI build being studied.

The CSV is written beside a `*.metadata.json` file. Encoded and decoded files
go to `artifacts/` for inspection.

## Check

```bash
uv run python -m unittest discover -s tests
```

For publishable runs, use a fixed lossless dataset, record the JPEG AI source
commit and model/checkpoint, keep the same machine and thread settings, and
report results across multiple operating points.
