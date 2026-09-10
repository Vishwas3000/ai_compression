#!/usr/bin/env python3
"""Render DEV article images from one benchmark run and native latent export."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BG = "#0b1020"
PANEL = "#151d33"
TEXT = "#f5f7ff"
MUTED = "#9ba8c7"
AI = "#30d5c8"
JPEG = "#ff8a4c"
MODEL = "#ffd166"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int, *,
         colour: str = TEXT, bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, value, fill=colour, font=font(size, bold), anchor=anchor)


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def rate_distortion(data: list[dict[str, str]], output: Path) -> None:
    image = Image.new("RGB", (1200, 760), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (70, 48), "Rate–distortion on the first test image", 42, bold=True)
    text(draw, (70, 98), "Higher PSNR is better; lower bits per pixel is smaller", 23, colour=MUTED)
    left, top, right, bottom = 105, 155, 1130, 650
    x_min, x_max, y_min, y_max = 0.1, 2.3, 29.0, 39.5

    def point(row: dict[str, str]) -> tuple[float, float]:
        x = left + (float(row["bits_per_pixel"]) - x_min) / (x_max - x_min) * (right - left)
        y = bottom - (float(row["psnr_db"]) - y_min) / (y_max - y_min) * (bottom - top)
        return x, y

    for bpp in (0.5, 1.0, 1.5, 2.0):
        x = left + (bpp - x_min) / (x_max - x_min) * (right - left)
        draw.line((x, top, x, bottom), fill="#28324c", width=2)
        text(draw, (int(x), bottom + 20), f"{bpp:.1f}", 18, colour=MUTED, anchor="ma")
    for psnr in (30, 32, 34, 36, 38):
        y = bottom - (psnr - y_min) / (y_max - y_min) * (bottom - top)
        draw.line((left, y, right, y), fill="#28324c", width=2)
        text(draw, (left - 18, int(y)), str(psnr), 18, colour=MUTED, anchor="rm")

    for codec, colour, label in (("jpeg", JPEG, "JPEG"), ("jpeg-ai", AI, "JPEG AI")):
        selected = sorted((row for row in data if row["codec"] == codec), key=lambda row: float(row["bits_per_pixel"]))
        points = [point(row) for row in selected]
        draw.line(points, fill=colour, width=5, joint="curve")
        for row, (x, y) in zip(selected, points):
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=colour)
            text(draw, (int(x + 11), int(y - 12)), row["setting"], 16, colour=colour)
        legend_x = 780 if codec == "jpeg" else 940
        draw.line((legend_x, 112, legend_x + 35, 112), fill=colour, width=5)
        text(draw, (legend_x + 45, 112), label, 19, colour=colour, bold=True, anchor="lm")

    text(draw, ((left + right) // 2, 704), "bits per pixel", 20, colour=MUTED, anchor="ma")
    text(draw, (28, (top + bottom) // 2), "PSNR (dB)", 20, colour=MUTED, anchor="lm")
    text(draw, (1130, 718), "ONE IMAGE • ONE RUN • SMOKE TEST", 15, colour="#ffcf80", anchor="ra")
    image.save(output, optimize=True)


def timing(data: list[dict[str, str]], output: Path) -> None:
    ai = [row for row in data if row["codec"] == "jpeg-ai"]
    values = {
        stage: [statistics.median(float(row[f"{stage}_{part}_ms"]) for row in ai)
                for part in ("codec", "model_load", "overhead")]
        for stage in ("encode", "decode")
    }
    image = Image.new("RGB", (1200, 680), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (70, 48), "The 4–6 second result was mostly startup", 42, bold=True)
    text(draw, (70, 100), "Median across five JPEG AI operating points; one image, one run each", 22, colour=MUTED)
    colours = (JPEG, MODEL, "#5576a8")
    labels = ("codec", "model loading", "Python / CUDA / process / I/O")
    x0, usable = 230, 760
    maximum = max(sum(parts) for parts in values.values())
    for row_index, stage in enumerate(("encode", "decode")):
        y = 210 + row_index * 190
        text(draw, (190, y + 38), stage.title(), 25, bold=True, anchor="rm")
        x = x0
        for part, colour in zip(values[stage], colours):
            width = part / maximum * usable
            draw.rounded_rectangle((x, y, x + width, y + 76), radius=10, fill=colour)
            if width > 105:
                text(draw, (int(x + width / 2), y + 38), f"{part / 1000:.2f}s", 19,
                     colour=BG if colour != "#5576a8" else TEXT, bold=True, anchor="mm")
            x += width
        text(draw, (int(x + 15), y + 38), f"{sum(values[stage]) / 1000:.2f}s total", 20, anchor="lm")
    legend_x = 220
    for label, colour in zip(labels, colours):
        draw.rounded_rectangle((legend_x, 570, legend_x + 24, 594), radius=5, fill=colour)
        text(draw, (legend_x + 34, 582), label, 17, colour=MUTED, anchor="lm")
        legend_x += 185 if label != labels[-1] else 0
    image.save(output, optimize=True)


def comparison(artifacts: Path, output: Path) -> None:
    names = [
        ("Original", artifacts / "sources/00030_TE_560x888_8bit_sRGB.png", "uncompressed source"),
        ("JPEG q70", artifacts / "jpeg/q70/00030_TE_560x888_8bit_sRGB.jpg", "1.081 bpp • 34.55 dB"),
        ("JPEG AI 75", artifacts / "jpeg-ai/75/00030_TE_560x888_8bit_sRGB.decoded.png", "1.092 bpp • 37.04 dB"),
    ]
    image = Image.new("RGB", (1200, 570), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (600, 45), "Nearly the same rate, different reconstruction", 38, bold=True, anchor="ma")
    text(draw, (600, 92), "A detail crop from the CC0 JPEG AI test image", 21, colour=MUTED, anchor="ma")
    for index, (label, path, caption) in enumerate(names):
        source = Image.open(path).convert("RGB").crop((65, 40, 515, 455)).resize((360, 332), Image.Resampling.LANCZOS)
        x = 25 + index * 390
        image.paste(source, (x, 150))
        draw.rectangle((x, 150, x + 359, 481), outline="#44506d", width=2)
        text(draw, (x + 180, 128), label, 23, bold=True, anchor="ma")
        text(draw, (x + 180, 510), caption, 18, colour=AI if index == 2 else MUTED, anchor="ma")
    text(draw, (1175, 550), "Smoke result; validate on a larger corpus before drawing codec-wide conclusions", 14,
         colour="#ffcf80", anchor="ra")
    image.save(output, optimize=True)


def latent_spaces(diagnostics: Path, output: Path) -> None:
    sources = [
        ("Input luma", diagnostics / "01-input-luma.png", "560 × 888 pixels"),
        ("y activation energy", diagnostics / "03-y-latent-energy.png", "160 × 56 × 35"),
        ("z activation energy", diagnostics / "05-z-hyperlatent-energy.png", "160 × 14 × 9"),
    ]
    image = Image.new("RGB", (1200, 720), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (600, 42), "What the native encoder sees", 40, bold=True, anchor="ma")
    text(draw, (600, 90), "Real tensors captured at Core ML model boundaries", 22, colour=MUTED, anchor="ma")
    for index, (label, path, caption) in enumerate(sources):
        panel = Image.new("RGB", (350, 520), PANEL)
        visual = fit(Image.open(path), (330, 470))
        panel.paste(visual, ((350 - visual.width) // 2, (520 - visual.height) // 2))
        x = 35 + index * 390
        image.paste(panel, (x, 140))
        text(draw, (x + 175, 125), label, 22, bold=True, anchor="ma")
        text(draw, (x + 175, 687), caption, 18, colour=AI if index else MUTED, anchor="ma")
    image.save(output, optimize=True)


def cover(diagnostics: Path, output: Path) -> None:
    image = Image.new("RGB", (1000, 420), BG)
    energy = Image.open(diagnostics / "03-y-latent-energy.png").convert("RGB").resize((330, 528), Image.Resampling.NEAREST)
    image.paste(energy.crop((0, 54, 330, 474)), (670, 0))
    draw = ImageDraw.Draw(image)
    for x in range(620, 740):
        alpha = (x - 620) / 120
        overlay = tuple(int(int(BG[i:i + 2], 16) * (1 - alpha)) for i in (1, 3, 5))
        draw.line((x, 0, x, 420), fill=overlay)
    draw.rounded_rectangle((55, 46, 272, 84), radius=19, fill=AI)
    text(draw, (163, 65), "CORE ML + me-tANS", 16, colour=BG, bold=True, anchor="mm")
    text(draw, (55, 125), "JPEG AI on", 52, bold=True)
    text(draw, (55, 184), "Apple silicon", 52, bold=True)
    text(draw, (55, 264), "Native encode. Native decode.", 25, colour="#d8e0f7")
    text(draw, (55, 300), "Benchmarks that separate codec time from startup.", 21, colour=MUTED)
    text(draw, (55, 371), "A reproducible engineering study", 17, colour=MODEL)
    image.save(output, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("diagnostics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = rows(args.results)
    rate_distortion(data, args.output / "rate-distortion.png")
    timing(data, args.output / "timing-breakdown.png")
    comparison(args.artifacts, args.output / "same-rate-comparison.png")
    latent_spaces(args.diagnostics, args.output / "latent-spaces.png")
    cover(args.diagnostics, args.output / "jpeg-ai-apple-cover.png")


if __name__ == "__main__":
    main()
