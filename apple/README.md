# Native Apple JPEG AI codec

This Swift package targets macOS 13 and iOS 16 using Core ML with CPU and GPU
compute. The current simple-profile path provides:

- PNG-to-JPEG-AI encoding with the reference rate presets
- JPEG AI marker and picture-header parsing
- native me-tANS factorized and SGM entropy encoding and decoding
- model-table selection plus hyper- and residual-latent decoding
- loading converted decoder stages as Core ML models
- MCM latent reconstruction, synthesis, BT.709 conversion, and RGB PNG output
- runtime `y`/`z` activation and entropy-mask visualizations

The current decoder covers untiled, 4:4:4 simple-profile codestreams. Chroma
subsampling and the optional enhancement/filter tools are not implemented yet.

## Check

```bash
cd apple
swift test
```

Encode a PNG, optionally exporting the actual inference tensors produced by
Core ML:

```bash
swift run jpegai-info --encode INPUT.png TABLES_DIR COREML_MODELS_DIR \
  OUTPUT.bits MODEL BETA [VISUALIZATIONS_DIR]
```

Inspect a codestream or decode it to PNG:

```bash
swift run jpegai-info INPUT.bits [TABLES_DIR [COREML_MODELS_DIR]]
swift run jpegai-info INPUT.bits TABLES_DIR COREML_MODELS_DIR OUTPUT.png
```

## macOS app

Open `apple/Package.swift` in Xcode, select the **JPEGAIDecoder** scheme and
**My Mac**, then press **Run**. When launched by Xcode, the app reads generated
tables and Core ML packages from `apple/Models/`; the self-contained bundle
uses its copied resources instead.

Build a self-contained, locally signed app from the tables and models under
`Models/`, then double-click it or launch it from Terminal:

```bash
./build_macos_app.sh
open dist/JPEGAIDecoder.app
```

The app can encode a PNG or decode a `.bits` file and displays the reconstructed
image and first/repeat timing. Its result separates the source PNG, the actual
JPEG AI `.bits` payload, a native macOS JPEG-at-quality-0.70 baseline, and the
lossless decoded PNG preview. A preview can be larger than the source PNG; it is
not the compressed payload. During encoding the app reports the active model or
codec stage. Leave **Export y/z inference visualizations** selected, then choose
**View Inference Steps** to browse the input luma, every learned `y` and rounded
`z` channel, energy maps, entropy-mask density, and quantized context residual
inside the app. These are real per-image tensors, not a conceptual illustration.
The explorer can also reveal the generated PNGs in Finder. Netron can separately
open an ONNX model such as `Models/onnx/tools_2/model_y/analysis.onnx` to inspect
the fixed layer graph.

The generated app stays under ignored `dist/`; the local build includes tables
and converted models for all four simple-profile models. Distribution to other
Macs requires Developer ID signing and notarization.

`TABLES_DIR` contains `unique_z_distributions.csv` and the matching `Y_*.csv`
and `UV_*.csv` mappings. PyTorch 2.11 requires the included legacy-export patch
before exporting the reference graphs:

```bash
git -C JPEG_AI_REFERENCE apply --unidiff-zero \
  "$PWD/patches/jpeg-ai-pytorch-2.11-onnx.patch"
```

Convert exported ONNX encoder and decoder graphs with:

```bash
python tools/convert_simple_profile.py ONNX_ROOT Models/apple-coreml-simple \
  --references Models/apple-coreml-reference
python tools/validate_coreml.py Models/apple-coreml-simple Models/apple-coreml-reference
```

Generated tables, references, and model packages belong under `apple/Models/`
and are intentionally excluded from Git.

## Reference check

The native decoder reproduces the official entropy control-point hashes for
the tested 560x888 model-1, model-2, and model-3 benchmark streams. Their RGB
outputs measure 58.677–64.462 dB PSNR versus the official PNGs. The remaining
error is concentrated at the bottom edge because the exported ONNX synthesis
graphs omit the reference decoder's runtime height crop.

A native model-2 stream also decoded successfully with the official PyTorch
reference decoder, with matching hyper-latent, scale, mask, and residual hashes.
Its native and reference reconstructions measured 58.37 dB PSNR against each
other; the same bottom-edge export difference remains, so this is an
interoperability milestone rather than a claim of complete pixel conformance.

The native entropy decoder is derived from the official JPEG AI reference
software under its BSD license. That license explicitly does not grant patent
rights.
