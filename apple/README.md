# Native Apple JPEG AI decoder

This Swift package targets macOS 13 and iOS 16 using Core ML with CPU and GPU
compute. The current simple-profile path provides:

- JPEG AI marker and picture-header parsing
- native me-tANS factorized and SGM entropy decoding
- model-table selection plus hyper- and residual-latent decoding
- loading converted decoder stages as Core ML models
- MCM latent reconstruction, synthesis, BT.709 conversion, and RGB PNG output

The current decoder covers untiled, 4:4:4 simple-profile codestreams. Chroma
subsampling and the optional enhancement/filter tools are not implemented yet.

## Check

```bash
cd apple
swift test
```

Inspect a codestream or decode it to PNG:

```bash
swift run jpegai-info INPUT.bits [TABLES_DIR [COREML_MODELS_DIR]]
swift run jpegai-info INPUT.bits TABLES_DIR COREML_MODELS_DIR OUTPUT.png
```

## macOS app

Build a self-contained, locally signed app from the tables and models under
`Models/`, then double-click it or launch it from Terminal:

```bash
./build_macos_app.sh
open dist/JPEGAIDecoder.app
```

The app opens a `.bits` picker, asks where to save the decoded PNG, and displays
the result in its window. The generated 72 MB app stays under ignored `dist/`;
the local build includes tables for all four simple-profile models. Distribution
to other Macs requires Developer ID signing and notarization.

`TABLES_DIR` contains `unique_z_distributions.csv` and the matching `Y_*.csv`
and `UV_*.csv` mappings. PyTorch 2.11 requires the included legacy-export patch
before exporting the reference graphs:

```bash
git -C JPEG_AI_REFERENCE apply --unidiff-zero \
  "$PWD/patches/jpeg-ai-pytorch-2.11-onnx.patch"
```

Convert exported ONNX decoder graphs with:

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

The native entropy decoder is derived from the official JPEG AI reference
software under its BSD license. That license explicitly does not grant patent
rights.
