# Native Apple JPEG AI decoder

This Swift package targets macOS 13 and iOS 16 using Core ML with CPU and GPU
compute. The current simple-profile path provides:

- JPEG AI marker and picture-header parsing
- native me-tANS factorized and SGM entropy decoding
- model-table selection and hyper-latent decoding
- loading converted decoder stages as Core ML models

It does not yet reconstruct residual latents or output RGB pixels.

## Check

```bash
cd apple
swift test
```

Inspect a codestream and optionally decode its hyper-latents:

```bash
swift run jpegai-info INPUT.bits [TABLES_DIR [COREML_MODELS_DIR]]
```

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

The native entropy decoder is derived from the official JPEG AI reference
software under its BSD license. That license explicitly does not grant patent
rights.
