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

The native decoder reproduces every entropy control-point hash from the
official decoder for the 560x888 model-2 benchmark stream. Its RGB output is
63.988 dB PSNR versus the official PNG (9,920 differing channel samples out of
1,491,840). The remaining visible-data error is confined to the synthesis
graph's bottom edge because the exported ONNX graph omits the reference
decoder's runtime height crop; above that 12-row band, PSNR is 82.227 dB.

The native entropy decoder is derived from the official JPEG AI reference
software under its BSD license. That license explicitly does not grant patent
rights.
