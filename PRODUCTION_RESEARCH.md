# JPEG AI production research

Research snapshot: 10 September 2026.

## Conclusion

JPEG AI is technically important: it is the first international image coding
standard built around an end-to-end learned codec, and it exposes a compact
representation for both reconstructed images and machine-vision tasks. It is
not yet the universal replacement for JPEG. Decoder availability, model size,
cold-start latency, conformance, power use, and ecosystem support still matter
more than a result from one image.

The right claim for this project is narrower: the native Apple port proves that
a useful part of the JPEG AI simple profile can run locally on Apple silicon,
produce an interoperable codestream for the tested path, and expose its real
latents for inspection. It does not yet prove full JPEG AI conformance or a
production speed advantage.

## What the current evidence says

At almost the same rate on the current single-image smoke test, JPEG quality 70
used 1.080695 bits per pixel and JPEG AI preset 75 used 1.091779 bits per pixel.
JPEG AI measured 37.040 dB RGB PSNR versus 34.548 dB for JPEG, 0.95358 versus
0.92747 SSIM-Y, and 0.06804 versus 0.07560 LPIPS-Alex (lower LPIPS is better).
That is encouraging, but it is one image and is not a publishable codec-wide
conclusion.

The reference implementation's timing split also explains the apparently huge
latency. In the timing-split run, median codec work was roughly 1.07 seconds to
encode and 0.257 seconds to decode. Model loading added roughly 0.875 and 0.831
seconds, while Python, CUDA initialization, process startup, and I/O added
roughly 3.73 and 3.68 seconds. Cold-start cost and inference cost must therefore
remain separate columns.

For a defensible study, add a fixed lossless corpus and several operating points
for every codec, then report:

- rate-distortion curves rather than equal numeric quality settings;
- RGB and luma PSNR, SSIM, MS-SSIM, LPIPS, and preferably a subjective check;
- cold and warm p50/p95 encode and decode latency;
- peak memory, package/model size, and energy per image;
- exact hardware, OS, implementation commit, model, and compute-unit policy;
- output payload size separately from the lossless PNG preview size.

JPEG, Jpegli, JPEG XL, and AVIF are the minimum useful still-image baselines.
The benchmark should not describe a winner until it has comparable points on
the same corpus.

## Production roadmap for Apple silicon

The order matters. Preserve codestream correctness first, remove lifecycle and
copy overhead second, and only then write custom kernels.

### 1. Remove repeated compilation and loading

`JPEGAICoreMLModelSet` currently calls `MLModel.compileModel(at:)` when a model
is first used and caches the loaded instance only inside the current process.
The GUI benefits from that cache, but a one-image CLI exits and pays the setup
cost again next time.

The first production experiment should persist device-compiled model resources
where supported and add a directory/batch command that keeps one model set
alive for many images. Apple explicitly recommends loading a Core ML model once
and shows compile, load, and prediction as separate measurements in its
[Core ML performance guidance](https://developer.apple.com/videos/play/wwdc2022/10027/).
This should improve cold/warm behavior without changing codec arithmetic.

### 2. Measure the Neural Engine instead of assuming it wins

The current configuration is `.cpuAndGPU`. Apple documents that this excludes
the Neural Engine, while `.all` allows Core ML to select CPU, GPU, and Neural
Engine. Test `.all`, `.cpuAndGPU`, and `.cpuAndNeuralEngine` on each target Mac
and iPhone using the same streams. Use Instruments and `MLComputePlan` to see
where every operation actually runs.

Do not promise that the Neural Engine is fastest. JPEG's April 2026 experiments
found FPGA most energy-efficient and GPU lowest-latency for the implementations
they tested. Hardware, tensor shapes, and hand-off costs can change the answer.

### 3. Stop copying tensors through Swift arrays

The current path repeatedly converts `[Float32]`, `[Int8]`, and `MLMultiArray`
values. It also materializes intermediate arrays with operations such as
`flatMap`, concatenation, `zip`, shuffle, and color conversion. Profile these
before changing the neural graphs.

Apple recommends Float16 inputs/outputs, preallocated output backing buffers,
and IOSurface-backed buffers to avoid data transformations and cross-device
copies. The practical target is a reusable buffer pipeline from image decode to
Core ML to `CGImage`/`CVPixelBuffer`, with PNG export outside the timed decoder
region.

### 4. Specialize the shapes that production actually uses

The converted models accept broad flexible dimensions. Core ML can optimize a
finite set of enumerated shapes more aggressively, and bounded ranges are more
optimizable than unbounded ones. Start with common photo sizes or standard
tiles, not every possible dimension.

JPEG AI itself supports tiling and spatial access, but this port currently
supports only untiled 4:4:4 simple-profile streams. Implement standard-compliant
tiling before inventing a private framing format. Tiles can bound memory and
create safe parallel work; context-dependent entropy stages cannot simply be
run out of order.

### 5. Reduce the model package carefully

Core ML Tools supports weight quantization, palettization, and pruning. Test
8-bit weights first, then 6- or 4-bit palettization only if the model-size gain
is worth the validation work. A smaller package can improve download, storage,
and load behavior, but it is not automatically a faster codec.

For a learned codec, normal "accuracy stayed close" validation is insufficient.
Numerical changes near quantization boundaries can alter entropy symbols and
the codestream. Every compressed model candidate must repeat:

1. native-versus-reference control-point hashes;
2. native stream decoding in the official decoder;
3. official stream decoding in the native decoder;
4. rate-distortion, perceptual-quality, and timing runs over the corpus.

### 6. Use Metal only for a measured hotspot

Do not replace Core ML graphs wholesale. First profile shuffles, color
conversion, quantization/dequantization, masks, and me-tANS. A small Metal
kernel is justified only when Instruments shows that one of those stages is a
material bottleneck and a native/vectorized CPU implementation is insufficient.
The four luma context-model stages are sequential by design; the independent
luma/chroma work around them is the safer source of concurrency.

### 7. Keep Core AI as a newer optional backend

Apple's 2026 Core AI framework adds ahead-of-time compilation, persistent
device specialization, preallocated tensors, pipelined inference, custom GPU
kernels, and a tensor debugger. It directly addresses several limitations in
this prototype. It is also beta technology and requires newer systems and
Apple-silicon hardware.

Keep Core ML as the macOS 13/iOS 16 compatibility backend. Prototype Core AI
only behind a newer deployment target, and adopt it after its bitstream output,
quality, cold/warm latency, memory, and energy are independently validated.

### 8. Production gates

Before describing the codec as production-ready:

- pass official conformance streams when JPEG AI Part 4 material is available;
- add malformed/truncated-stream fuzzing and strict dimension/allocation limits;
- finish chroma subsampling, optional tools, tiling, metadata, and file-format
  support required by the chosen profile;
- test bit-exact behavior across supported Apple chips and OS releases;
- sign/notarize the app and make model/version compatibility explicit;
- review the standard's patent position separately from its source-code
  licenses.

## Can the idea extend to video?

Yes, but encoding each frame independently is only an intra-frame video codec.
It misses most temporal redundancy. A practical learned video codec also needs
reference-frame reconstruction, motion or temporal context, rate control,
random access, error recovery, bounded memory, and deterministic cross-device
decoding.

JPEG's January 2026 plan explicitly added a video streaming/storage use case in
which JPEG AI acts as a deterministic still-image engine inside a video
pipeline. That makes this port useful first for keyframes, all-intra editing,
screen-share snapshots, and low-latency experiments. It is not yet a competitor
to a complete AV1, AV2, HEVC, or VVC video stack.

Research is closing the runtime gap. MobileNVC demonstrated greater than 30 fps
1080p neural decoding on a mobile device by splitting work across neural,
graphics, and motion hardware. DCVC-RT reported 125.2/112.8 fps encode/decode at
1080p on an A100 and an average 21% bitrate saving against H.266/VTM. Those are
promising research results on their stated hardware, not evidence that this
Apple implementation will reach the same numbers.

## Current alternatives and adjacent work

"Superior" depends on the job:

| Need | Strong current direction | Why it matters |
| --- | --- | --- |
| Existing `.jpg` compatibility | Jpegli | A libjpeg62-compatible encoder/decoder with improved perceptual quantization and floating-point processing; low migration cost. |
| Modern still images | JPEG XL or AVIF | Mature formats with lossless/lossy modes and features such as HDR; JPEG XL also supports progressive coding and lossless JPEG recompression. |
| Learned still-image coding and machine-readable latents | JPEG AI | International learned-coding standard with a single compressed representation for human and machine consumption; ecosystem is still early. |
| Visually lossless live links | JPEG XS | Standardized, line-scale latency and low implementation complexity, but designed for much lower compression ratios than distribution codecs. |
| Next-generation conventional video | AV2 | AOMedia published the AV2 v1.0 bitstream specification in May 2026; deployment and hardware support will take time. |
| Learned video research | MobileNVC and DCVC-RT | Shows that mobile/real-time learned video is possible, but interoperability and widespread hardware support lag conventional standards. |
| Newer Apple-only inference runtime | Core AI | AOT compilation, specialization cache, profiler/debugger, and custom kernels; currently beta and a runtime rather than a codec. |

JPEG AI itself is still moving. In 2026 the committee reported work on
bit-exact reconstruction, energy across CPU/GPU/FPGA, RGB retraining, mobile
encoders/decoders, error resilience, compressed-domain analysis, and stronger
test conditions. Its reported earth-observation experiment used JPEG AI
latents for segmentation with nearly ten times fewer model parameters. That
compressed-domain path—not only smaller pictures—may become its most distinct
production advantage.

## How to contribute upstream

There are two different contribution paths.

### Reference-software code

The official source is public on
[GitLab](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software). Fork it,
create a focused branch, run `make unittest` plus the relevant codec tests, and
open a merge request. Discuss a large Apple-platform proposal in an issue or
small design merge request before moving generated models or a full UI.

The most reviewable upstream sequence from this project is:

1. me-tANS encoder/decoder code with official fixture hashes and bounds tests;
2. reproducible ONNX-to-Core-ML conversion and validation scripts;
3. small interoperability fixtures and documented Apple build instructions;
4. the native Swift codec core;
5. the macOS app and Homebrew tap only if maintainers want distribution code in
   the reference repository.

Do not add the large generated Core ML model packages to Git unless maintainers
explicitly choose an artifact policy. Keep commits small and include the exact
reference commit, model, and control-point hashes.

This repository is MIT licensed and preserves the official BSD notice for
derived entropy code. Before copying new MIT files upstream, ask the maintainers
which copyright headers and contribution terms they require and explicitly
offer the code under their accepted license. The reference source license says
that it grants no patent rights, so a merged open-source implementation is not
by itself a commercial patent clearance.

### The JPEG AI standard

A merge request improves software; it does not change the international
standard. JPEG says standards experts participate through their ISO/IEC
National Body. Contact the National Body for your country and ask to join the
JPEG AI ad-hoc group if you want to submit experiments, requirements, or
normative changes. At this snapshot, the next listed meeting is JPEG 113 in
Hangzhou, 18–23 October 2026.

The most useful standards contribution from this work would be reproducible
Apple-device evidence: bit-exactness across chips, cold/warm latency, energy,
peak memory, model footprint, and the exact toolchain used. That aligns with
the committee's current mobile, energy, and bit-exact core experiments.

## Primary sources

- [JPEG AI overview](https://jpeg.org/jpegai/)
- [JPEG AI workplan and specification status](https://jpeg.org/jpegai/workplan.html)
- [JPEG 110 meeting: mobile, video, energy, and error-resilience plans](https://jpeg.org/items/20260328_press.html)
- [JPEG 111 meeting: hardware trade-offs and compressed-domain analysis](https://jpeg.org/items/20260608_press.html)
- [Official JPEG participation route](https://jpeg.org/participate.html)
- [Official JPEG AI reference software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software)
- [Apple Core ML compute-unit policy](https://developer.apple.com/documentation/coreml/mlcomputeunits)
- [Apple Core ML performance and data-flow guidance](https://developer.apple.com/videos/play/wwdc2022/10027/)
- [Core ML flexible-shape guidance](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
- [Core ML model-compression guidance](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- [Apple Core AI specialization and caching](https://developer.apple.com/documentation/coreai/managing-model-specialization-and-caching)
- [JPEG XL overview](https://jpeg.org/jpegxl/)
- [Jpegli source and design summary](https://github.com/google/jpegli)
- [AOMedia AVIF overview and specification](https://aomedia.org/specifications/avif/)
- [AOMedia AV2 v1.0 specification](https://av2.aomedia.org/v1.0.0/20260528_38f28e7_AV2_Spec_v1.0.0.pdf)
- [JPEG XS overview](https://jpeg.org/jpegxs/)
- [MobileNVC paper](https://openaccess.thecvf.com/content/WACV2024/html/van_Rozendaal_MobileNVC_Real-Time_1080p_Neural_Video_Compression_on_a_Mobile_Device_WACV_2024_paper.html)
- [DCVC-RT paper](https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.html)
