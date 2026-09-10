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

## Recent learned-compression directions (2024–2026)

The research frontier is not one race with one winner. Each branch optimizes a
different combination of fidelity, perceived quality, latency, model size,
adaptability, and interoperability. Results below are reported by the paper
authors on their own datasets and hardware; they are not directly comparable to
one another or to this repository's one-image smoke test.

| Direction | Representative work | What changed | Production trade-off |
| --- | --- | --- | --- |
| Hardware-aware perceptual coding | Apple's PICO (CVPR 2026) | Searches millions of network configurations while optimizing both perceived quality and on-device runtime. The authors report 2.3–3x bitrate savings in subjective tests against AV1, AV2, VVC, ECM, and JPEG AI, plus 230 ms encode and 150 ms decode for a 12 MP image on an iPhone 17 Pro Max. | The numbers are especially relevant to this Apple port, but they measure a particular perceptual target, device, and compiler setup. They do not establish equal-PSNR superiority or standards-level interoperability. |
| Faster learned transforms and probability models | LALIC and dictionary-based entropy coding (CVPR 2025), Cassic (ICCV 2025), and sparse-attention/adaptive-frequency coding (CVPR 2026) | Uses linear attention, learned dictionaries of common structures, content-dependent scan orders, and adaptive spatial/frequency paths to predict latents with less serial or redundant work. | These are promising research codecs, but latency claims must be repeated using the same resolution, device, software stack, and complete entropy-coding path. |
| Per-image or per-video fitting | C3 (CVPR 2024), Wasserstein-C3 (CVPR 2025), and FNLIC (CVPR 2025) | Optimizes a small representation or model for each asset rather than relying only on one large general decoder. C3 reports VTM-like image rate-distortion with under 3,000 decode MACs per pixel; FNLIC applies the idea to lossless coding. | Encoding can require substantial optimization. This asymmetry is attractive for archives and repeatedly served media, but less suitable for instant camera capture. Model/parameter bits must be counted in the payload. |
| Generative and diffusion decoding | MRIDC (CVPR 2025) and DiT-IC (CVPR 2026) | Reconstructs visually plausible detail at extremely low rates; DiT-IC reduces a multi-step diffusion decoder to a latent, single-step design and reports up to 30x faster decoding than earlier diffusion codecs. | Plausible detail is not necessarily the original detail. These codecs need explicit hallucination and task-safety evaluation and are a poor default for medical, scientific, legal, or evidentiary images. |
| Progressive learned coding | Variance-aware masking (WACV 2025) | Sends a base latent first, then importance-ranked residual elements so the reconstruction improves as more bytes arrive. | Useful for previews and variable networks, but it needs a stable scalable bitstream and careful intermediate-quality testing. |
| Compression for machines | JPEG AI compressed-domain inference and the Visual Token Codec preprint (August 2026) | Compresses latents or vision-transformer tokens for downstream segmentation, detection, or distributed inference instead of always reconstructing RGB pixels. | Rate must be evaluated against task accuracy, not only PSNR. The Visual Token Codec is a new preprint, and model/feature compatibility remains an open deployment constraint. |
| Learned temporal coding | MobileNVC (WACV 2024) and DCVC-RT (CVPR 2025) | Exploits motion and temporal context and maps work onto practical hardware. DCVC-RT reports 125.2/112.8 fps encode/decode for 1080p on an A100 and 21% average bitrate savings against H.266/VTM. | These are hardware-specific research results. Random access, rate control, error recovery, power, and interoperable decoders remain decisive for production video. |

The strongest near-term lesson for this project is not to copy every new model.
It is to make the benchmark multidimensional. PICO belongs in the Apple-device
comparison if runnable codec artifacts become available. C3 is a useful test of
slow-encode/cheap-decode asymmetry. A generative codec belongs in a separate
perceptual experiment with hallucination checks, not on the same chart as a
fidelity codec without an explicit warning. Machine-facing approaches need task
accuracy and transmitted-feature size alongside image metrics.

The practical scorecard is therefore: payload bits, fidelity, human preference,
task accuracy, warm and cold latency, energy, memory, model/package size,
determinism, and bitstream interoperability. A method is "superior" only after
the intended product assigns weights to those dimensions.

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
- [Apple PICO: practical learned image compression (CVPR 2026)](https://machinelearning.apple.com/research/compression)
- [C3: per-image and per-video neural compression (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_C3_High-Performance_and_Low-Complexity_Neural_Compression_from_a_Single_Image_CVPR_2024_paper.html)
- [Good, Cheap, and Fast: Wasserstein-C3 (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Balle_Good_Cheap_and_Fast_Overfitted_Image_Compression_with_Wasserstein_Distortion_CVPR_2025_paper.html)
- [Fitted Neural Lossless Image Compression (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Fitted_Neural_Lossless_Image_Compression_CVPR_2025_paper.html)
- [Dictionary-based learned entropy model (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Learned_Image_Compression_with_Dictionary-based_Entropy_Model_CVPR_2025_paper.html)
- [LALIC linear-attention compression (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_Linear_Attention_Modeling_for_Learned_Image_Compression_CVPR_2025_paper.html)
- [Cassic content-adaptive state-space compression (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/html/Qin_Cassic_Towards_Content-Adaptive_State-Space_Models_for_Learned_Image_Compression_ICCV_2025_paper.html)
- [Sparse-attention and adaptive-frequency compression (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_Learned_Image_Compression_via_Sparse_Attention_and_Adaptive_Frequency_CVPR_2026_paper.html)
- [MRIDC region-adaptive diffusion compression (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_Decouple_Distortion_from_Perception_Region_Adaptive_Diffusion_for_Extreme-low_Bitrate_CVPR_2025_paper.html)
- [DiT-IC single-step diffusion compression (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Shi_DiT-IC_Aligned_Diffusion_Transformer_for_Efficient_Image_Compression_CVPR_2026_paper.html)
- [Variance-aware progressive learned compression (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/html/Presta_Efficient_Progressive_Image_Compression_with_Variance-Aware_Masking_WACV_2025_paper.html)
- [Visual Token Codec preprint (August 2026)](https://arxiv.org/abs/2608.08832)
- [MobileNVC mobile neural video compression (WACV 2024)](https://openaccess.thecvf.com/content/WACV2024/html/van_Rozendaal_MobileNVC_Real-Time_1080p_Neural_Video_Compression_on_a_Mobile_Device_WACV_2024_paper.html)
- [DCVC-RT real-time neural video compression (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.html)
- [JPEG XL overview](https://jpeg.org/jpegxl/)
- [Jpegli source and design summary](https://github.com/google/jpegli)
- [AOMedia AVIF overview and specification](https://aomedia.org/specifications/avif/)
- [AOMedia AV2 v1.0 specification](https://av2.aomedia.org/v1.0.0/20260528_38f28e7_AV2_Spec_v1.0.0.pdf)
- [JPEG XS overview](https://jpeg.org/jpegxs/)
- [MobileNVC paper](https://openaccess.thecvf.com/content/WACV2024/html/van_Rozendaal_MobileNVC_Real-Time_1080p_Neural_Video_Compression_on_a_Mobile_Device_WACV_2024_paper.html)
- [DCVC-RT paper](https://openaccess.thecvf.com/content/CVPR2025/html/Jia_Towards_Practical_Real-Time_Neural_Video_Compression_CVPR_2025_paper.html)
