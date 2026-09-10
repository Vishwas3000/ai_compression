---
title: "I ported JPEG AI to Apple silicon—and the first benchmark was mostly startup"
published: false
description: "A native Core ML JPEG AI encoder/decoder, real y and z activation maps, and why a five-second result did not mean the codec was slow."
cover_image: "https://raw.githubusercontent.com/Vishwas3000/ai_compression/main/blog/assets/jpeg-ai-apple-cover.png"
tags: ai, performance, macos, opensource
---

My first JPEG AI decode result took almost five seconds. That number looked
terrible next to JPEG—until I measured what those seconds contained.

Only about 0.26 seconds was the reference decoder's codec operation. Model
loading took about 0.83 seconds. Roughly 3.68 seconds went to Python imports,
CUDA initialization, process startup, and I/O.

That distinction became the point of this project: compare JPEG and JPEG AI,
but do it in a way that does not confuse neural inference with cold-start
machinery. Then take the same codec path to Apple silicon using Core ML and a
native entropy coder.

The code, benchmark harness, results, and macOS app are in the
[ai_compression repository](https://github.com/Vishwas3000/ai_compression).

## What JPEG AI actually is

JPEG AI is not “run a chatbot on a JPEG.” It is a learned image codec. A neural
analysis transform replaces the hand-designed frequency transform and
quantization pipeline familiar from conventional JPEG. A standards-defined
bitstream still carries the result, and a decoder reconstructs pixels with a
neural synthesis transform.

The JPEG committee lists the core coding system as
[ISO/IEC 6048-1:2025](https://jpeg.org/jpegai/workplan.html). Its
[reference software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software)
contains encoder, decoder, training, and evaluation code. The implementation is
available under its included BSD license, although that license explicitly does
not grant patent rights. “Open source implementation” and “unrestricted codec
patent rights” are different claims.

## From pixels to `y` and `z`

The simple-profile encoder used here follows this path:

```text
RGB pixels
   ↓ BT.709 YUV conversion
analysis transform
   ↓
y latent ──────────────┐
   ↓ hyper-encoder     │
z hyper-latent         │
   ↓ quantize + code   │
hyper decoders         │
   ↓ scale, mask, ψ    │
four-stage context model
   ↓ predict y + quantize residual
me-tANS entropy coder
   ↓
JPEG AI bitstream
```

`y` is the main learned representation. For the luma model in this test it has
160 channels at roughly one-sixteenth the input width and height. It retains
spatial content, but its channels are not red, green, blue, or named features.
The network learns whatever representation minimizes its training objective.

`z` is a lower-resolution hyper-latent, roughly one-sixty-fourth the input
dimensions. It describes statistics that help the decoder predict and entropy
code `y`: which values are likely, which positions should be coded, and how to
reconstruct the latent in context. Spending a small number of bits on `z` can
make the much larger `y` stream cheaper.

![Input luma, y activation energy, and z activation energy captured during one Core ML encode](https://raw.githubusercontent.com/Vishwas3000/ai_compression/main/blog/assets/latent-spaces.png)

These are real tensors captured at Core ML model boundaries. The `y` energy
map still resembles the bird and its surroundings. The `z` map is much coarser
and sparser. That is expected: it describes coding context, not a thumbnail.

The app can also export every one of the 160 luma `y` channels, every rounded
`z` channel, and entropy-mask density. Orange and blue in the channel sheets
mean positive and negative activation. Each channel is normalized separately,
so those colors must not be interpreted as directly comparable magnitudes.

For the fixed network topology, I use [Netron](https://netron.app/) to open an
ONNX file such as `apple/Models/onnx/tools_2/model_y/analysis.onnx`. Netron shows
operators, weights, shapes, and connections. It does not show the values from a
particular inference. The app's tensor export provides that second half.

## How the model is trained

Training and inference are separate phases. During training, the encoder and
decoder weights are optimized against a rate-distortion objective commonly
written as:

```text
loss = distortion + β × rate
```

Distortion penalizes reconstruction error. Rate estimates the bits needed for
the latents. A larger `β` makes bits more expensive; a smaller one allows a
larger stream to preserve more detail.

The reference recipe trains four base rate-distortion models with beta values
0.002, 0.012, 0.075, and 0.5. Gain parameters and a beta displacement expose
intermediate operating points without shipping a completely separate network
for every bitrate.

The current reference documentation describes 132 optimization epochs per
base model, followed by a statistics stage. It progresses from fixed-rate MSE
training, through mixed MSE/MS-SSIM distortion, to decoder-side variable-rate
training. The last pass records activation-clipping statistics needed by the
codec. See the official
[training recipe](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/main/docs/architecture/13-training.md)
and [network component guide](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/-/blob/main/docs/architecture/08-neural-network-components.md).

During ordinary encoding or decoding, none of that training happens. The
weights are frozen. Inference is a sequence of tensor transforms plus
quantization and entropy coding.

## The first rate-distortion result

I ran conventional JPEG at qualities 30, 50, 70, and 90, and JPEG AI at five
reference operating points. The source was one 560×888 image from the JPEG AI
test set.

![Rate-distortion plot for JPEG and JPEG AI on one test image](https://raw.githubusercontent.com/Vishwas3000/ai_compression/main/blog/assets/rate-distortion.png)

Near the same rate, JPEG quality 70 produced 1.081 bits per pixel and 34.55 dB
RGB PSNR. JPEG AI point 75 produced 1.092 bits per pixel and 37.04 dB: a 2.49 dB
advantage on this image.

![Original, JPEG, and JPEG AI detail crops at nearly equal rates](https://raw.githubusercontent.com/Vishwas3000/ai_compression/main/blog/assets/same-rate-comparison.png)

This is a smoke result, not a codec-wide conclusion. It has one image, one
measurement per point, no warmup, and GPU contention on the remote machine.
The JPEG baseline also used 4:2:0 chroma while this JPEG AI path used 4:4:4.
A publishable study needs a fixed multi-image corpus, repeated uncontended
runs, confidence intervals, and perceptual metrics in addition to RGB PSNR.

## Why the first timing looked so high

The benchmark preserves total wall time and parses timers printed by the
official reference implementation. That made the hidden startup cost visible.

![JPEG AI encode and decode timing split into codec, model loading, and process overhead](https://raw.githubusercontent.com/Vishwas3000/ai_compression/main/blog/assets/timing-breakdown.png)

Across the five operating points, the component medians were:

| Stage | Codec operation | Model loading | Python/CUDA/process/I/O |
|---|---:|---:|---:|
| Encode | 1.07 s | 0.87 s | 3.73 s |
| Decode | 0.26 s | 0.83 s | 3.68 s |

Starting a new Python process for every image is valid end-to-end latency, but
it is not steady-state codec throughput. A long-running service, application,
or hardware decoder loads models once and amortizes that cost. Both numbers
matter, so the CSV keeps them in separate columns rather than deleting the cold
start.

## Moving the path to Apple silicon

The native implementation converts the reference ONNX graphs to Core ML and
runs them with Apple's compute stack. Swift handles image conversion,
bitstream structure, model orchestration, and reconstruction. A small C++
module ports the reference me-tANS entropy encoder/decoder where exact integer
state transitions matter.

The integer hyper-scale decoder was the sharpest edge. Its output selects an
entropy distribution. A one-index mismatch does not create a slightly different
pixel; it can desynchronize entropy decoding. That stage therefore has to match
the reference exactly, while small floating-point differences in synthesis can
be evaluated with numerical tolerances.

The native encoder produced a 63,599-byte model-2 stream for the test image:
1.023 bits per pixel and 36.51 dB against the source. The official PyTorch
decoder accepted it, and the hyper-latent, scale, mask, and residual control
hashes matched. That verifies bitstream interoperability through entropy
decoding.

The native and official reconstructed PNGs measured 58.37 dB against each
other. Their remaining difference is concentrated at the bottom edge, where
the exported synthesis graph does not yet reproduce the reference decoder's
runtime crop. This is an interoperability milestone, not complete pixel-exact
conformance.

On the Mac, the first native encode took 1.63 seconds and a later encode in the
same process took 0.86 seconds. Those are still provisional single-image
figures, but they show why keeping a model cache inside the app matters.

## Try the macOS app and inspect an inference

Open `apple/Package.swift` in Xcode, select the `JPEGAIDecoder` scheme and **My
Mac**, then press Run. Or build the self-contained local app:

```bash
cd apple
./build_macos_app.sh
open dist/JPEGAIDecoder.app
```

Choose **Encode PNG**, select a rate point, and leave **Export y/z inference
visualizations** enabled. After the encode, choose **Reveal Visualizations**.
The folder is numbered in pipeline order:

1. input luma;
2. all `y` channels;
3. `y` activation energy;
4. all quantized `z` channels;
5. `z` activation energy;
6. entropy-mask density.

The app also performs a verification decode, displays the reconstruction, and
labels the timing as a first or repeat run for that model.

## Could this become a video codec?

Encoding every frame independently would work today, just as Motion JPEG does,
but it would waste temporal redundancy. A competitive video codec needs motion
estimation or learned temporal prediction, reference-frame management, random
access, rate control, error resilience, and hardware-friendly scheduling. The
image model is useful research input, not a drop-in video standard.

## Could JPEG AI become mainstream?

It has two ingredients many learned-codec experiments lack: a published
standards family and a reference bitstream implementation. That gives software
and hardware vendors a common target.

Mainstream adoption still requires fast low-power decoders, stable conformance
tests, manageable model storage, browser and operating-system support, creation
tools, and clear licensing decisions. Conventional JPEG wins today because it
is everywhere and essentially free to decode. JPEG AI must offer enough quality
or functionality to pay for a much more complex decoder.

The next work in this repository is deliberately less glamorous: repair the
remaining crop difference, benchmark a proper corpus with warm caches and no
contention, add an iOS target, and only then prepare an upstream contribution
and public launch post.

The sample image shown here comes from the JPEG AI test material, which the
[JPEG committee identifies as CC0](https://jpeg.org/jpegai/dataset.html).
