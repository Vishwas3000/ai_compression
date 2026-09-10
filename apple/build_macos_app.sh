#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
resources="$root/dist/JPEGAIDecoder.app/Contents/Resources"
executable="$root/dist/JPEGAIDecoder.app/Contents/MacOS"

for table in unique_z_distributions.csv residual_transitions.csv residual_bounds.csv \
    residual_encode_transitions.csv residual_state_maps.csv \
    Y_0.002.csv UV_0.002.csv Y_0.002_gain.csv UV_0.002_gain.csv \
    Y_0.012.csv UV_0.012.csv Y_0.012_gain.csv UV_0.012_gain.csv \
    Y_0.075.csv UV_0.075.csv Y_0.075_gain.csv UV_0.075_gain.csv \
    Y_0.5.csv UV_0.5.csv Y_0.5_gain.csv UV_0.5_gain.csv; do
    if [ ! -f "$root/Models/$table" ]; then
        echo "Missing codec table: $root/Models/$table" >&2
        exit 1
    fi
done
test -d "$root/Models/apple-coreml-simple"
swift build --package-path "$root" -c release --product JPEGAIDecoder

# ponytail: overwrite this generated bundle in place; add clean builds if stale assets become a problem.
mkdir -p "$resources/Tables" "$executable"
cp "$root/.build/release/JPEGAIDecoder" "$executable/JPEGAIDecoder"
cp "$root/App/Info.plist" "$root/dist/JPEGAIDecoder.app/Contents/Info.plist"
cp "$root"/Models/*.csv "$resources/Tables/"
ditto "$root/Models/apple-coreml-simple" "$resources/apple-coreml-simple"
codesign --force --deep --sign - "$root/dist/JPEGAIDecoder.app"
codesign --verify --deep --strict "$root/dist/JPEGAIDecoder.app"
echo "$root/dist/JPEGAIDecoder.app"
