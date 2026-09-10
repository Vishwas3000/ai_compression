#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
version=${1:-0.1.0}
archive="$root/dist/jpeg-ai-apple-models-$version.tar.gz"

case "$version" in
    *[!0-9.]* | .* | *.)
        echo "Version must contain only dot-separated numbers." >&2
        exit 2
        ;;
esac

"$root/build_macos_app.sh" >/dev/null
swift build --package-path "$root" --disable-sandbox -c release --product jpegai-info
test "$("$root/.build/release/jpegai-info" --version)" = "jpeg-ai $version"

staging=$(mktemp -d "${TMPDIR:-/tmp}/jpeg-ai-homebrew.XXXXXX")
trap 'rm -rf -- "$staging"' EXIT HUP INT TERM
mkdir -p "$staging/Tables" "$staging/Models" "$root/dist"
ditto "$root/dist/JPEGAIDecoder.app/Contents/Resources/Tables" "$staging/Tables"
ditto "$root/dist/JPEGAIDecoder.app/Contents/Resources/apple-coreml-simple" "$staging/Models"
cp "$root/../LICENSE" "$root/THIRD_PARTY_NOTICES.md" "$staging/"
COPYFILE_DISABLE=1 tar -czf "$archive" -C "$staging" \
    Tables Models LICENSE THIRD_PARTY_NOTICES.md

echo "$archive"
shasum -a 256 "$archive"
