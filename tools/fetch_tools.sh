#!/usr/bin/env bash
# Download and verify FreeCAD + ParaView portable binaries (Linux/macOS variant).
# Linux users typically install freecad + paraview via apt/brew instead -- this
# script exists for parity with the Windows workflow. See the Dockerfile for the
# apt-package approach.
#
# Reads tools/MANIFEST.sha256 and fetches each entry from the GitHub Release.
# Verifies SHA256 before extracting. Exits non-zero on any failure.
#
# Usage (from the repo root):
#   bash tools/fetch_tools.sh
#   RELEASE_TAG=v0.1-tools bash tools/fetch_tools.sh
#   FORCE=1 bash tools/fetch_tools.sh   # re-download even if already extracted

set -euo pipefail

RELEASE_TAG="${RELEASE_TAG:-v0.1-tools}"
REPO_SLUG="${REPO_SLUG:-JoepVdE/LTS_Rutherford_Workflow_shared}"
FORCE="${FORCE:-0}"

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$TOOLS_DIR/MANIFEST.sha256"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

# Pick a sha256 binary
if command -v sha256sum >/dev/null 2>&1; then
    SHA_CMD=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
    SHA_CMD=(shasum -a 256)
else
    echo "Need sha256sum or shasum to verify downloads." >&2
    exit 1
fi

# Pick a downloader
if command -v curl >/dev/null 2>&1; then
    DL_CMD=(curl -L --fail --progress-bar -o)
elif command -v wget >/dev/null 2>&1; then
    DL_CMD=(wget -O)
else
    echo "Need curl or wget to download." >&2
    exit 1
fi

# Pick an extractor
if ! command -v unzip >/dev/null 2>&1; then
    echo "Need 'unzip' to extract archives." >&2
    exit 1
fi

echo "Fetching tools from release '$RELEASE_TAG' of $REPO_SLUG"

while IFS= read -r line; do
    line="${line#"${line%%[![:space:]]*}"}"   # ltrim
    [[ -z "$line" || "$line" == \#* ]] && continue
    # Format: <sha256>  <filename>  <bytes>  <unpack_dir>
    read -r sha filename bytes unpack_dir <<< "$line"
    if [[ -z "$sha" || -z "$filename" || -z "$unpack_dir" ]]; then
        echo "Skipping malformed manifest line: $line" >&2
        continue
    fi
    extracted="$TOOLS_DIR/$unpack_dir"
    if [[ -d "$extracted" && -n "$(ls -A "$extracted" 2>/dev/null)" && "$FORCE" != "1" ]]; then
        echo "[$unpack_dir] already populated, skipping (set FORCE=1 to redo)."
        continue
    fi

    url="https://github.com/$REPO_SLUG/releases/download/$RELEASE_TAG/$filename"
    tmp_zip="${TMPDIR:-/tmp}/lts_rutherford_$filename"

    echo "[$unpack_dir] downloading $filename ($(( bytes / 1024 / 1024 )) MB)..."
    echo "  URL: $url"
    rm -f "$tmp_zip"
    "${DL_CMD[@]}" "$tmp_zip" "$url"

    echo "[$unpack_dir] verifying SHA256..."
    got="$("${SHA_CMD[@]}" "$tmp_zip" | awk '{print tolower($1)}')"
    expected="$(echo "$sha" | tr 'A-Z' 'a-z')"
    if [[ "$got" != "$expected" ]]; then
        rm -f "$tmp_zip"
        echo "SHA256 mismatch for $filename: expected $expected, got $got." >&2
        exit 1
    fi

    echo "[$unpack_dir] extracting..."
    rm -rf "$extracted"
    unzip -q -o "$tmp_zip" -d "$TOOLS_DIR"
    rm -f "$tmp_zip"

    if [[ ! -d "$extracted" ]]; then
        echo "Extraction completed but expected directory missing: $extracted" >&2
        exit 1
    fi
    echo "[$unpack_dir] OK"
done < "$MANIFEST"

echo "All tools ready under: $TOOLS_DIR"
