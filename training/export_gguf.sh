#!/usr/bin/env bash
# Convert a merged Bibley model to GGUF and shrink it under 1 GB.
#   bash training/export_gguf.sh Bibley-Qwen3-0.6B
# Uses the size_target from models.json (F16, Q8_0 or Q4_K_M).
set -euo pipefail
NAME="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET=$(python3 -c "import json;print([m['size_target'] for m in json.load(open('$HERE/models.json'))['models'] if m['bibley_name']=='$NAME'][0])")
SRC="$HERE/out/$NAME/merged"
DST="$HERE/out/$NAME/gguf"
mkdir -p "$DST"
[ -d "$HERE/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$HERE/llama.cpp"
pip install -q -r "$HERE/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"
python3 "$HERE/llama.cpp/convert_hf_to_gguf.py" "$SRC" --outtype f16 --outfile "$DST/$NAME-F16.gguf"
if [ "$TARGET" != "F16" ]; then
  if [ ! -x "$HERE/llama.cpp/build/bin/llama-quantize" ]; then
    cmake -S "$HERE/llama.cpp" -B "$HERE/llama.cpp/build" -DGGML_NATIVE=OFF >/dev/null
    cmake --build "$HERE/llama.cpp/build" --target llama-quantize -j >/dev/null
  fi
  "$HERE/llama.cpp/build/bin/llama-quantize" "$DST/$NAME-F16.gguf" "$DST/$NAME-$TARGET.gguf" "$TARGET"
  rm "$DST/$NAME-F16.gguf"
fi
cp "$SRC/LICENSE" "$SRC/NOTICE" "$SRC/README.md" "$DST/"
ls -lh "$DST"
SIZE=$(du -b "$DST"/*.gguf | awk '{s+=$1} END {print s}')
[ "$SIZE" -lt 1000000000 ] && echo "OK: under 1 GB" || echo "WARNING: over 1 GB, use a smaller quant"
