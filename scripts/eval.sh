#!/usr/bin/env bash
# Generate a batch of games in parallel to measure the generator.
#
# Usage: scripts/eval.sh [N] [CONFIG] [OUT_DIR]
#   N        games to generate (default 4)
#   CONFIG   game config file (default game.example.yml)
#   OUT_DIR  where each game goes, one folder per game (default eval/<timestamp>)
#
# Each game gets OUT_DIR/gameN/ and OUT_DIR/gameN.log. Review the games with the
# rubric in docs/EVAL_RUBRIC.md: one reviewer per game, then compare the SUMMARY
# blocks with earlier batches.
set -euo pipefail

count="${1:-4}"
config="${2:-game.example.yml}"
out="${3:-eval/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$out"

# WeasyPrint finds Homebrew's libraries only through this on macOS
if [[ "$(uname)" == "Darwin" ]]; then
  export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/lib}"
fi

for n in $(seq 1 "$count"); do
  (
    start=$(date +%s)
    status=0
    uv run mystery-agents "$config" --no-images --keep-work-dir --output-dir "$out/game$n" \
      >"$out/game$n.log" 2>&1 || status=$?
    echo "EXIT $status ELAPSED $(($(date +%s) - start))s" >>"$out/game$n.log"
  ) &
  # Staggered starts keep the first calls from hitting the rate limit together
  sleep 5
done
wait

for n in $(seq 1 "$count"); do
  printf 'game%s: %s  %s\n' "$n" "$(tail -1 "$out/game$n.log")" \
    "$(ls -d "$out/game$n"/game_* 2>/dev/null || echo 'no game folder')"
done
echo "Rubric: docs/EVAL_RUBRIC.md"
