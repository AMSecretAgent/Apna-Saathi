#!/usr/bin/env bash
# Set up a local working copy: fetch the Suno Sutra platform, apply our fixes to
# it, and overlay our application on top.
#
#   ./setup.sh              platform source only  (fast, ~20MB)
#   ./setup.sh --with-lfs   also pull the ~2.6GB model archives, which you only
#                           need if you are going to flash or provision a board
#
# Safe to re-run. Nothing here touches a device.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$HERE/vendor/suno-sutra-sw"
UPSTREAM="https://github.com/currentai-org/suno-sutra-sw.git"
WITH_LFS=0
[ "${1:-}" = "--with-lfs" ] && WITH_LFS=1

echo "==> platform source"
if [ -d "$VENDOR/.git" ]; then
  echo "    already present at vendor/suno-sutra-sw"
else
  mkdir -p "$HERE/vendor"
  # Skip the LFS smudge by default. The repository LFS-tracks ~2.6GB of model
  # archives that are only needed to provision a board from scratch; a clone
  # without them still contains all the source you need to develop against.
  GIT_LFS_SKIP_SMUDGE=1 git clone "$UPSTREAM" "$VENDOR"
fi

if [ "$WITH_LFS" = "1" ]; then
  echo "==> pulling LFS model archives (~2.6GB, slow)"
  git -C "$VENDOR" lfs pull
  # Anything still 135 bytes is an unresolved pointer, not a model.
  sz=$(stat -c%s "$VENDOR/rootfs/roles/indic/files/bhashini_models.zip" 2>/dev/null || echo 0)
  if [ "$sz" -lt 1000000 ]; then
    echo "    WARNING: bhashini_models.zip is $sz bytes -- LFS did not resolve."
    echo "    Install git-lfs and re-run, or provisioning will silently fail."
  fi
fi

echo "==> applying our platform fixes"
for p in "$HERE"/patches/*.patch; do
  name="$(basename "$p")"
  if git -C "$VENDOR" apply --check "$p" 2>/dev/null; then
    git -C "$VENDOR" apply "$p"
    echo "    applied  $name"
  elif git -C "$VENDOR" apply --reverse --check "$p" 2>/dev/null; then
    echo "    already applied  $name"
  else
    echo "    SKIPPED  $name  (does not apply -- upstream may have changed)"
  fi
done

echo "==> overlaying our application"
cp "$HERE/pocketinfer/retrieval.py"   "$VENDOR/python/pocketinfer/"
cp "$HERE/pocketinfer/textshape.py"   "$VENDOR/python/pocketinfer/"
cp "$HERE/pocketinfer/applications/apna_saathi.py" "$VENDOR/python/pocketinfer/applications/"
mkdir -p "$VENDOR/python/pocketinfer/data" "$VENDOR/python/tools"
cp "$HERE/pocketinfer/data/corpus.tsv"  "$VENDOR/python/pocketinfer/data/"
cp "$HERE/pocketinfer/data/eval.jsonl"  "$VENDOR/python/pocketinfer/data/"
for d in canned eval_wav; do
  if compgen -G "$HERE/pocketinfer/data/$d/*.wav" > /dev/null; then
    mkdir -p "$VENDOR/python/pocketinfer/data/$d"
    cp "$HERE/pocketinfer/data/$d/"*.wav "$VENDOR/python/pocketinfer/data/$d/"
  fi
done
cp "$HERE/tools/"*.py "$VENDOR/python/tools/" 2>/dev/null || true
echo "    done"

echo "==> self-checks (no hardware, no models, no network)"
python3 "$HERE/pocketinfer/retrieval.py"
python3 "$HERE/pocketinfer/textshape.py"
python3 "$HERE/tools/eval_apna_saathi.py" --retrieval-only

cat <<'EOF'

Ready.

  Next, with a device reachable over SSH or Tailscale:
    export DEV=ubuntu@<name-or-ip>          # password: ubuntu
    less docs/BOARD_BRINGUP.md              # ten ordered steps, ~40 min
    ./tools/deploy.sh $DEV

  Without a device, everything below still runs:
    python3 tools/eval_apna_saathi.py --retrieval-only -v
    python3 tools/eval_apna_saathi.py --sweep-threshold

  The platform source is in vendor/suno-sutra-sw (gitignored, not ours to commit).
EOF
