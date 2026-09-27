#!/usr/bin/env bash
# Obtain everything the campaign reads from outside this repository.
#
#   vendor/bootstrap.sh robotwin   clone RoboTwin at the pinned revision, fetch
#                                  the asset tree, and symlink it into the
#                                  checkout (the default verb)
#   vendor/bootstrap.sh env        create the conda environment and the three
#                                  editable installs it depends on
#   vendor/bootstrap.sh tokenizer  fetch the PaliGemma tokenizer the checkpoint
#                                  was trained against (gated on Hugging Face)
#   vendor/bootstrap.sh check      verify the layout and the load-bearing patches
#
# Nothing in the Python pipeline fetches any of this: `robotwin_multitask.py`
# asserts that a checkout and an asset tree are already in place and stops with
# instructions if they are not. This script is where they come from.
#
# Overridable: ROBOTWIN_DIR, ASSETS_DIR, ENV_NAME, PYTHON, CONDA.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_DIR="${ROBOTWIN_DIR:-$ROOT/third_party/RoboTwin-official}"
ASSETS_DIR="${ASSETS_DIR:-$ROOT/third_party/RoboTwin-assets/assets}"
ENV_NAME="${ENV_NAME:-taco}"
PYTHON="${PYTHON:-python3}"
CONDA="${CONDA:-conda}"

ROBOTWIN_URL="https://github.com/robotwin-Platform/RoboTwin.git"
# The asset bundle the official RoboTwin's own assets/_download.py fetches.
# ~30 GB extracted.
ASSET_REPO="TianxingChen/RoboTwin2.0"
# LeRobot's default for pi05, which is what the checkpoint was trained against.
TOKENIZER_REPO="google/paligemma-3b-pt-224"

# The revision is pinned in the pipeline, not here, so that "which benchmark is
# this" has exactly one answer and the shell cannot drift from the Python.
pinned_revision() {
    "$PYTHON" -c \
        "import sys; sys.path.insert(0, '$ROOT/scripts'); import robotwin_multitask as m; print(m.ROBOTWIN_REVISION)"
}

log() { printf '==> %s\n' "$*" >&2; }

cmd_robotwin() {
    local revision
    revision="$(pinned_revision)"
    log "pinned RoboTwin revision: $revision"

    if [ ! -e "$ROBOTWIN_DIR/.git" ]; then
        log "cloning $ROBOTWIN_URL -> $ROBOTWIN_DIR"
        mkdir -p "$(dirname "$ROBOTWIN_DIR")"
        git clone "$ROBOTWIN_URL" "$ROBOTWIN_DIR"
    fi

    # Refuse rather than discard: the pipeline requires an unmodified checkout
    # (it only ever symlinks assets in), so a dirty one is someone's work.
    local dirty
    dirty="$(git -C "$ROBOTWIN_DIR" status --porcelain)"
    if [ -n "$dirty" ]; then
        printf 'error: %s has local modifications; refusing to check out over them:\n%s\n' \
            "$ROBOTWIN_DIR" "$dirty" >&2
        exit 1
    fi

    log "fetching and checking out $revision"
    git -C "$ROBOTWIN_DIR" fetch --all --tags --quiet
    git -C "$ROBOTWIN_DIR" checkout --quiet "$revision"

    cmd_assets

    log "linking assets and asserting the revision"
    "$PYTHON" "$ROOT/scripts/robotwin_multitask.py" \
        --robotwin "$ROBOTWIN_DIR" \
        bootstrap --assets "$ASSETS_DIR"
}

cmd_assets() {
    if [ -d "$ASSETS_DIR/embodiments" ] \
        && [ -d "$ASSETS_DIR/objects" ] \
        && [ -d "$ASSETS_DIR/background_texture" ]; then
        log "asset tree already present at $ASSETS_DIR"
        return
    fi
    log "downloading the asset bundle (~30 GB) into $ASSETS_DIR"
    mkdir -p "$ASSETS_DIR"
    "$PYTHON" - "$ASSETS_DIR" "$ASSET_REPO" <<'PY'
import sys
import zipfile
from pathlib import Path

target, repo = Path(sys.argv[1]), sys.argv[2]
from huggingface_hub import snapshot_download

staging = target / "_zips"
snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    allow_patterns=["background_texture.zip", "embodiments.zip", "objects.zip"],
    local_dir=str(staging),
)
for archive in sorted(staging.glob("*.zip")):
    print(f"extracting {archive.name}", file=sys.stderr)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target)
PY
}

cmd_env() {
    if "$CONDA" env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
        log "conda env '$ENV_NAME' exists; updating it to environment.yml"
        "$CONDA" env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
    else
        log "creating conda env '$ENV_NAME'"
        "$CONDA" env create -n "$ENV_NAME" -f "$ROOT/environment.yml"
    fi

    # environment.yml deliberately omits these three: as releases they would be
    # an *unpatched* lerobot, which degrades every result silently. They have to
    # come from this working tree.
    local python
    python="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
    log "installing the vendored packages editable from $ROOT"
    "$python" -m pip install -e "$ROOT/cfn"
    "$python" -m pip install -e "$ROOT/third_party/lerobot"
    "$python" -m pip install -e "$ROOT/third_party/lerobot/src/transformers"

    log "environment ready; verify with: vendor/bootstrap.sh check"
}

cmd_tokenizer() {
    # Read the destination from the settings rather than assuming one: fetching
    # into one directory while the pipeline reads another looks like it worked.
    local target
    target="$("$PYTHON" -c \
        "import sys; sys.path.insert(0, '$ROOT/scripts'); import robotwin_multitask as m; print(m.load_settings(m.ROOT).get('tokenizer', ''))" \
        2>/dev/null)"
    target="${target:-$ROOT/third_party/paligemma-3b-pt-224}"

    if [ -f "$target/tokenizer.json" ]; then
        log "tokenizer already present at $target"
        return
    fi
    # Gated on Hugging Face: accept the PaliGemma licence and run
    # `hf auth login` first, or this fails with a 401.
    log "downloading $TOKENIZER_REPO -> $target"
    mkdir -p "$target"
    "$PYTHON" - "$target" "$TOKENIZER_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[2], local_dir=sys.argv[1])
PY
}

cmd_check() {
    "$PYTHON" "$ROOT/handoff/check_env.py"
}

case "${1:-robotwin}" in
    robotwin)  cmd_robotwin ;;
    env)       cmd_env ;;
    tokenizer) cmd_tokenizer ;;
    check)     cmd_check ;;
    *)         printf 'usage: %s [robotwin|env|tokenizer|check]\n' "$0" >&2; exit 2 ;;
esac
