#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/install-dynamo.sh [OPTIONS]

Clone Dynamo, check out the request trace/replay branch, create a uv venv, build
the Python bindings, and install Dynamo into the venv.

Options:
  --workdir PATH       Cache/work directory.
                       Default: ${XDG_CACHE_HOME:-$HOME/.cache}/pi-dynamo-provider
  --dynamo-dir PATH    Dynamo checkout path. Default: <workdir>/dynamo
  --dynamo-repo URL    Dynamo git repo. Default: https://github.com/ai-dynamo/dynamo.git
  --dynamo-ref REF     Dynamo branch/ref. Default: ishan/mooncake-replay-hashes
  -h, --help           Show this help.

Environment overrides:
  PI_DYNAMO_WORKDIR
  DYNAMO_DIR
  DYNAMO_REPO
  DYNAMO_REF
EOF
}

log() {
    printf '[pi-dynamo-install] %s\n' "$*"
}

die() {
    printf '[pi-dynamo-install] ERROR: %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

DEFAULT_WORKDIR="${XDG_CACHE_HOME:-$HOME/.cache}/pi-dynamo-provider"
WORKDIR="${PI_DYNAMO_WORKDIR:-$DEFAULT_WORKDIR}"
DYNAMO_DIR="${DYNAMO_DIR:-}"
DYNAMO_REPO="${DYNAMO_REPO:-https://github.com/ai-dynamo/dynamo.git}"
DYNAMO_REF="${DYNAMO_REF:-ishan/mooncake-replay-hashes}"

while (($# > 0)); do
    case "$1" in
        --workdir)
            WORKDIR="$2"
            shift 2
            ;;
        --dynamo-dir)
            DYNAMO_DIR="$2"
            shift 2
            ;;
        --dynamo-repo)
            DYNAMO_REPO="$2"
            shift 2
            ;;
        --dynamo-ref)
            DYNAMO_REF="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

DYNAMO_DIR="${DYNAMO_DIR:-$WORKDIR/dynamo}"

require_cmd git
require_cmd uv
require_cmd python3

mkdir -p "$WORKDIR"

if [[ ! -d "$DYNAMO_DIR/.git" ]]; then
    log "Cloning Dynamo into $DYNAMO_DIR"
    git clone "$DYNAMO_REPO" "$DYNAMO_DIR"
fi

cd "$DYNAMO_DIR"

if ! git diff --quiet || ! git diff --cached --quiet; then
    die "Dynamo checkout has local changes: $DYNAMO_DIR"
fi

log "Fetching Dynamo ref $DYNAMO_REF"
git fetch origin "$DYNAMO_REF"
git checkout --detach FETCH_HEAD

log "Creating uv venv at $DYNAMO_DIR/.venv"
uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing build tools"
uv pip install -U pip maturin

log "Building Dynamo Python bindings"
(cd lib/bindings/python && maturin develop --uv)

log "Installing Dynamo package"
uv pip install -e .

cat <<EOF

Dynamo installed.

Checkout:
  $DYNAMO_DIR

Activate:
  cd $DYNAMO_DIR
  source .venv/bin/activate

Launch GLM-4.7-Flash:
  $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch-agg-agent.sh --dynamo-dir $DYNAMO_DIR

EOF
