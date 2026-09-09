#!/usr/bin/env bash
# Update the ONE cluster checkout in place to match local HEAD.
#
# The cluster has a normal git checkout of this repo at $CLUSTER_TOPLEVEL
# (branch main; floorplan-inpainting/{data,models,outputs} are real gitignored
# dirs). This ships local commits there as a git bundle (no GitHub credentials
# needed) and fast-forwards. No per-commit release directory, no symlink setup.
#
#   commit locally  ->  scripts/deploy_cluster.sh  ->  ssh <host> 'cd <dir> && sbatch ...'
#
# Fast-forward only: it refuses if the cluster branch has commits not in local
# HEAD (resolve by hand in that case). It never touches data/models/outputs.
set -euo pipefail

CLUSTER_HOST=${CLUSTER_HOST:-login1.pegasus.kl.dfki.de}
CLUSTER_TOPLEVEL=${CLUSTER_TOPLEVEL:-/netscratch/espandar/inpainting}
ORIGIN_URL=${ORIGIN_URL:-https://github.com/EEZeus/applying-inpainting-as-augmentation-method-on-floorplan-images.git}

REPO_ROOT=$(git rev-parse --show-toplevel)
if [[ -n $(git -C "$REPO_ROOT" status --porcelain --untracked-files=no) ]]; then
  echo "refusing: tracked files have uncommitted changes - commit first" >&2
  exit 1
fi
SHORT=$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)

BUNDLE=$(mktemp "${TMPDIR:-/tmp}/fp-main.XXXXXX.bundle")
trap 'rm -f "$BUNDLE"' EXIT
git -C "$REPO_ROOT" bundle create "$BUNDLE" main
rsync -a "$BUNDLE" "$CLUSTER_HOST:$CLUSTER_TOPLEVEL/.deploy/fp-main.bundle"

ssh "$CLUSTER_HOST" "set -eu
  cd '$CLUSTER_TOPLEVEL'
  git bundle verify .deploy/fp-main.bundle >/dev/null
  git fetch --no-tags .deploy/fp-main.bundle main
  if ! git merge-base --is-ancestor HEAD FETCH_HEAD; then
    echo 'cluster main has diverged from local HEAD; resolve manually' >&2
    exit 2
  fi
  git merge --ff-only FETCH_HEAD
  git remote set-url origin '$ORIGIN_URL' 2>/dev/null || true
  echo -n 'cluster now at: '; git -C floorplan-inpainting log --oneline -1
  test -d floorplan-inpainting/data && test -d floorplan-inpainting/outputs \
    && echo 'artifacts intact (data/, outputs/)'
"
echo "deploy_cluster: $CLUSTER_TOPLEVEL updated to $SHORT"
echo "submit with:    ssh $CLUSTER_HOST 'cd $CLUSTER_TOPLEVEL/floorplan-inpainting && sbatch slurm/<file>.sbatch'"
