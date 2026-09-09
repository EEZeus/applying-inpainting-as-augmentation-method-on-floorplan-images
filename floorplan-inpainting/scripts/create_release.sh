#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <git-ref> [destination-directory]" >&2
    exit 2
fi

ref=$1
destination=${2:-dist}
project_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(git -C "$project_dir" rev-parse --show-toplevel)
commit=$(git -C "$repo_root" rev-parse --verify "${ref}^{commit}")
short_commit=$(git -C "$repo_root" rev-parse --short=12 "$commit")
archive_name="floorplan-inpainting-${short_commit}.tar.gz"

if [[ -n $(git -C "$repo_root" status --porcelain --untracked-files=no) ]]; then
    echo "Refusing to release from a dirty tracked working tree." >&2
    exit 1
fi

mkdir -p "$destination"
git -C "$repo_root" archive \
    --format=tar.gz \
    --prefix="floorplan-inpainting-${short_commit}/" \
    --output="$destination/$archive_name" \
    "$commit:floorplan-inpainting"
(
    cd "$destination"
    sha256sum "$archive_name" >"$archive_name.sha256"
)

echo "$destination/$archive_name"
echo "$destination/$archive_name.sha256"
