#!/bin/bash
# Pull the latest changes for the current branch.
# Local uncommitted changes are stashed and re-applied automatically.
# Usage: ./gitpull.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch="$(git rev-parse --abbrev-ref HEAD)"

git fetch origin

if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git pull --rebase --autostash
elif git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    git pull --rebase --autostash origin "$branch"
    git branch --set-upstream-to="origin/$branch" "$branch"
else
    echo "Branch '$branch' does not exist on origin yet; nothing to pull."
    exit 0
fi

echo "'$branch' is up to date with origin."
