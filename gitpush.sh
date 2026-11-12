#!/bin/bash
# Stage all changes, commit, and push the current branch.
# Usage: ./gitpush.sh "commit message"   (message optional; defaults to a timestamp)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch="$(git rev-parse --abbrev-ref HEAD)"
msg="${1:-Update $(date '+%Y-%m-%d %H:%M:%S')}"

git add -A

if git diff --cached --quiet; then
    echo "Nothing new to commit."
else
    git commit -m "$msg"
fi

# Set upstream automatically the first time a branch is pushed
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    git push
else
    git push -u origin "$branch"
fi

echo "Pushed '$branch' to origin."
