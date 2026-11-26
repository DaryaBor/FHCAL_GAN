#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REF=reference_FHCal_GMM
REV=84fd872c03de9ed4bd03010c894150e4d8042bbd
if [[ ! -d "$REF/.git" ]]; then
  git clone https://github.com/voboch/FHCal_GMM.git "$REF"
fi
git -C "$REF" fetch --all --tags
git -C "$REF" checkout --detach "$REV"
actual=$(git -C "$REF" rev-parse HEAD)
[[ "$actual" == "$REV" ]] || { echo "Bad FHCal_GMM commit"; exit 1; }
echo "Author reference at $actual; unmodified upstream files in $REF"
