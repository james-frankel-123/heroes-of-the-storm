#!/bin/bash
# Pin (or re-verify) this host's code bundle in a pass's own code manifest (review C7).
# Usage (worker, from ~/hots/repo/training): bash personalization/remote/p3t_code_pin.sh pin|verify <manifest> <pass> <host>
set -euo pipefail
V=${1:?pin|verify}; M=${2:?manifest}; P=${3:?pass}; H=${4:?host label}
mkdir -p "$(dirname "$M")"
if [ "$V" = pin ]; then
  [ -s "$M" ] || echo '{"schema": 1, "files": []}' > "$M"
  python personalization/p3_manifest.py code "$M" --pass "$P" --host "$H"
else
  python personalization/p3_manifest.py verify "$M" --role code --host "$H"
fi
