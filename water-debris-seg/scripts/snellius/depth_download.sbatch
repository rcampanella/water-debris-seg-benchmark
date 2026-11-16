#!/bin/bash
#SBATCH --partition=staging
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -J wdseg-depth-dl
#SBATCH -o /scratch-shared/%u/water-debris-seg/logs/depth_download-%j.out

# Depth benchmark D01, step 1 (README § Launch plan, D8): download the raw streams of one source to scratch.
#   SOURCE=pohang00..pohang05 : stereo.zip, lidar/lidar_front.zip, navigation.zip, calibration.zip (anonymous S3)
#   SOURCE=multiaqua          : MULTIAQUA_full.zip (single connection, resumable; the server rate-limits)
# Writes <dest>/manifest.json (file, bytes, SHA-256, zip member count) and copies it to the repo sources.
set -euo pipefail
SOURCE="${SOURCE:?set SOURCE=pohang00..pohang05|multiaqua}"
SCRATCH="/scratch-shared/${USER}/water-debris-seg"
DEST="${SCRATCH}/depth_raw/${SOURCE}"
REPO="${HOME}/projects/water-debris-seg-benchmark"
EVID="${REPO}/eval_outputs/water_debris_depth/sources/d01_downloads"
mkdir -p "${DEST}" "${SCRATCH}/logs" "${EVID}"
cd "${DEST}"

if [[ "${SOURCE}" == pohang0[0-5] ]]; then
  B="s3://pohang-canal-dataset/${SOURCE}"
  for f in calibration.zip navigation.zip lidar/lidar_front.zip stereo.zip; do
    mkdir -p "$(dirname "$f")"
    aws s3 cp --no-sign-request --only-show-errors "${B}/${f}" "${f}"
  done
  FILES="calibration.zip navigation.zip lidar/lidar_front.zip stereo.zip"
elif [[ "${SOURCE}" == multiaqua ]]; then
  wget -q -c --tries=20 --waitretry=60 https://data.lmi.link/datasets/MULTIAQUA_full.zip
  FILES="MULTIAQUA_full.zip"
else
  echo "unknown SOURCE ${SOURCE}"; exit 1
fi

python3 - "${DEST}" "${SOURCE}" ${FILES} <<'EOF'
import sys, os, json, hashlib, zipfile, datetime
dest, source, files = sys.argv[1], sys.argv[2], sys.argv[3:]
out = {"source": source, "dest": dest, "date": datetime.datetime.now().isoformat(timespec="seconds"), "files": []}
for f in files:
    p = os.path.join(dest, f); h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""): h.update(chunk)
    with zipfile.ZipFile(p) as z: n = len(z.infolist())
    out["files"].append({"file": f, "bytes": os.path.getsize(p), "sha256": h.hexdigest(), "zip_members": n})
    print(f, os.path.getsize(p), h.hexdigest(), n, flush=True)
json.dump(out, open(os.path.join(dest, "manifest.json"), "w"), indent=1)
EOF
cp "${DEST}/manifest.json" "${EVID}/manifest_${SOURCE}.json"
echo "done ${SOURCE}"
