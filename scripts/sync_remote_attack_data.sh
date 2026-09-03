#!/usr/bin/env bash
set -u

SOURCE="mvgsage-4090:/root/autodl-tmp/UrbanPhotoMeshQA/data/HK3D-Individualised-Attack/"
DESTINATION="/Volumes/SANDISK-ELE/UrbanPhotoMeshQA-Data/HK3D-Individualised-Attack/"
mkdir -p "$DESTINATION"

for attempt in $(seq 1 30); do
  echo "sync attempt ${attempt}"
  if rsync -a --partial -e ssh "$SOURCE" "$DESTINATION"; then
    echo "sync pass complete"
    exit 0
  fi
  sleep 5
done

echo "sync failed after 30 attempts" >&2
exit 1
