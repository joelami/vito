"""
Downloads `Datasets/` from Cloudflare R2 on boot if it isn't already present
locally — the raw historical data (7.6GB, NHL alone is 6.7GB of that) is
too large to commit to git, so it lives in an R2 bucket instead and gets
pulled down once per container the first time it starts.

R2 is S3-API-compatible (this uses `boto3`'s S3 client pointed at R2's
endpoint), so this same code works unchanged against real AWS S3,
Backblaze B2, or MinIO if the hosting choice ever changes — only the
endpoint URL and credentials differ, both read from environment variables,
never hardcoded.

Idempotent and safe to call on every boot: if `Datasets/` already has real
content (local dev, or a Railway Volume that survived from a previous
deploy), this is a fast no-op — it only downloads when the directory is
missing or empty. Strongly recommended to pair with a persistent volume
mounted at the Datasets/ path in production (see docs/DEPLOYMENT.md) so the
7.6GB download only ever happens once, not on every restart.
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent.parent / "Datasets"

# A handful of files that must exist for the app to function at all —
# used as the "is this already populated" check, cheaper than listing the
# whole 7.6GB tree just to decide whether to skip the download.
_SENTINEL_FILES = [
    "NFL/nfl.xlsx",
    "College Football/cfb-games.csv",
    "NHL/archive/nhl_data_plus.csv",
]


def _already_populated() -> bool:
    return all((DATASETS_DIR / f).exists() for f in _SENTINEL_FILES)


def sync_datasets(force: bool = False) -> None:
    """
    Downloads every object in the configured R2 bucket into `Datasets/`,
    preserving the bucket's key structure as the local directory layout
    (a key like `NFL/nfl.xlsx` becomes `Datasets/NFL/nfl.xlsx`). No-op if
    `Datasets/` already looks populated, unless `force=True`.

    Required environment variables: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME. If any are missing, this prints a
    warning and returns rather than crashing the app — local development
    with `Datasets/` already on disk never needs these set at all.
    """
    if not force and _already_populated():
        print("[dataset_sync] Datasets/ already populated, skipping download.")
        return

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME")

    if not all([account_id, access_key, secret_key, bucket]):
        print("[dataset_sync] R2 credentials not fully set (need R2_ACCOUNT_ID, "
              "R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME) and Datasets/ "
              "isn't already present — the app will fail to load any sport's data.",
              file=sys.stderr)
        return

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    paginator = client.get_paginator("list_objects_v2")
    objects = [obj for page in paginator.paginate(Bucket=bucket) for obj in page.get("Contents", [])]

    # Downloaded in parallel (16 concurrent files) rather than one at a time.
    # Real problem this fixes: on a from-scratch container (first Railway
    # boot, no volume yet) this download blocks the entire app startup —
    # 1633 sequential files at typical per-request overhead alone easily
    # blows past a platform's healthcheck window, confirmed directly against
    # a real failed Railway deploy (5-minute healthcheck timeout, container
    # killed before it ever got through the download). boto3's S3 client is
    # safe to share across threads for this read-only usage; a lock only
    # guards the progress counters, not the actual HTTP calls.
    lock = threading.Lock()
    downloaded, total_bytes = 0, 0

    def _download_one(obj):
        key = obj["Key"]
        dest = DATASETS_DIR / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(dest))
        return obj["Size"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_download_one, obj): obj for obj in objects}
        for future in as_completed(futures):
            size = future.result()  # re-raises if this file's download failed
            with lock:
                downloaded += 1
                total_bytes += size
                if downloaded % 25 == 0 or downloaded == len(objects):
                    print(f"[dataset_sync] {downloaded}/{len(objects)} files, "
                          f"{total_bytes / 1e6:.0f}MB downloaded so far...")

    print(f"[dataset_sync] done — {downloaded} files, {total_bytes / 1e9:.2f}GB total.")


if __name__ == "__main__":
    sync_datasets(force="--force" in sys.argv)
