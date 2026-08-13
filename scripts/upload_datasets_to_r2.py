#!/usr/bin/env python3
"""
One-time (or "run again after adding new data") upload of Datasets/ to the
R2 bucket the deployed app downloads from at boot (see
backend/core/dataset_sync.py). Run this from your own machine, where
Datasets/ already exists — NOT something the deployed app itself runs.

Requires the same four env vars dataset_sync.py reads at boot:
  R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME

Usage:
  export R2_ACCOUNT_ID=...
  export R2_ACCESS_KEY_ID=...
  export R2_SECRET_ACCESS_KEY=...
  export R2_BUCKET_NAME=vito-datasets
  python3 scripts/upload_datasets_to_r2.py

Skips re-uploading a file whose size already matches what's in the bucket
(cheap, size-only check — good enough for "did this file already go up,"
not meant as a byte-for-byte integrity guarantee). Pass --force to
re-upload everything regardless.
"""

import os
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent / "Datasets"


def main():
    force = "--force" in sys.argv

    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME")
    if not all([account_id, access_key, secret_key, bucket]):
        print("Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME first.", file=sys.stderr)
        sys.exit(1)

    if not DATASETS_DIR.exists():
        print(f"{DATASETS_DIR} doesn't exist — nothing to upload.", file=sys.stderr)
        sys.exit(1)

    import boto3

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )

    # existing object sizes, to skip files that are already up there unchanged
    existing = {}
    if not force:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                existing[obj["Key"]] = obj["Size"]

    files = [f for f in DATASETS_DIR.rglob("*") if f.is_file()]
    print(f"Found {len(files)} files under {DATASETS_DIR}")

    uploaded, skipped, total_bytes = 0, 0, 0
    for f in files:
        key = str(f.relative_to(DATASETS_DIR))
        size = f.stat().st_size
        if not force and existing.get(key) == size:
            skipped += 1
            continue
        client.upload_file(str(f), bucket, key)
        uploaded += 1
        total_bytes += size
        print(f"  uploaded {key} ({size / 1e6:.1f}MB)")

    print(f"\nDone — {uploaded} uploaded ({total_bytes / 1e9:.2f}GB), {skipped} already up to date.")


if __name__ == "__main__":
    main()
