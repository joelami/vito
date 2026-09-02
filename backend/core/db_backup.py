"""
Backs up the SQLite database (database.py's DB_PATH -- bet log, forward-test
track record, forward_parlays, settings) to the same Cloudflare R2 bucket
that already hosts Datasets/ (see dataset_sync.py). Real incident this
exists to prevent from recurring: on 2026-09-02, the Railway Volume backing
DB_PATH was recreated (an infrastructure event, not an application bug --
confirmed directly: nothing in this codebase issues a wholesale DELETE or
DROP against forward_picks/forward_parlays) and the entire live forward-test
history -- 140+ settled picks, the app's actual track record -- was lost
with no way back, because the only copy of that data ever existed on that
one Volume.

Same boto3-against-R2's-S3-API pattern as dataset_sync.py, reusing the
identical R2_* environment variables -- no new credentials, no new bucket.
Backups live under a `db_backups/` prefix in the SAME bucket Datasets/
already uses, timestamped, with only the most recent KEEP_LAST_N kept (old
ones deleted on upload) so this doesn't grow unbounded.

This is NOT a replacement for a real persistent Volume -- it's a second,
independent copy so a Volume-level incident (recreation, corruption,
accidental deletion) costs at most one day of picks, not the entire
history. Runs once daily from the in-process scheduler (see scheduler.py),
piggybacking on the existing full-run cycle rather than adding a new timer.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import database

BACKUP_PREFIX = "db_backups/"
KEEP_LAST_N = 30  # ~a month of daily backups -- bounds R2 storage/list cost, still a generous recovery window


def _r2_client():
    account_id = os.environ.get("R2_ACCOUNT_ID")
    access_key = os.environ.get("R2_ACCESS_KEY_ID")
    secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("R2_BUCKET_NAME")
    if not all([account_id, access_key, secret_key, bucket]):
        return None, None

    import boto3
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
    return client, bucket


def backup_database() -> Optional[str]:
    """
    Uploads the current DB_PATH file to R2 under a timestamped key, then
    deletes older backups beyond KEEP_LAST_N. Returns the uploaded key, or
    None if R2 credentials aren't configured (same graceful no-op as
    dataset_sync.py -- local dev never needs this) or the DB file doesn't
    exist yet (nothing to back up on a truly fresh boot).
    """
    if not database.DB_PATH.exists():
        print("[db_backup] no database file yet, nothing to back up.")
        return None

    client, bucket = _r2_client()
    if client is None:
        print("[db_backup] R2 credentials not set -- skipping backup (expected for local dev).")
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    key = f"{BACKUP_PREFIX}sports_bet_{timestamp}.db"

    try:
        client.upload_file(str(database.DB_PATH), bucket, key)
        size_mb = database.DB_PATH.stat().st_size / 1e6
        print(f"[db_backup] uploaded {key} ({size_mb:.1f}MB)")
    except Exception as e:
        print(f"[db_backup] upload FAILED: {e}", file=sys.stderr)
        return None

    _prune_old_backups(client, bucket)
    return key


def _prune_old_backups(client, bucket: str) -> None:
    """Keeps only the KEEP_LAST_N most recent backups -- deletes the rest.
    Failure here is non-fatal (a few extra old backups sitting in R2 costs
    a little storage, not correctness), so this only ever logs, never
    raises past this function."""
    try:
        paginator = client.get_paginator("list_objects_v2")
        objects = []
        for page in paginator.paginate(Bucket=bucket, Prefix=BACKUP_PREFIX):
            objects.extend(page.get("Contents", []))
        objects.sort(key=lambda o: o["Key"], reverse=True)  # timestamped keys sort chronologically
        stale = objects[KEEP_LAST_N:]
        for obj in stale:
            client.delete_object(Bucket=bucket, Key=obj["Key"])
        if stale:
            print(f"[db_backup] pruned {len(stale)} backup(s) beyond the last {KEEP_LAST_N}")
    except Exception as e:
        print(f"[db_backup] prune step FAILED (non-fatal, old backups just accumulate): {e}", file=sys.stderr)


def restore_latest_backup(force: bool = False) -> Optional[str]:
    """
    Downloads the most recent backup from R2 and overwrites the local
    DB_PATH with it. Real recovery tool for exactly the incident this
    module exists to prevent -- run manually (`python3 -m core.db_backup
    restore`) after confirming the current database is the one you want to
    replace. Refuses to overwrite an existing, non-empty database unless
    force=True, since restoring is destructive to whatever's currently
    there -- the whole point is not to compound one data-loss incident
    with a second, careless one.
    """
    client, bucket = _r2_client()
    if client is None:
        print("[db_backup] R2 credentials not set -- cannot restore.", file=sys.stderr)
        return None

    if database.DB_PATH.exists() and database.DB_PATH.stat().st_size > 0 and not force:
        print(f"[db_backup] {database.DB_PATH} already exists and is non-empty -- refusing to overwrite "
              f"without force=True. If you're SURE you want to replace it, re-run with force.", file=sys.stderr)
        return None

    paginator = client.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=BACKUP_PREFIX):
        objects.extend(page.get("Contents", []))
    if not objects:
        print("[db_backup] no backups found in R2.", file=sys.stderr)
        return None

    latest = max(objects, key=lambda o: o["Key"])["Key"]
    database.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, latest, str(database.DB_PATH))
    print(f"[db_backup] restored {latest} -> {database.DB_PATH}")
    return latest


if __name__ == "__main__":
    if "restore" in sys.argv[1:]:
        restore_latest_backup(force="--force" in sys.argv[1:])
    else:
        backup_database()
