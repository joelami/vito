# Deploying Vito (GitHub + Railway)

Real constraint that shapes everything below: `Datasets/` is 7.6GB (NHL alone is 6.7GB), which can't go into git. It's hosted on Cloudflare R2 instead and downloaded once by the app itself (`backend/core/dataset_sync.py`) the first time it boots. Everything else here follows from that.

## 1. Cloudflare R2 — one-time setup

1. Sign up / log in at [dash.cloudflare.com](https://dash.cloudflare.com), go to **R2 Object Storage**, create a bucket (e.g. `vito-datasets`). R2's free tier is 10GB storage — this fits.
2. **R2 → Manage API Tokens → Create API Token.** Permissions: Object Read & Write, scoped to the bucket you just made. Save the **Access Key ID** and **Secret Access Key** it shows you — that's the only time the secret is shown.
3. Your **Account ID** is on the R2 overview page (right sidebar).

You now have four values: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`.

## 2. Upload the datasets (from your own machine, once)

```bash
pip install boto3   # if not already installed
export R2_ACCOUNT_ID=...
export R2_ACCESS_KEY_ID=...
export R2_SECRET_ACCESS_KEY=...
export R2_BUCKET_NAME=vito-datasets
python3 scripts/upload_datasets_to_r2.py
```

This takes a while the first time (7.6GB). Re-run it any time you add new data — it skips files whose size already matches what's up there, so a re-run after adding one small file is fast.

## 3. Push the code to GitHub

`Datasets/` is already excluded via `.gitignore` (along with `logs/`, `*.db`, `__pycache__/`). Everything else — `backend/`, `frontend/`, `docs/`, `scripts/`, `requirements.txt`, `railway.toml` — is real code and belongs in git.

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/joelami/vito.git
git push -u origin main
```

That last step needs your own GitHub auth (either `gh auth login` first, or a credential manager already configured) — run it yourself rather than through me.

## 4. Railway — create the project

1. [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo** → pick `joelami/vito`.
2. Railway auto-detects Python via Nixpacks and reads `railway.toml` for the start command (`cd backend && python3 main.py`) — no manual build config needed.
3. **Variables** tab, add:
   - `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` — same four values from step 1.
   - `DB_PATH` = `/data/sports_bet.db` (see step 5 — this only works once the volume below is mounted there).
   - `PORT` — Railway sets this automatically, don't set it yourself; `main.py` already reads it (`os.environ.get("PORT", 8010)`).

## 5. Add a persistent Volume (for the database)

Without this, `sports_bet.db` — your entire bet log, forward-test track record, and settled-picks history — gets wiped on every redeploy, since Railway's container filesystem is ephemeral by default.

1. On the service → **Settings → Volumes → New Volume**.
2. Mount path: `/data`.
3. Confirms with the `DB_PATH=/data/sports_bet.db` variable from step 4 — the app writes its database onto the persistent volume instead of the container's throwaway disk.

Worth doing the same for `Datasets/` too, purely as a speed optimization: mount a second volume at `/app/Datasets` (or wherever the container's working directory lands `Datasets/` — check the first deploy's logs to confirm the exact path `core/dataset_sync.py` resolves to). Without it, every restart re-downloads 7.6GB from R2 before the app can serve a single request; with it, that only happens once, ever.

## 6. First deploy

Push to `main` (or click Deploy in Railway) and watch the build logs. First boot will take a few minutes — it's downloading 7.6GB from R2 before it can build any sport's model. Look for:

```
[dataset_sync] done — N files, 7.XXGB total.
[startup] NFL: 5431 games loaded, ...
[startup] MLB: ...
```

Once you see `Uvicorn running on http://0.0.0.0:...`, Railway will start routing traffic to it. Visit the Railway-provided URL (or a custom domain, set under **Settings → Networking**).

## 7. The harness — an in-process scheduler, not Railway Cron

`launchd` (the current local scheduling — `com.sportsbet.harness.plist` / `.syncOnly.plist`) is macOS-only and doesn't exist on Railway. The obvious next thought is a separate Railway **Cron Job** service — but Railway Volumes are scoped to a single service, so a separate cron service wouldn't share the same `/data` volume as the main web service unless cross-service volume sharing is confirmed available on your plan. Get that wrong and the cron job silently writes to its own disconnected database — the harness "runs successfully" in its logs, but the web app never sees a single pick it logged. That's a worse failure mode than not scheduling it at all, because it looks like it's working.

Instead, `backend/scheduler.py` runs the harness **in-process**, inside the same container as the API server — guaranteeing "same filesystem, same `DB_PATH`" by construction, no cross-service assumptions needed. It's opt-in (off by default, so local dev via `python3 main.py` never double-runs against your own machine's launchd-scheduled harness):

1. Railway → `vito` service → **Variables** → add `ENABLE_SCHEDULER=1`.
2. That's it. On every boot it immediately runs a full harness pass once (so a fresh deploy/volume doesn't sit empty until the next scheduled time), then keeps running daily: a full pass at 09:00 UTC, a lighter sync-only pass at 21:00 UTC — both idempotent (see `harness.py`'s `snapshot_new_picks`/`settle_finished_picks` docstrings), so an extra run never double-logs anything.
3. Override the two default times if needed via `SCHEDULER_FULL_RUN_UTC` / `SCHEDULER_SYNC_ONLY_UTC` (both `"HH:MM"` in UTC).

Tradeoff worth knowing: the harness's model rebuild briefly shares CPU with the live web server once or twice a day (and once more on every redeploy, via the boot-time run). For this app's traffic level that's a non-issue; if it ever isn't, revisit a real separate service — but only after confirming volume sharing actually works, not before.

## What's already handled in code

- `main.py` reads `$PORT` (was hardcoded to 8010 before — fixed).
- `database.py`'s `DB_PATH` already reads `$DB_PATH` from the environment, defaulting to a local file for dev — no code change needed, just set the variable.
- `core/dataset_sync.py` is a no-op if `Datasets/` is already present (local dev, or a populated volume) — safe to leave the call in permanently, it doesn't slow down a normal local run at all.
