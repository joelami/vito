"""
Observability infrastructure for the model/harness itself — not for game
predictions, but for tracking how the SYSTEM evolves over time. Three
pillars, the standard software-observability triad applied to this project's
own iteration process:

  1. VERSIONING — every meaningful change to the model/harness gets a
     manifest recording what changed and why, in `harness_versions/`.
  2. DECISION LOG — every tuning/iteration decision is logged with its
     reasoning AND its outcome (not just "we changed K-factor to 25" but
     "we changed it because X, and when we ran it, Y happened"), in
     `decision_log.jsonl`.
  3. EXECUTION TRACE — every harness run logs each step with timing, in
     `logs/harness_traces/`, so any run's behavior and duration can be
     reconstructed after the fact without re-running it.

Why this exists: a model that "keeps getting better" is only trustworthy if
you can look back and see WHY each change was made and WHAT ACTUALLY
HAPPENED when it was tried — otherwise "iteration" is indistinguishable from
undocumented fiddling, which is exactly the kind of thing that turns into
p-hacking without anyone noticing. This is the audit trail that keeps the
iteration process itself honest, not just any single backtest.
"""

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
VERSIONS_DIR = BASE_DIR / "harness_versions"
DECISION_LOG_PATH = BASE_DIR / "decision_log.jsonl"
TRACES_DIR = BASE_DIR.parent / "logs" / "harness_traces"

VERSIONS_DIR.mkdir(exist_ok=True)
TRACES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pillar 1: Versioning
# ---------------------------------------------------------------------------
def _next_version_number() -> int:
    existing = [p for p in VERSIONS_DIR.iterdir() if p.is_dir() and p.name.startswith("v")]
    nums = []
    for p in existing:
        try:
            nums.append(int(p.name.split("_")[0][1:]))
        except (ValueError, IndexError):
            continue
    return max(nums, default=0) + 1


def record_version(name: str, description: str, why: str, sport: str = "NFL",
                    config_snapshot: dict = None, files_changed: list = None,
                    result: str = None) -> Path:
    """
    Creates `harness_versions/vN_{name}/manifest.json`. Call this whenever a
    change to the model/features/harness is significant enough that a future
    reader would want to know it happened as a distinct step, not buried in
    a diff — new feature, new sport, tuned constants, architecture change.

    `result` is optional at record time (you may not know the outcome yet)
    but should be filled in via `update_version_result()` once the change
    has actually been run and measured — an undocumented "why" without a
    documented "what happened" is only half the audit trail.
    """
    n = _next_version_number()
    folder = VERSIONS_DIR / f"v{n}_{name}"
    folder.mkdir(exist_ok=True)
    manifest = {
        "version": n,
        "name": name,
        "sport": sport,
        "recorded_at": datetime.now().isoformat(),
        "description": description,
        "why": why,
        "config_snapshot": config_snapshot or {},
        "files_changed": files_changed or [],
        "result": result,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    _update_registry()
    return folder


def update_version_result(name_or_version, result: str):
    """Fills in the outcome of a version once it's actually been measured."""
    for folder in VERSIONS_DIR.iterdir():
        if not folder.is_dir():
            continue
        if folder.name == str(name_or_version) or folder.name.split("_", 1)[-1] == str(name_or_version) \
                or folder.name.startswith(f"v{name_or_version}_"):
            manifest_path = folder / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                manifest["result"] = result
                manifest["result_recorded_at"] = datetime.now().isoformat()
                manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
                return True
    return False


def _update_registry():
    """Top-level `harness_versions/manifest.json` — a registry of every
    version folder, purpose stated up front, so a reader doesn't have to
    open every subfolder to get the timeline."""
    versions = []
    for folder in sorted(VERSIONS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        manifest_path = folder / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            versions.append({"version": m["version"], "name": m["name"], "sport": m.get("sport"),
                              "recorded_at": m["recorded_at"], "description": m["description"],
                              "result": m.get("result")})
    registry = {
        "purpose": "Registry of every recorded model/harness version — each subfolder's own "
                   "manifest.json has the full why/what/result; this file is the index.",
        "versions": sorted(versions, key=lambda v: v["version"]),
    }
    (VERSIONS_DIR / "manifest.json").write_text(json.dumps(registry, indent=2, default=str))


# ---------------------------------------------------------------------------
# Pillar 2: Decision log
# ---------------------------------------------------------------------------
def log_decision(decision: str, reasoning: str, experience: str = None, sport: str = "NFL"):
    """
    Append-only log of tuning/iteration decisions. `decision` = what was
    changed. `reasoning` = why (the hypothesis). `experience` = what
    actually happened when it was run (fill in once known — pass None at
    decision time, call again with the same `decision` text plus the
    outcome once measured, or just call once if decision and outcome are
    known together).
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "sport": sport,
        "decision": decision,
        "reasoning": reasoning,
        "experience": experience,
    }
    with open(DECISION_LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def read_decision_log() -> list:
    if not DECISION_LOG_PATH.exists():
        return []
    return [json.loads(line) for line in DECISION_LOG_PATH.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Pillar 3: Execution trace
# ---------------------------------------------------------------------------
class RunTrace:
    """Records each step of one harness run with timing, written to
    `logs/harness_traces/{timestamp}.json` at the end of the run."""

    def __init__(self, run_name: str = "harness"):
        self.run_name = run_name
        self.started_at = datetime.now()
        self.steps = []

    @contextmanager
    def step(self, name: str, **details):
        t0 = time.time()
        error = None
        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            self.steps.append({
                "step": name, "duration_s": round(time.time() - t0, 3),
                "details": details, "error": error,
            })

    def save(self):
        ended_at = datetime.now()
        trace = {
            "run_name": self.run_name,
            "started_at": self.started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "total_duration_s": round((ended_at - self.started_at).total_seconds(), 3),
            "steps": self.steps,
        }
        path = TRACES_DIR / f"{self.started_at.strftime('%Y-%m-%dT%H-%M-%S')}_{self.run_name}.json"
        path.write_text(json.dumps(trace, indent=2, default=str))
        return path
