"""
Daily maintenance job. Runs once a day from .github/workflows/daily.yaml.

  python maintenance.py discover   -> probes candidate tournaments, writes TARGETS.json
  python maintenance.py check      -> exits 1 (so GitHub emails you) if something needs you

Why discovery lives here and not in the 20-minute forecast job: the Metaculus
client retries a bad tournament slug three times with blocking backoff (up to
about 2.5 minutes each). Doing that once a day is fine; doing it every 20
minutes would stall every run.

The daily commit of TARGETS.json also counts as repository activity, which stops
GitHub from auto-disabling the scheduled workflows after 60 idle days.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Callable

logger = logging.getLogger("maintenance")

ROOT = Path(__file__).resolve().parent
TARGETS_FILE = ROOT / "TARGETS.json"
OPENROUTER_KEY_URL = os.getenv("OPENROUTER_KEY_URL", "https://openrouter.ai/api/v1/auth/key")
FALL_2026_ID = 33121


def season_of(date: dt.date) -> str:
    return "spring" if date.month <= 4 else "summer" if date.month <= 8 else "fall"


def season_slugs(date: dt.date) -> list[str]:
    """Current and next season, in both naming schemes Metaculus has used."""
    nxt_month = date.month + 4
    nxt_year = date.year + (1 if nxt_month > 12 else 0)
    nxt_month = (nxt_month - 1) % 12 + 1
    out = []
    for season, year in [(season_of(date), date.year), (season_of(dt.date(nxt_year, nxt_month, 1)), nxt_year)]:
        out += [f"{season}-futureeval-{year}", f"{season}-aib-{year}"]
    return out


def current_season_slugs(date: dt.date) -> list[str]:
    return season_slugs(date)[:2]


def known_targets() -> list[int | str]:
    extra = [x.strip() for x in os.getenv("EXTRA_TOURNAMENTS", "").split(",") if x.strip()]
    try:
        from forecasting_tools import MetaculusClient

        pkg = getattr(MetaculusClient, "CURRENT_AI_COMPETITION_ID", None)
    except Exception:  # noqa: BLE001
        pkg = None
    return _dedupe([*extra, pkg, FALL_2026_ID])


def _dedupe(items) -> list[int | str]:
    seen, out = set(), []
    for c in items:
        if c is None or str(c) in seen:
            continue
        seen.add(str(c))
        out.append(int(c) if str(c).isdigit() else c)
    return out


def discover(today: dt.date, probe: Callable[[int | str], int]) -> dict:
    """
    probe(tournament) returns the number of open questions, or raises if the
    tournament doesn't exist. Returns the TARGETS payload.
    """
    candidates = _dedupe([*known_targets(), *season_slugs(today)])
    valid, open_counts, rejected = [], {}, []
    for c in candidates:
        try:
            n = probe(c)
            valid.append(c)
            open_counts[str(c)] = n
        except Exception as e:  # noqa: BLE001
            rejected.append(f"{c}: {type(e).__name__}")
    in_fall_2026 = today.year == 2026 and today.month >= 9
    current_found = in_fall_2026 or any(s in [str(v) for v in valid] for s in current_season_slugs(today))
    try:
        minibench_ok = probe("minibench") >= 0
    except Exception:  # noqa: BLE001
        minibench_ok = False
    return {
        "updated": today.isoformat(),
        "main": valid,
        "open_questions": open_counts,
        "rejected": rejected,
        "current_season_found": current_found,
        "minibench_ok": minibench_ok,
    }


def check(targets: dict | None, today: dt.date, credits_remaining: float | None) -> list[str]:
    """Return a list of human-readable problems. Empty list means all good."""
    problems = []
    # GitHub passes unset repo variables as "", so a plain getenv default is not enough.
    floor = float(os.getenv("CREDIT_FLOOR_USD") or 25)
    if credits_remaining is not None and credits_remaining < floor:
        problems.append(
            f"OpenRouter credits are low: ${credits_remaining:.2f} left (alert floor ${floor:.0f}). "
            "Request more through the Metaculus credits form, or set ROSTER_MODE=lean / "
            "SKIP_MINIBENCH=true to stretch what's left."
        )
    if targets is None:
        problems.append("TARGETS.json is missing: the discover step failed. Check the workflow log.")
        return problems
    # Seasons open partway through their first month, so only worry from the second month on.
    first_month = {"spring": 1, "summer": 5, "fall": 9}[season_of(today)]
    if not targets.get("current_season_found") and today.month != first_month:
        problems.append(
            f"No {season_of(today)} {today.year} bot tournament found. Metaculus may have renamed it. "
            "Find the tournament ID on metaculus.com/aib and add it as the repo variable EXTRA_TOURNAMENTS."
        )
    if not targets.get("minibench_ok"):
        problems.append("The MiniBench tournament could not be reached today.")
    return problems


def fetch_credits_remaining() -> float | None:
    """None means 'unknown or no limit' (never alarms)."""
    import requests

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(OPENROUTER_KEY_URL, headers={"Authorization": f"Bearer {key}"}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})
        rem = data.get("limit_remaining")
        return float(rem) if rem is not None else None
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Could not read OpenRouter credit balance: {e}")
        return None


def metaculus_probe(tournament: int | str) -> int:
    from forecasting_tools import MetaculusClient

    return len(MetaculusClient().get_all_open_questions_from_tournament(tournament))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    today = dt.date.today()
    if cmd == "discover":
        payload = discover(today, metaculus_probe)
        TARGETS_FILE.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info(f"Wrote {TARGETS_FILE.name}: {payload}")
    elif cmd == "check":
        targets = json.loads(TARGETS_FILE.read_text()) if TARGETS_FILE.exists() else None
        remaining = fetch_credits_remaining()
        if remaining is not None:
            logger.info(f"OpenRouter credits remaining: ${remaining:.2f}")
        problems = check(targets, today, remaining)
        for p in problems:
            print(f"::error::{p}")
        sys.exit(1 if problems else 0)
    else:
        sys.exit(f"unknown command {cmd!r}")
