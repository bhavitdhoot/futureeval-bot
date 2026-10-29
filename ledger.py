"""
Spending ledger: what forecasts actually cost, and how often questions open.

The budget plan used to rest on guesses (a fixed cost per tier, 8 questions a
day). This records measured numbers instead, so the planner can buy the most
forecasting the remaining credits can sustain until the season ends:

  - every open question the bot sees, with when it first saw it
    (gives the real question rate, per tournament);
  - every batch it forecast: how many model forecasts per question, at which
    reasoning effort, and how much the credit balance fell
    (gives the real cost of one forecast "slot").

It is a small JSON file. On GitHub Actions it survives between runs through
the Actions cache (see .github/workflows/forecast.yaml); locally it just sits
in .ledger/. Losing it is harmless: the planner falls back to priors measured
in October 2026 production runs.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("ledger")

# Measured in production (Oct 2026): one high-effort GPT forecast plus web
# research and parsing cost $0.08-0.12 per question.
PRIOR_SLOT_COST = {"high": 0.10, "medium": 0.06}
PRIOR_QUESTIONS_PER_DAY = 8.0
PRIOR_MINIBENCH_SHARE = 0.25
KEEP_DAYS = 30
RATE_WINDOW_DAYS = 7
COST_WINDOW = 25  # most recent spend records used per effort level


def ledger_path() -> Path:
    return Path(os.getenv("LEDGER_PATH", "").strip() or Path(__file__).resolve().parent / ".ledger" / "ledger.json")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse(ts: str) -> dt.datetime:
    when = dt.datetime.fromisoformat(ts)
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


@dataclass
class Stats:
    questions_per_day: float
    minibench_share: float
    slot_cost: dict[str, float]
    measured: bool  # False when every number is a prior
    why: str = ""


@dataclass
class Ledger:
    seen: dict[str, dict] = field(default_factory=dict)  # question key -> {tournament, first_seen}
    spend: list[dict] = field(default_factory=list)  # {ts, effort, slots, questions, usd}

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: Path | None = None) -> Ledger:
        path = path or ledger_path()
        try:
            data = json.loads(path.read_text())
            return cls(seen=dict(data.get("seen", {})), spend=list(data.get("spend", [])))
        except FileNotFoundError:
            return cls()
        except Exception as e:  # noqa: BLE001 - a corrupt ledger must never stop forecasting
            logger.warning(f"Ignoring unreadable ledger {path}: {type(e).__name__}: {e}")
            return cls()

    def save(self, path: Path | None = None, now: dt.datetime | None = None) -> None:
        path = path or ledger_path()
        self.prune(now)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"seen": self.seen, "spend": self.spend}, indent=1))
            tmp.replace(path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not save ledger {path}: {type(e).__name__}: {e}")

    def prune(self, now: dt.datetime | None = None) -> None:
        cutoff = (now or _now()) - dt.timedelta(days=KEEP_DAYS)
        self.seen = {k: v for k, v in self.seen.items() if _parse(v["first_seen"]) >= cutoff}
        self.spend = [s for s in self.spend if _parse(s["ts"]) >= cutoff]

    # ---------------------------------------------------------------- record
    def record_seen(self, key: str, tournament: str, now: dt.datetime | None = None) -> None:
        if key not in self.seen:
            self.seen[key] = {"tournament": str(tournament), "first_seen": (now or _now()).isoformat()}

    def record_spend(
        self, effort: str, slots: int, questions: int, usd: float, now: dt.datetime | None = None
    ) -> None:
        if questions <= 0 or slots <= 0 or usd < 0:
            return
        self.spend.append(
            {"ts": (now or _now()).isoformat(), "effort": effort, "slots": slots, "questions": questions, "usd": round(usd, 5)}
        )

    # ---------------------------------------------------------------- stats
    def stats(self, now: dt.datetime | None = None) -> Stats:
        now = now or _now()
        notes, measured = [], False

        window_start = now - dt.timedelta(days=RATE_WINDOW_DAYS)
        recent = [v for v in self.seen.values() if _parse(v["first_seen"]) >= window_start]
        first = min((_parse(v["first_seen"]) for v in self.seen.values()), default=None)
        span_days = min(RATE_WINDOW_DAYS, (now - first).total_seconds() / 86400) if first else 0
        if span_days >= 2 and recent:
            rate = len(recent) / span_days
            share = sum(1 for v in recent if v["tournament"] == "minibench") / len(recent)
            notes.append(f"{rate:.1f} questions/day measured")
            measured = True
        else:
            rate, share = PRIOR_QUESTIONS_PER_DAY, PRIOR_MINIBENCH_SHARE

        found = {}
        for effort in PRIOR_SLOT_COST:
            rows = [s for s in self.spend if s["effort"] == effort][-COST_WINDOW:]
            slots = sum(s["slots"] * s["questions"] for s in rows)
            # Totals, not a median of per-batch costs: the balance can lag a
            # call by a few seconds, which moves cost between neighbouring
            # batches but not out of the total.
            if slots >= 3:
                found[effort] = sum(s["usd"] for s in rows) / slots
                notes.append(f"${found[effort]:.3f}/forecast at {effort} effort measured")
                measured = True
        # A level without data is scaled from one with data, at the priors' ratio.
        ratio = PRIOR_SLOT_COST["medium"] / PRIOR_SLOT_COST["high"]
        slot_cost = dict(PRIOR_SLOT_COST)
        slot_cost.update(found)
        if "high" in found and "medium" not in found:
            slot_cost["medium"] = found["high"] * ratio
        if "medium" in found and "high" not in found:
            slot_cost["high"] = found["medium"] / ratio
        return Stats(rate, share, slot_cost, measured, ", ".join(notes) or "priors only")
