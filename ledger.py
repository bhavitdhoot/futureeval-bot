"""
Spending ledger: what forecasts actually cost, and how often questions open.

The budget plan used to rest on guesses (a fixed cost per tier, 8 questions a
day). This records measured numbers instead, so the planner can buy the most
forecasting the remaining credits can sustain until the season ends:

  - every open question the bot sees, with when it first saw it
    (gives the real question rate, per tournament);
  - every batch it forecast: how many model forecasts per question, whether
    reasoning effort was capped, and how much the credit balance fell
    (gives the real cost of one forecast "slot").

It is a small JSON file. On GitHub Actions it survives between runs through
the Actions cache (see .github/workflows/forecast.yaml); locally it just sits
in .ledger/. Losing it is harmless: the planner falls back to priors measured
in October 2026 production runs.

Research is a fixed cost per question, so where per-call cost reports show
how much of a batch went on research, that part is priced per question and
only the rest per forecast slot.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("ledger")

# Spend buckets: which roster (flagship or the previous-generation "economy"
# one) at which effort cap ("uncapped" = whatever is configured, high by
# default).
UNCAPPED = "uncapped"
ECONOMY = "economy"
ECONOMY_MEDIUM = "economy-medium"
ECONOMY_LOW = "economy-low"


def bucket(economy: bool, effort_cap: str | None) -> str:
    if not economy:
        return effort_cap or UNCAPPED
    return f"{ECONOMY}-{effort_cap}" if effort_cap else ECONOMY


# Prior cost of one forecast slot, research and parsing included, from runs
# of this code on 2026-10-29:
#   flagship: GPT-6.1 Sol at high effort, $0.35 a question on the practice
#     tournament (before research was capped);
#   economy: GPT-5.6 Sol at high effort, $0.25 on both the practice tournament
#     and the first production question with uncontaminated balance readings;
#   economy-medium: $0.15 on the practice tournament;
#   economy-low: an estimate, scaled from the medium/high ratio.
PRIOR_SLOT_COST = {UNCAPPED: 0.35, ECONOMY: 0.25, ECONOMY_MEDIUM: 0.15, ECONOMY_LOW: 0.09}
PRIOR_QUESTIONS_PER_DAY = 8.0
PRIOR_MINIBENCH_SHARE = 0.25
KEEP_DAYS = 30
RATE_WINDOW_DAYS = 7
MIN_RATE_DAYS = 2  # measure the question rate only over at least this long
# Questions seen within this long of a fresh ledger's start were already open:
# they are a backlog, not arrivals, and would inflate the rate if counted.
BASELINE = dt.timedelta(minutes=30)
COST_WINDOW = 25  # most recent spend records used per bucket
# Bump when what a spend bucket means changes (e.g. which roster it measures);
# spend rows from another version are dropped on load, question sightings kept.
SCHEMA_VERSION = 2


def ledger_path() -> Path:
    return Path(os.getenv("LEDGER_PATH", "").strip() or Path(__file__).resolve().parent / ".ledger" / "ledger.json")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse(ts: str) -> dt.datetime:
    when = dt.datetime.fromisoformat(ts)
    return when if when.tzinfo else when.replace(tzinfo=dt.timezone.utc)


def _valid_seen(v) -> bool:
    try:
        return isinstance(v, dict) and isinstance(v["tournament"], str) and bool(_parse(v["first_seen"]))
    except Exception:  # noqa: BLE001
        return False


def _valid_spend(s) -> bool:
    try:
        return (
            isinstance(s, dict)
            and bool(_parse(s["ts"]))
            and isinstance(s["effort"], str)
            and int(s["slots"]) > 0
            and int(s["questions"]) > 0
            and float(s["usd"]) >= 0
        )
    except Exception:  # noqa: BLE001
        return False


@dataclass
class Stats:
    questions_per_day: float
    minibench_share: float
    slot_cost: dict[str, float]  # per forecast slot, research excluded when measured
    measured: bool  # False when every number is a prior
    why: str = ""
    research_cost: float = 0.0  # per question; 0 while research is folded into slot_cost

    def plan_cost(self, bucket_name: str, slots: float) -> float:
        cost = self.slot_cost.get(bucket_name, PRIOR_SLOT_COST.get(bucket_name, float("inf")))
        return self.research_cost + cost * slots


@dataclass
class Ledger:
    seen: dict[str, dict] = field(default_factory=dict)  # question key -> {tournament, first_seen}
    spend: list[dict] = field(default_factory=list)  # {ts, effort, slots, questions, usd}
    created: str = field(default_factory=lambda: _now().isoformat())

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: Path | None = None) -> Ledger:
        """Never raises: a missing, corrupt or half-valid file gives what can be salvaged."""
        path = path or ledger_path()
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return cls()
        except Exception as e:  # noqa: BLE001 - a corrupt ledger must never stop forecasting
            logger.warning(f"Ignoring unreadable ledger {path}: {type(e).__name__}: {e}")
            return cls()
        if not isinstance(data, dict):
            return cls()
        seen = data.get("seen") if isinstance(data.get("seen"), dict) else {}
        spend = data.get("spend") if isinstance(data.get("spend"), list) else []
        if data.get("version") != SCHEMA_VERSION:
            if spend:
                logger.warning(f"Ledger {path} is from another schema version; discarding its spend records")
            spend = []
        book = cls(
            seen={str(k): v for k, v in seen.items() if _valid_seen(v)},
            spend=[s for s in spend if _valid_spend(s)],
        )
        try:
            book.created = _parse(data["created"]).isoformat()
        except Exception:  # noqa: BLE001 - older or damaged file: treat as fresh
            pass
        dropped = len(seen) - len(book.seen) + len(spend) - len(book.spend)
        if dropped:
            logger.warning(f"Dropped {dropped} malformed ledger entries from {path}")
        return book

    def save(self, path: Path | None = None, now: dt.datetime | None = None) -> None:
        path = path or ledger_path()
        try:
            self.prune(now)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            payload = {"version": SCHEMA_VERSION, "created": self.created, "seen": self.seen, "spend": self.spend}
            tmp.write_text(json.dumps(payload, indent=1))
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
        self,
        effort: str,
        slots: int,
        questions: int,
        usd: float,
        now: dt.datetime | None = None,
        research_share: float | None = None,
    ) -> None:
        """
        `usd` is the balance drop (authoritative total). `research_share` is
        the fraction of it that went on research, from per-call cost reports,
        when known: research is a fixed cost per question, not per forecast.
        """
        if questions <= 0 or slots <= 0:
            return
        row = {
            "ts": (now or _now()).isoformat(),
            "effort": effort,
            "slots": slots,
            "questions": questions,
            "usd": round(max(usd, 0.0), 5),
        }
        if research_share is not None and 0 <= research_share <= 1:
            row["research_share"] = round(research_share, 4)
        self.spend.append(row)

    def add_late_charges(self, usd: float) -> None:
        """
        Charges that reached the balance only after a batch's closing reading
        belong to that batch (nothing else spends from the key between
        batches). Without this they would fall between readings and never be
        counted, making forecasts look cheaper than they are.
        """
        if usd > 0 and self.spend:
            self.spend[-1]["usd"] = round(self.spend[-1]["usd"] + usd, 5)

    # ---------------------------------------------------------------- stats
    def stats(self, now: dt.datetime | None = None) -> Stats:
        now = now or _now()
        notes, measured = [], False

        created = _parse(self.created)
        window_start = max(created, now - dt.timedelta(days=RATE_WINDOW_DAYS))
        span_days = (now - window_start).total_seconds() / 86400
        arrivals = [
            v
            for v in self.seen.values()
            if _parse(v["first_seen"]) >= window_start and _parse(v["first_seen"]) > created + BASELINE
        ]
        if span_days >= MIN_RATE_DAYS and arrivals:
            rate = len(arrivals) / span_days
            share = sum(1 for v in arrivals if v["tournament"] == "minibench") / len(arrivals)
            notes.append(f"{rate:.1f} questions/day measured")
            measured = True
        else:
            rate, share = PRIOR_QUESTIONS_PER_DAY, PRIOR_MINIBENCH_SHARE

        # Research is a fixed cost per question. Where rows say how much of
        # their spend it was, take it out of the per-slot cost and price it per
        # question, so multi-forecast plans aren't overpriced.
        split_rows = [s for s in self.spend if "research_share" in s][-COST_WINDOW:]
        research_cost = 0.0
        if sum(s["questions"] for s in split_rows) >= 3:
            research_cost = sum(s["usd"] * s["research_share"] for s in split_rows) / sum(
                s["questions"] for s in split_rows
            )
            notes.append(f"${research_cost:.3f}/question research measured")
            measured = True

        def slot_usd(s: dict) -> float:
            if "research_share" in s:
                return s["usd"] * (1 - s["research_share"])
            if research_cost:
                return max(s["usd"] - research_cost * s["questions"], 0.0)
            return s["usd"]

        found = {}
        for name in PRIOR_SLOT_COST:
            rows = [s for s in self.spend if s["effort"] == name][-COST_WINDOW:]
            slots = sum(s["slots"] * s["questions"] for s in rows)
            # Totals, not a median of per-batch costs: a charge that posts
            # during the next batch moves cost between neighbouring batches but
            # not out of the total.
            if slots >= 3:
                found[name] = sum(slot_usd(s) for s in rows) / slots
                notes.append(f"${found[name]:.3f}/forecast {name} measured")
                measured = True
        # Buckets without data are scaled from the ones with data, at the
        # priors' ratios (averaged when several buckets have data).
        slot_cost = dict(PRIOR_SLOT_COST)
        if found:
            scale = sum(found[b] / PRIOR_SLOT_COST[b] for b in found) / len(found)
            slot_cost = {b: found.get(b, PRIOR_SLOT_COST[b] * scale) for b in PRIOR_SLOT_COST}
        return Stats(rate, share, slot_cost, measured, ", ".join(notes) or "priors only", research_cost)
