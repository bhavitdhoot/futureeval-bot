"""
FutureEval tournament bot (Fall 2026 onward).

Built on the official Metaculus template (template_bot.py, prompts unchanged).
What this file changes, and why:

1. Models. The stock template silently falls back to GPT-4o when only an
   OpenRouter key is set. Spring 2026 results showed frontier models at high
   reasoning effort roughly doubling the template's score, so we run a
   two-vendor roster (OpenAI + Anthropic, the providers the donated
   Metaculus key allows) and aggregate with the framework's own median/mixture.
2. Fallbacks. Every model call walks a chain: the other roster member, then
   backup models. One provider outage or one retired model slug never costs a
   question. The parser (which turns reasoning into numbers) gets the same
   treatment, because if it breaks, every question breaks.
3. Tournament targeting. The template's pinned package still pointed at the
   finished Summer 2026 tournament. We target the Fall 2026 ID explicitly, the
   package constant, and whatever the daily maintenance job (maintenance.py)
   discovers for the current season, so the bot rolls into future seasons
   without code edits.
4. One event loop for the whole run (the template calls asyncio.run twice with
   a class-level semaphore, which can bind to the wrong loop).
5. No Metaculus Cup: bots are not prize-eligible there, so it only burns credits.

Everything configurable lives in environment variables (GitHub repo
"Variables"), so changing models later never requires editing code.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
import os
import sys
import time
from collections import defaultdict

import dotenv

dotenv.load_dotenv()

from forecasting_tools import GeneralLlm  # noqa: E402
from forecasting_tools.data_models.questions import MetaculusQuestion  # noqa: E402

import maintenance  # noqa: E402
from template_bot import SummerTemplateBot2026  # noqa: E402

logger = logging.getLogger("forecast-bot")

# --------------------------------------------------------------------------
# Configuration (override any of these with GitHub repo Variables)
# --------------------------------------------------------------------------
MINIBENCH = "minibench"
TEST_AREA = "bot-testing-area"

# Slugs verified in use on OpenRouter in September 2026. Order matters.
DEFAULT_PRIMARY = [
    ("openrouter/openai/gpt-5.6-sol", "high"),
    ("openrouter/anthropic/claude-opus-4.8", "high"),
]
DEFAULT_BACKUPS = [
    ("openrouter/openai/gpt-5.5", "high"),
    ("openrouter/anthropic/claude-opus-4.7", "high"),
    ("openrouter/openai/gpt-5.2", "high"),
]
DEFAULT_PARSERS = [
    "openrouter/openai/gpt-4.1-mini",  # the framework's own default parser
    "openrouter/openai/gpt-5.6-luna",
    "openrouter/openai/gpt-4o-mini",
]
# Research models, tried in order. OpenRouter's ":online" suffix adds live web
# search to any model. The framework's old default (gpt-4o-search-preview) was
# retired from OpenRouter, which silently left the bot forecasting with no news
# at all, so these are checked against the live model list and overridable with
# the RESEARCH_MODELS variable. If all of them fail we forecast without
# research rather than let a non-search model invent the news.
DEFAULT_RESEARCH = [
    "openrouter/openai/gpt-5.6-terra:online",
    "openrouter/openai/gpt-5.6-sol:online",
]


def research_models() -> list[str]:
    raw = os.getenv("RESEARCH_MODELS", "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()] or DEFAULT_RESEARCH


def _parse_model_list(env_name: str, default: list[tuple[str, str | None]]):
    """FORECAST_MODELS="openrouter/openai/x:high,openrouter/anthropic/y:high"."""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item.rsplit("/", 1)[-1]:
            name, effort = item.rsplit(":", 1)
            out.append((name, effort or None))
        else:
            out.append((item, None))
    return out or default


def make_llm(model: str, effort: str | None = None, **overrides) -> GeneralLlm:
    params = dict(temperature=None, timeout=480, allowed_tries=2, max_tokens=32000)
    params.update(overrides)
    if effort and effort.lower() != "none":
        params["reasoning"] = {"effort": effort}
    return GeneralLlm(model=model, **params)


class RosterLlm(GeneralLlm):
    """
    Looks like one GeneralLlm to the framework, but:
      - rotates the first-choice model across repeated identical prompts, so
        prediction #1 for a question goes to model A, #2 to model B, etc.;
      - on any error, falls through the remaining roster, then the backups.
    The framework then aggregates the predictions (median for binary,
    mixture for numeric), which is what gives us a real ensemble.
    """

    def __init__(self, primaries: list[GeneralLlm], backups: list[GeneralLlm]):
        if not primaries:
            raise ValueError("RosterLlm needs at least one model")
        super().__init__(model=primaries[0].model, temperature=None)
        self.primaries = primaries
        self.backups = backups
        self._calls: dict[str, int] = defaultdict(int)
        self.usage_log: list[str] = []  # which model actually answered

    async def invoke(self, prompt, system_prompt: str | None = None) -> str:
        key = hashlib.sha256(repr((prompt, system_prompt)).encode()).hexdigest()
        k = self._calls[key]
        self._calls[key] += 1  # incremented before any await: safe under asyncio
        n = len(self.primaries)
        rotated = self.primaries[k % n :] + self.primaries[: k % n]
        chain = rotated + [b for b in self.backups if b not in rotated]
        errors = []
        for i, llm in enumerate(chain):
            try:
                if system_prompt is None:
                    answer = await llm.invoke(prompt)
                else:
                    answer = await llm.invoke(prompt, system_prompt=system_prompt)
                if i > 0:
                    logger.warning(
                        f"Fallback used: {llm.model} answered after {i} failure(s): {errors}"
                    )
                self.usage_log.append(llm.model)
                return answer
            except Exception as e:  # noqa: BLE001 - any failure means try the next model
                errors.append(f"{llm.model}: {type(e).__name__}: {str(e)[:300]}")
        raise RuntimeError("Every model in the chain failed: " + " | ".join(errors))


def build_roster(n_predictions: int, max_tokens: int = 32000) -> RosterLlm:
    primary_specs = _parse_model_list("FORECAST_MODELS", DEFAULT_PRIMARY)
    backup_specs = _parse_model_list("BACKUP_MODELS", DEFAULT_BACKUPS)
    primaries = [make_llm(m, e, max_tokens=max_tokens) for m, e in primary_specs]
    # "full" mode can ask for more predictions than roster members; repeat in order.
    while len(primaries) < n_predictions:
        primaries.append(primaries[len(primaries) % len(primary_specs)])
    backups = [make_llm(m, e, max_tokens=max_tokens) for m, e in backup_specs]
    return RosterLlm(primaries[:max(n_predictions, 1)], backups)


def build_parser_llm() -> RosterLlm:
    names = [m for m, _ in _parse_model_list("PARSER_MODELS", [(p, None) for p in DEFAULT_PARSERS])]
    llms = [GeneralLlm(model=m, temperature=None, timeout=120, allowed_tries=2) for m in names]
    return RosterLlm(llms[:1], llms[1:])


def asknews_configured() -> bool:
    return bool(
        (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
        or os.getenv("ASKNEWS_API_KEY")
    )


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------
class FutureEvalBot(SummerTemplateBot2026):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(2)  # replaced per event loop in main()

    async def run_research(self, question: MetaculusQuestion) -> str:
        """Template research (AskNews if configured), then fallbacks, then none."""
        try:
            research = await super().run_research(question)
            if research and research.strip():
                return research
            logger.warning(f"Empty research for {question.page_url}; trying fallback")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Primary research failed for {question.page_url}: {e}")

        prompt = (
            "You are an assistant to a superforecaster. Give a concise rundown of the "
            "most relevant recent news for the question below, including what the "
            "status quo outcome would be if nothing changed. Do not produce a forecast.\n\n"
            f"Question: {question.question_text}\n\n"
            f"Resolution criteria: {question.resolution_criteria}\n\n"
            f"Fine print: {question.fine_print}"
        )
        for model in research_models():
            try:
                llm = GeneralLlm(model=model, temperature=None, timeout=180, allowed_tries=1)
                return await llm.invoke(prompt)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Research fallback {model} failed: {e}")
        logger.warning(f"No research available for {question.page_url}; forecasting without it")
        return ""


def make_bot(n_predictions: int, publish: bool, max_tokens: int = 32000) -> FutureEvalBot:
    researcher = "asknews/news-summaries" if asknews_configured() else GeneralLlm(
        model=research_models()[0], temperature=None, timeout=180, allowed_tries=1
    )
    parser = build_parser_llm()
    return FutureEvalBot(
        research_reports_per_question=1,
        predictions_per_research_report=n_predictions,
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        required_successful_predictions=0.01,  # one surviving model is enough to publish
        llms={
            "default": build_roster(n_predictions, max_tokens),
            "summarizer": parser,  # unused (summaries disabled) but must be set
            "researcher": researcher,
            "parser": parser,
        },
    )


# --------------------------------------------------------------------------
# Which tournaments to forecast on
# --------------------------------------------------------------------------
def main_tournaments(today: dt.date) -> list[int | str]:
    """
    Known IDs always, plus whatever the daily maintenance job validated and
    wrote to TARGETS.json. If that file is stale (the daily job broke), add
    only the single most likely current-season slug, to keep runs fast.
    """
    known = maintenance.known_targets()
    try:
        targets = json.loads(maintenance.TARGETS_FILE.read_text())
        age = (today - dt.date.fromisoformat(targets["updated"])).days
        if 0 <= age <= 3:
            return maintenance._dedupe([*known, *targets.get("main", [])])
    except Exception:  # noqa: BLE001 - missing or malformed file
        pass
    in_fall_2026 = today.year == 2026 and today.month >= 9
    guess = [] if in_fall_2026 else maintenance.current_season_slugs(today)[:1]
    return maintenance._dedupe([*known, *guess])


# --------------------------------------------------------------------------
# Spending plan
# --------------------------------------------------------------------------
def plan_for_budget(remaining: float | None) -> tuple[int, int, int, str]:
    """
    Decide how much effort each question gets, from the credits left.

    Metaculus releases credits incrementally: an initial grant, then more if
    MiniBench performance is above average (plus a bonus for open-source bots).
    So when credits are tight, MiniBench keeps the two-model ensemble, because
    that is the tournament that unlocks the rest of the funding, while
    tournament questions drop to one forecast so coverage never hits zero.

    Returns (tournament forecasts, MiniBench forecasts, max_tokens, why).
    """
    mode = os.getenv("ROSTER_MODE", "").strip().lower() or "auto"
    if mode == "full":
        return 3, 2, 32000, "ROSTER_MODE=full"
    if mode == "lean":
        return 2, 1, 32000, "ROSTER_MODE=lean"
    if remaining is None:
        return 2, 1, 32000, "credit balance unknown, assuming lean"
    # Thresholds assume roughly $0.07 per model call, measured over the first
    # week (the original $0.25 estimate was borrowed and far too pessimistic).
    # Remaining season is on the order of 450 questions, so ~$65 at two models
    # each: spending down to about $60 is affordable, and unspent credits are
    # worth nothing.
    if remaining >= 200:
        return 3, 2, 32000, f"${remaining:.0f} left, running full strength"
    if remaining >= 60:
        return 2, 2, 24000, f"${remaining:.0f} left, two models everywhere"
    if remaining >= 25:
        return 1, 2, 12000, f"${remaining:.0f} left, MiniBench prioritised, shorter reasoning"
    if remaining >= 6:
        return 1, 1, 8000, f"${remaining:.0f} left, minimum viable coverage"
    return 0, 0, 8000, f"${remaining:.2f} left, paused"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
async def run(mode: str, publish: bool) -> int:
    # Fresh semaphore bound to this event loop.
    FutureEvalBot._concurrency_limiter = asyncio.Semaphore(FutureEvalBot._max_concurrent_questions)

    remaining = maintenance.fetch_credits_remaining()
    main_preds, mini_preds, max_tokens, note = plan_for_budget(remaining)
    logger.info(
        f"Budget plan: {main_preds} forecast(s) per tournament question, "
        f"{mini_preds} per MiniBench question, max_tokens={max_tokens} ({note})"
    )
    if main_preds == 0 and mini_preds == 0:
        logger.error("Out of credits: skipping this run. Request more through the Metaculus form.")
        return 0

    main_bot = make_bot(main_preds, publish, max_tokens)
    mini_bot = make_bot(max(mini_preds, 1), publish, max_tokens)

    if mode == "test_questions":
        main_bot.skip_previously_forecasted_questions = False
        jobs = [(main_bot, TEST_AREA)]
    else:
        # MiniBench first: it is the cheaper tournament AND the one Metaculus
        # measures when deciding whether to release more credits. If a run is
        # cut short, this is the half we want finished.
        jobs = []
        if mini_preds and os.getenv("SKIP_MINIBENCH", "").lower() not in ("1", "true", "yes"):
            jobs.append((mini_bot, MINIBENCH))
        if main_preds:
            jobs += [(main_bot, t) for t in main_tournaments(dt.date.today())]

    successes, failures = 0, 0
    for bot, tournament in jobs:
        try:
            reports = await bot.forecast_on_tournament(tournament, return_exceptions=True)
        except Exception as e:  # noqa: BLE001 - e.g. a guessed slug that doesn't exist
            logger.info(f"Skipping tournament {tournament!r}: {type(e).__name__}: {str(e)[:200]}")
            continue
        for r in reports:
            if isinstance(r, BaseException):
                failures += 1
                logger.error(f"[{tournament}] question failed: {type(r).__name__}: {str(r)[:500]}")
            else:
                successes += 1
                logger.info(f"[{tournament}] forecast OK: {r.question.page_url}")
    used = main_bot.get_llm("default", "llm").usage_log + mini_bot.get_llm("default", "llm").usage_log
    logger.info(f"Run finished: {successes} forecast(s), {failures} failure(s); models used: {used}")
    left = maintenance.fetch_credits_remaining()
    if remaining is not None and left is not None:
        spent = remaining - left
        per_q = f"${spent / successes:.3f}" if successes else "n/a"
        logger.info(f"Credits: ${left:.2f} left, ${spent:.2f} spent this run, {per_q} per question")

    # Fail the workflow (which makes GitHub email you) only for systematic
    # breakage: something failed and nothing succeeded. Isolated failures are
    # retried automatically on the next run, 20 minutes later.
    return 1 if failures and not successes else 0


async def watch(mode: str, publish: bool, minutes: float, interval_seconds: float) -> int:
    """
    Keep forecasting for a while inside one job.

    GitHub drops most frequent cron triggers: observed gaps between scheduled
    runs were 2.5 to 8 hours, while tournament questions stay open for about
    90 minutes. So instead of trusting the schedule, one run stays alive and
    polls. Cycles cost nothing unless there is a question to forecast.

    Returns the last cycle's exit code.
    """
    end = time.monotonic() + minutes * 60
    rc = await run(mode, publish)
    cycles = 1
    while time.monotonic() + interval_seconds <= end:
        await asyncio.sleep(interval_seconds)
        rc = await run(mode, publish)
        cycles += 1
    logger.info(f"Watch window finished after {cycles} cycle(s)")
    return rc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tournament", "test_questions"], default="tournament")
    ap.add_argument("--dry-run", action="store_true", help="do everything except publish")
    ap.add_argument(
        "--watch-minutes",
        type=float,
        default=float(os.getenv("WATCH_MINUTES") or 0),
        help="keep polling for this many minutes instead of doing one pass",
    )
    ap.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("WATCH_INTERVAL_SECONDS") or 300),
        help="how often to look for new questions while watching",
    )
    args = ap.parse_args()
    for var in ("METACULUS_TOKEN", "OPENROUTER_API_KEY"):
        if not os.getenv(var):
            sys.exit(f"Missing required secret: {var}")
    if args.watch_minutes > 0:
        sys.exit(asyncio.run(watch(args.mode, not args.dry_run, args.watch_minutes, args.interval_seconds)))
    sys.exit(asyncio.run(run(args.mode, publish=not args.dry_run)))
