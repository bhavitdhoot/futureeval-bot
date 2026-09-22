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
RESEARCH_FALLBACKS = [
    "openrouter/openai/gpt-4o-search-preview",
]


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


def build_roster(n_predictions: int) -> RosterLlm:
    primary_specs = _parse_model_list("FORECAST_MODELS", DEFAULT_PRIMARY)
    backup_specs = _parse_model_list("BACKUP_MODELS", DEFAULT_BACKUPS)
    primaries = [make_llm(m, e) for m, e in primary_specs]
    # "full" mode can ask for more predictions than roster members; repeat in order.
    while len(primaries) < n_predictions:
        primaries.append(primaries[len(primaries) % len(primary_specs)])
    return RosterLlm(primaries[:max(n_predictions, 1)], [make_llm(m, e) for m, e in backup_specs])


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
        for model in RESEARCH_FALLBACKS:
            try:
                llm = GeneralLlm(model=model, temperature=None, timeout=180, allowed_tries=1)
                return await llm.invoke(prompt)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Research fallback {model} failed: {e}")
        logger.warning(f"No research available for {question.page_url}; forecasting without it")
        return ""


def make_bot(n_predictions: int, publish: bool) -> FutureEvalBot:
    researcher = "asknews/news-summaries" if asknews_configured() else GeneralLlm(
        model=RESEARCH_FALLBACKS[0], temperature=None, timeout=180, allowed_tries=1
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
            "default": build_roster(n_predictions),
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
# Entry point
# --------------------------------------------------------------------------
async def run(mode: str, publish: bool) -> int:
    # Fresh semaphore bound to this event loop.
    FutureEvalBot._concurrency_limiter = asyncio.Semaphore(FutureEvalBot._max_concurrent_questions)

    full = os.getenv("ROSTER_MODE", "lean").strip().lower() == "full"
    main_bot = make_bot(3 if full else 2, publish)
    mini_bot = make_bot(2 if full else 1, publish)

    if mode == "test_questions":
        main_bot.skip_previously_forecasted_questions = False
        jobs = [(main_bot, TEST_AREA)]
    else:
        jobs = [(main_bot, t) for t in main_tournaments(dt.date.today())]
        if os.getenv("SKIP_MINIBENCH", "").lower() not in ("1", "true", "yes"):
            jobs.append((mini_bot, MINIBENCH))

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

    # Fail the workflow (which makes GitHub email you) only for systematic
    # breakage: something failed and nothing succeeded. Isolated failures are
    # retried automatically on the next run, 20 minutes later.
    return 1 if failures and not successes else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tournament", "test_questions"], default="tournament")
    ap.add_argument("--dry-run", action="store_true", help="do everything except publish")
    args = ap.parse_args()
    for var in ("METACULUS_TOKEN", "OPENROUTER_API_KEY"):
        if not os.getenv(var):
            sys.exit(f"Missing required secret: {var}")
    sys.exit(asyncio.run(run(args.mode, publish=not args.dry_run)))
