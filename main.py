"""
FutureEval tournament bot (Fall 2026 onward).

Built on the official Metaculus template (template_bot.py, kept unchanged for
diffing against upstream). What this file changes, and why:

1. Models. The stock template silently falls back to GPT-4o when only an
   OpenRouter key is set. Spring 2026 results showed frontier models at high
   reasoning effort roughly doubling the template's score, so we run a
   two-vendor roster (OpenAI + Anthropic, the providers the donated
   Metaculus key allows).
2. Fallbacks. Every model call walks a chain: the other roster member, then
   backup models. One provider outage or one retired model slug never costs a
   question. An empty or truncated answer counts as a failure too, so it moves
   down the chain instead of reaching the parser.
3. Prompts (prompts.py). The template never told the model when a question
   resolves, never asked for a base rate, and sent AskNews a long instruction
   prompt as its search query. See prompts.py for the full list.
4. Parsing. Final answers are parsed locally when they follow the requested
   format, with the LLM parser as fallback. A 30k-token forecast is never
   thrown away because two parser samples disagreed or percentiles came back
   out of order.
5. Aggregation. Binary forecasts are pooled with the geometric mean of odds
   rather than the median of probabilities (the better-performing rule in
   studies of Metaculus forecasts).
6. Tournament targeting. Fall 2026 ID, the package constant, and whatever the
   daily maintenance job (maintenance.py) discovers, so the bot rolls into
   future seasons without code edits. A question reachable through two
   tournament IDs is forecast once per cycle.
7. Cost safety. A question that keeps failing is retried a few times per job,
   not every five minutes for five hours.
8. No Metaculus Cup: bots are not prize-eligible there, so it only burns credits.

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
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable

import dotenv

dotenv.load_dotenv()

from forecasting_tools import (  # noqa: E402
    AskNewsSearcher,
    BinaryPrediction,
    BinaryQuestion,
    DatePercentile,
    DateQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (  # noqa: E402
    MonetaryCostManager,
)
from forecasting_tools.data_models.questions import MetaculusQuestion  # noqa: E402

import ledger  # noqa: E402
import maintenance  # noqa: E402
import prompts  # noqa: E402
from template_bot import SummerTemplateBot2026  # noqa: E402

logger = logging.getLogger("forecast-bot")

# --------------------------------------------------------------------------
# Configuration (override any of these with GitHub repo Variables)
# --------------------------------------------------------------------------
MINIBENCH = "minibench"
TEST_AREA = "bot-testing-area"

# Each roster slot lists candidates in order of preference, separated by "|";
# the bot uses the first one OpenRouter currently lists (see resolve_model).
# Put a newer model in front and it is picked up the day it is listed; the
# others are what the slot falls back to if it is retired. The first
# candidates below were all listed on OpenRouter in a live run on 2026-10-29.
# Three vendors, because an ensemble gains most from members that err
# differently ("Diversity is the Strength of the AI Crowd", 2026); with fewer
# forecasts per question the later slots are the first fallbacks.
DEFAULT_PRIMARY = [
    ("openrouter/openai/gpt-6.1-sol|openrouter/openai/gpt-6-sol|openrouter/openai/gpt-5.6-sol", "high"),
    ("openrouter/anthropic/claude-opus-5.5|openrouter/anthropic/claude-opus-5|openrouter/anthropic/claude-opus-4.8", "high"),
    ("openrouter/google/gemini-4-argon|openrouter/google/gemini-3.1-pro-preview", "high"),
]
# The roster for tight budgets: the previous generation, proven in production
# this season at ~$0.10 a question including research. When credits run short
# the bot drops to these at high effort rather than starve the newest models
# of reasoning: Metaculus's Spring 2026 analysis found the same prompt at
# standard effort scored less than half as well as at high effort.
DEFAULT_ECONOMY = [
    ("openrouter/openai/gpt-5.6-sol", "high"),
    ("openrouter/anthropic/claude-opus-4.8", "high"),
]
DEFAULT_BACKUPS = [
    ("openrouter/openai/gpt-5.6-sol", "high"),
    ("openrouter/anthropic/claude-opus-4.8", "high"),
    ("openrouter/openai/gpt-5.5", "high"),
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

# Reasoning efforts OpenRouter accepts, weakest first.
EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh"]

# Output cap per call. Reasoning tokens count toward it, so it is also the
# main cost lever: in a live test the previous generation at high effort cost
# $0.26 a question with 32k against $0.08-0.12 in production at 8k, with the
# same research. A cap that's too low truncates the answer instead (the
# roster then retries once and the parser gets the rest), so budget tiers use
# ECONOMY_MAX_TOKENS, 25% above production's proven 8k.
MAX_TOKENS = 32000
ECONOMY_MAX_TOKENS = 10000


def _env(name: str) -> str:
    # GitHub passes unset repo variables as "", so a plain getenv default is not enough.
    return os.getenv(name, "").strip()


# OpenRouter's catalog, read once per process; a failed read is retried after
# CATALOG_RETRY_SECONDS rather than pinning a 5-hour job to the fallbacks.
_catalog_cache: set[str] | None = None
_catalog_read_at: float | None = None
CATALOG_RETRY_SECONDS = 600
_resolved_logged: set[str] = set()


def _catalog() -> set[str] | None:
    """OpenRouter's public model list. None if it can't be read right now."""
    global _catalog_cache, _catalog_read_at
    stale = _catalog_read_at is None or (
        _catalog_cache is None and time.monotonic() - _catalog_read_at >= CATALOG_RETRY_SECONDS
    )
    if stale:
        _catalog_read_at = time.monotonic()
        _catalog_cache = maintenance.fetch_available_models()
    return _catalog_cache


def resolve_model(spec: str) -> str:
    """
    The first candidate in "a|b|c" that OpenRouter lists. If the catalog can't
    be read (or lists none of them), the first candidate: it is the preferred
    one, and if it fails at call time the roster's fallback chain takes over.
    """
    candidates = [c.strip() for c in spec.split("|") if c.strip()]
    if len(candidates) <= 1:
        return candidates[0] if candidates else spec
    chosen = candidates[0]
    available = _catalog()
    if available:
        listed = [c for c in candidates if maintenance.openrouter_id(c) in available]
        if listed:
            chosen = listed[0]
        else:
            logger.warning(f"None of {candidates} is listed on OpenRouter; trying {chosen}")
    if spec not in _resolved_logged:
        _resolved_logged.add(spec)
        logger.info(f"Model slot {spec!r} -> {chosen}")
    return chosen


def research_model_specs() -> list[str]:
    """RESEARCH_MODELS as configured, candidates unresolved."""
    return [m.strip() for m in _env("RESEARCH_MODELS").split(",") if m.strip()] or DEFAULT_RESEARCH


def research_models() -> list[str]:
    return [resolve_model(m) for m in research_model_specs()]


def _parse_model_list(env_name: str, default: list[tuple[str, str | None]]):
    """
    FORECAST_MODELS="openrouter/openai/x:high,openrouter/anthropic/y:high".
    Only a known reasoning effort counts as a suffix, so "model:online" or
    "model:free" keep their OpenRouter variant instead of becoming an effort.
    """
    raw = _env(env_name)
    if not raw:
        return default
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, suffix = item.rpartition(":")
        if sep and "/" not in suffix and (suffix.lower() in EFFORTS or suffix == ""):
            out.append((name, suffix.lower() or None))
        else:
            out.append((item, None))
    return out or default


def cap_effort(effort: str | None, cap: str | None) -> str | None:
    """Lower a configured effort to `cap`; never raise it."""
    if not effort or not cap or effort.lower() not in EFFORTS or cap not in EFFORTS:
        return effort
    return effort if EFFORTS.index(effort.lower()) <= EFFORTS.index(cap) else cap


def make_llm(model: str, effort: str | None = None, **overrides) -> GeneralLlm:
    params = dict(temperature=None, timeout=480, allowed_tries=2, max_tokens=MAX_TOKENS)
    params.update(overrides)
    if effort and effort.lower() != "none":
        params["reasoning"] = {"effort": effort}
    return GeneralLlm(model=resolve_model(model), **params)


def _identity(llm: GeneralLlm) -> tuple[str, str]:
    return llm.model, repr(llm.litellm_kwargs.get("reasoning"))


class RosterLlm(GeneralLlm):
    """
    Looks like one GeneralLlm to the framework, but:
      - rotates the first-choice model across repeated calls with the same
        `rotation_key` (the question; by default the prompt), so prediction #1
        for a question goes to model A, #2 to model B, etc.;
      - on any error or an empty answer, falls through the remaining roster,
        then the backups;
      - an answer that fails `validate` (no final forecast where one was
        asked for) gets one more model's attempt. If that misses too, the
        first answer goes on to the parser rather than paying for the whole
        chain: the check is a heuristic, the parser is the real judge.
    The framework then aggregates the predictions, which is what gives us a
    real ensemble.
    """

    def __init__(self, primaries: list[GeneralLlm], backups: list[GeneralLlm]):
        if not primaries:
            raise ValueError("RosterLlm needs at least one model")
        super().__init__(model=primaries[0].model, temperature=None)
        self.primaries = primaries
        self.backups = backups
        self._calls: dict[str, int] = defaultdict(int)
        self.usage_log: list[str] = []  # which model actually answered

    def chain(self, k: int) -> list[GeneralLlm]:
        """Models to try for the k-th identical prompt, without repeats."""
        n = len(self.primaries)
        rotated = self.primaries[k % n :] + self.primaries[: k % n]
        out, seen = [], set()
        for llm in rotated + self.backups:
            if _identity(llm) not in seen:
                seen.add(_identity(llm))
                out.append(llm)
        return out

    async def invoke(
        self,
        prompt,
        system_prompt: str | None = None,
        validate: Callable[[str], bool] | None = None,
        rotation_key: str | None = None,
    ) -> str:
        # Keyed on the question when the caller says which: prompts carry the
        # current time, so two predictions built either side of a minute
        # boundary would otherwise both start with model A.
        key = rotation_key or hashlib.sha256(repr((prompt, system_prompt)).encode()).hexdigest()
        k = self._calls[key]
        self._calls[key] += 1  # incremented before any await: safe under asyncio
        errors: list[str] = []
        unvalidated: tuple[str, str] | None = None  # (model, answer) of the first miss
        for i, llm in enumerate(self.chain(k)):
            try:
                if system_prompt is None:
                    answer = await llm.invoke(prompt)
                else:
                    answer = await llm.invoke(prompt, system_prompt=system_prompt)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty answer")
            except Exception as e:  # noqa: BLE001 - any failure means try the next model
                errors.append(f"{llm.model}: {type(e).__name__}: {str(e)[:300]}")
                continue
            if validate is not None and not validate(answer):
                errors.append(f"{llm.model}: no final forecast (ends: {answer[-150:]!r})")
                if unvalidated is None:
                    unvalidated = (llm.model, answer)
                    continue
                break  # second miss: stop paying, let the parser try the first
            if i > 0:
                logger.warning(f"Fallback used: {llm.model} answered after {i} failure(s): {errors}")
            self.usage_log.append(llm.model)
            return answer
        if unvalidated is not None:
            logger.warning(f"Using {unvalidated[0]}'s answer unvalidated after: {errors}")
            self.usage_log.append(unvalidated[0])
            return unvalidated[1]
        raise RuntimeError("Every model in the chain failed: " + " | ".join(errors))


def build_roster(
    n_predictions: int,
    effort_cap: str | None = None,
    economy: bool = False,
    max_tokens: int = MAX_TOKENS,
) -> RosterLlm:
    if economy:
        primary_specs = _parse_model_list("ECONOMY_MODELS", DEFAULT_ECONOMY)
    else:
        primary_specs = _parse_model_list("FORECAST_MODELS", DEFAULT_PRIMARY)
    backup_specs = _parse_model_list("BACKUP_MODELS", DEFAULT_BACKUPS)
    roster = [make_llm(m, cap_effort(e, effort_cap), max_tokens=max_tokens) for m, e in primary_specs]
    backups = [make_llm(m, cap_effort(e, effort_cap), max_tokens=max_tokens) for m, e in backup_specs]
    n = max(n_predictions, 1)
    # "full" mode can ask for more predictions than roster members; repeat in order.
    primaries = [roster[i % len(roster)] for i in range(n)]
    # Roster members left out of rotation are still the best fallbacks we have.
    return RosterLlm(primaries, roster[n:] + backups)


def build_parser_llm() -> RosterLlm:
    names = [m for m, _ in _parse_model_list("PARSER_MODELS", [(p, None) for p in DEFAULT_PARSERS])]
    llms = [GeneralLlm(model=resolve_model(m), temperature=None, timeout=120, allowed_tries=2) for m in names]
    return RosterLlm(llms[:1], llms[1:])


def asknews_configured() -> bool:
    return bool(
        (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
        or os.getenv("ASKNEWS_API_KEY")
    )


# --------------------------------------------------------------------------
# Parsing and aggregation
# --------------------------------------------------------------------------
_NUM = r"-?\d+(?:\.\d+)?"
# Anchored to the start of a line, so "the prior probability: 10%" mid-sentence
# is never mistaken for the final answer.
_PROBABILITY = re.compile(rf"^[\s*#>\-\"']*Probability\s*\**\s*[:=]\s*\**\s*({_NUM})\s*%", re.I | re.M)
_PERCENTILE_NUMBER = re.compile(
    rf"^[\s*\-•\"'>]*Percentile\s*(\d{{1,2}})\s*[:=]\s*\**\s*({_NUM})\s*\**\s*(?:\([^)]*\))?[\s\"']*$",
    re.I | re.M,
)
_PERCENTILE_DATE = re.compile(
    r"^[\s*\-•\"'>]*Percentile\s*(\d{1,2})\s*[:=]\s*\**\s*"
    r"(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?)\s*\**\s*(?:\([^)]*\))?[\s\"']*$",
    re.I | re.M,
)
REQUIRED_PERCENTILES = (10, 20, 40, 60, 80, 90)
TAIL_PERCENTILES = (5, 95)  # asked for, but an answer without them still parses


def parse_binary(text: str) -> float | None:
    """The last "Probability: ZZ%" in the text, as a decimal, or None."""
    found = _PROBABILITY.findall(text)
    if not found:
        return None
    value = float(found[-1]) / 100
    return value if 0 <= value <= 1 else None


_OPTION_LINE = re.compile(
    rf"^\s*(?:[-*•]\s+|\d+[.)]\s+)?[*_`\"']*(.+?)[*_`\"']*\s*[:=]\s*\**\s*({_NUM})\s*%",
    re.M,
)


def _option_name(text: str) -> str:
    return " ".join(text.split()).casefold()


def parse_multiple_choice(text: str, options: list[str]) -> PredictedOptionList | None:
    """
    One "Option: XX%" line per option, using the exact option names, or None.
    Names must match exactly (ignoring case, spacing and markdown), so option
    "10" never picks up the line for ">10" or "10+".
    """
    wanted = {_option_name(o): o for o in options}
    if len(wanted) != len(options):
        return None  # options that differ only by case or spacing: leave it to the parser
    probs = {}
    for name, value in _OPTION_LINE.findall(text):
        option = wanted.get(_option_name(name))
        if option is not None:
            probs[option] = float(value)  # later lines win: the final answer comes last
    if len(probs) != len(options):
        return None
    total = sum(probs.values())
    if not 95 <= total <= 105:
        return None
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=o, probability=p / total) for o, p in probs.items()
        ]
    )


def _last_by_percentile(matches) -> dict[int, str]:
    out = {}
    for pct, value in matches:
        out[int(pct)] = value  # later lines win: the final answer comes last
    return out


def _wanted(found: dict[int, str]) -> list[int] | None:
    if not all(p in found for p in REQUIRED_PERCENTILES):
        return None
    return sorted({*REQUIRED_PERCENTILES, *(p for p in TAIL_PERCENTILES if p in found)})


def parse_numeric(text: str) -> list[Percentile] | None:
    found = _last_by_percentile(_PERCENTILE_NUMBER.findall(text))
    wanted = _wanted(found)
    if wanted is None:
        return None
    return [Percentile(percentile=p / 100, value=float(found[p])) for p in wanted]


def parse_date(text: str) -> list[Percentile] | None:
    found = _last_by_percentile(_PERCENTILE_DATE.findall(text))
    wanted = _wanted(found)
    if wanted is None:
        return None
    out = []
    for p in wanted:
        try:
            when = dt.datetime.fromisoformat(found[p].replace(" ", "T"))
        except ValueError:
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.timezone.utc)
        out.append(Percentile(percentile=p / 100, value=when.timestamp()))
    return out


def repair_percentiles(percentiles: list[Percentile]) -> list[Percentile]:
    """
    Values listed in the wrong order otherwise discard the whole forecast (the
    template prompt warns models about it for a reason). Pairing sorted values
    with sorted percentiles keeps the intended distribution.
    """
    pcts = sorted(p.percentile for p in percentiles)
    values = [p.value for p in sorted(percentiles, key=lambda p: p.percentile)]
    if values != sorted(values):
        logger.warning(f"Percentile values out of order, sorting them: {values}")
        values = sorted(values)
    return [Percentile(percentile=p, value=v) for p, v in zip(pcts, values)]


# The framework rejects a whole distribution if any declared value lies more
# than twice the question's range beyond a bound.
_BOUND_MARGIN = 1.9


def fit_to_bounds(percentiles: list[Percentile], question) -> list[Percentile]:
    """
    Pull tails that run far past the question's range back to just inside the
    framework's limit, instead of losing the forecast. Seen live: a date
    question with an open bound in 2027 got 90th/95th percentiles in 2038 and
    2045. Only the shape far beyond the bound changes; the probability mass
    beyond it, which is what gets scored, barely moves.
    """
    lower, upper = question.lower_bound, question.upper_bound
    if isinstance(question, DateQuestion):
        lower, upper = lower.timestamp(), upper.timestamp()
    span = upper - lower
    lo, hi = lower - _BOUND_MARGIN * span, upper + _BOUND_MARGIN * span
    zero_point = getattr(question, "zero_point", None)
    if zero_point is not None:
        lo = max(lo, zero_point)
    fitted = [Percentile(percentile=p.percentile, value=min(max(p.value, lo), hi)) for p in percentiles]
    if fitted != percentiles:
        logger.warning(f"Pulled far-out tail values inside the question range: {[p.value for p in percentiles]}")
    return fitted


def pool_binary(predictions: list[float]) -> float:
    """
    Geometric mean of odds, optionally extremized (BINARY_EXTREMIZE, default 1
    = none). Unlike a median of two, it lets a confident forecaster pull the
    pool toward the tail instead of splitting the difference in probability
    space, which is what log scoring rewards when the confidence is earned.
    """
    try:
        factor = float(_env("BINARY_EXTREMIZE") or 1.0)
    except ValueError:
        # A typo in a repo variable must not block every paid-for binary forecast.
        logger.warning(f"Ignoring unparsable BINARY_EXTREMIZE={_env('BINARY_EXTREMIZE')!r}")
        factor = 1.0
    clipped = [min(max(p, 0.001), 0.999) for p in predictions]
    mean_logit = sum(math.log(p / (1 - p)) for p in clipped) / len(clipped)
    pooled = 1 / (1 + math.exp(-mean_logit * factor))
    return min(max(pooled, 0.01), 0.99)


def _validator(question: MetaculusQuestion) -> Callable[[str], bool]:
    """
    Cheap check that the model reached its final answer. It catches truncated
    or empty-content replies before they are paid for twice (once here, once
    by a parser that can only fail). Deliberately looser than the parsers, but
    it only looks at the end, where the final answer was asked to be.
    """
    if isinstance(question, BinaryQuestion):
        return lambda text: bool(re.search(rf"{_NUM}\s*%", text[-1500:]))
    if isinstance(question, (NumericQuestion, DateQuestion)):
        return lambda text: bool(re.search(r"percentile|\b9[05](?:th)?\b", text[-1500:], re.I))
    return lambda text: bool(re.search(rf"{_NUM}\s*%", text[-3000:]))


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------
class FutureEvalBot(SummerTemplateBot2026):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(2)  # replaced per event loop in run()
    _web_limiter = asyncio.Semaphore(4)  # likewise

    # Set by make_bot. Kept off __init__ because the framework builds each
    # forecast's published config from the __init__ signature.
    use_asknews = False
    research_costs: dict[str, float]  # question key -> USD, set per bot in make_bot
    # With AskNews configured, also run web search (news alone misses base
    # rates, official data and schedules). Without AskNews, web search is the
    # only source and always runs.
    web_research = True

    # ------------------------------ research ------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
        # Exact research cost (OpenRouter reports each call's cost; the
        # manager is per task, so concurrent questions don't mix). Feeds the
        # ledger's research/forecast split.
        with MonetaryCostManager() as cost:
            research = await self._run_research(question)
        self.research_costs[question_key(question)] = self.research_costs.get(question_key(question), 0.0) + cost.current_usage
        return research

    async def _run_research(self, question: MetaculusQuestion) -> str:
        jobs = {}
        if self.use_asknews:
            jobs["Recent news (AskNews)"] = self._asknews_research(question)
        if self.web_research or not self.use_asknews:
            jobs["Web research"] = self._web_research(question)
        results = dict(zip(jobs, await asyncio.gather(*jobs.values())))
        if self.use_asknews and not any(results.values()) and "Web research" not in results:
            logger.warning(f"No AskNews results for {question.page_url}; trying web search")
            results["Web research"] = await self._web_research(question)
        sections = [f"## {title}\n{text.strip()}" for title, text in results.items() if text]
        if not sections:
            logger.warning(f"No research available for {question.page_url}; forecasting without it")
        research = "\n\n".join(sections)
        logger.info(f"Research for {question.page_url}:\n{research}")
        return research

    async def _asknews_research(self, question: MetaculusQuestion) -> str:
        try:
            async with self._concurrency_limiter:  # AskNews free tier is rate limited
                text = await AskNewsSearcher().get_formatted_news_async(prompts.asknews_query(question))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"AskNews failed for {question.page_url}: {type(e).__name__}: {e}")
            return ""
        if not text or "No articles were found" in text:
            return ""
        return text

    async def _web_research(self, question: MetaculusQuestion) -> str:
        # A MiniBench batch opens many questions at once; searching for all of
        # them in parallel invites rate limits, which would empty the research.
        async with self._web_limiter:
            return await self._web_research_unthrottled(question)

    async def _web_research_unthrottled(self, question: MetaculusQuestion) -> str:
        prompt = prompts.web_research_prompt(question)
        for model in research_models():
            # Low effort: searching and summarising don't need deep reasoning,
            # and reasoning tokens are billed as output. Retried without the
            # parameter if a (non-reasoning) model rejects it.
            for reasoning in ({"effort": "low"}, None):
                extra = {"reasoning": reasoning} if reasoning else {}
                try:
                    llm = GeneralLlm(model=model, temperature=None, timeout=180, allowed_tries=1, **extra)
                    text = await llm.invoke(prompt)
                    if text and text.strip():
                        return text
                    logger.warning(f"Web research {model} returned nothing")
                    break
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Web research {model} failed: {type(e).__name__}: {e}")
                    if not (reasoning and "reasoning" in str(e).lower()):
                        break
        return ""

    # ------------------------------ forecasting ------------------------------
    async def _reason(self, question: MetaculusQuestion, prompt: str) -> str:
        reasoning = await self.get_llm("default", "llm").invoke(
            prompt, validate=_validator(question), rotation_key=f"forecast:{question_key(question)}"
        )
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        return reasoning

    async def _llm_parse(self, reasoning: str, output_type, instructions: str | None = None):
        """
        The LLM parser, for answers that don't follow the requested format.
        If the validation samples disagree, take a single sample rather than
        discard a forecast that cost far more than the parse.
        """
        kwargs = dict(model=self.get_llm("parser", "llm"), additional_instructions=instructions)
        try:
            return await structure_output(
                reasoning, output_type, num_validation_samples=self._structure_output_validation_samples, **kwargs
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Validated parse failed ({type(e).__name__}: {str(e)[:300]}); using one sample")
            return await structure_output(reasoning, output_type, num_validation_samples=1, **kwargs)

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str):
        note = self._get_conditional_disclaimer_if_necessary(question)
        return await self._binary_prompt_to_forecast(question, prompts.binary_prompt(question, research, note))

    async def _binary_prompt_to_forecast(self, question: BinaryQuestion, prompt: str):
        reasoning = await self._reason(question, prompt)
        value = parse_binary(reasoning)
        if value is None:
            value = (await self._llm_parse(reasoning, BinaryPrediction)).prediction_in_decimal
        value = max(0.01, min(0.99, value))
        logger.info(f"Forecasted URL {question.page_url} with prediction: {value}.")
        return ReasonedPrediction(prediction_value=value, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(self, question: MultipleChoiceQuestion, research: str):
        note = self._get_conditional_disclaimer_if_necessary(question)
        return await self._multiple_choice_prompt_to_forecast(
            question, prompts.multiple_choice_prompt(question, research, note)
        )

    async def _multiple_choice_prompt_to_forecast(self, question: MultipleChoiceQuestion, prompt: str):
        reasoning = await self._reason(question, prompt)
        options = parse_multiple_choice(reasoning, question.options)
        if options is None:
            instructions = clean_indents(
                f"""
                Make sure that all option names are one of the following:
                {question.options}

                The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
                Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
                """
            )
            options = await self._llm_parse(reasoning, PredictedOptionList, instructions)
        logger.info(f"Forecasted URL {question.page_url} with prediction: {options}.")
        return ReasonedPrediction(prediction_value=options, reasoning=reasoning)

    async def _run_forecast_on_numeric(self, question: NumericQuestion, research: str):
        bounds = self._create_upper_and_lower_bound_messages(question)
        note = self._get_conditional_disclaimer_if_necessary(question)
        return await self._numeric_prompt_to_forecast(
            question, prompts.numeric_prompt(question, research, bounds, note)
        )

    async def _numeric_prompt_to_forecast(self, question: NumericQuestion, prompt: str):
        reasoning = await self._reason(question, prompt)
        percentiles = parse_numeric(reasoning)
        if percentiles is None:
            instructions = clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a numeric question.
                - This text is trying to answer the numeric question: "{question.question_text}".
                - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
                - The units for the forecast are: {question.unit_of_measure}
                - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                - Turn any values that are in scientific notation into regular numbers.
                """
            )
            percentiles = await self._llm_parse(reasoning, list[Percentile], instructions)
        prediction = NumericDistribution.from_question(
            fit_to_bounds(repair_percentiles(percentiles), question), question
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}.")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _run_forecast_on_date(self, question: DateQuestion, research: str):
        bounds = self._create_upper_and_lower_bound_messages(question)
        note = self._get_conditional_disclaimer_if_necessary(question)
        return await self._date_prompt_to_forecast(question, prompts.date_prompt(question, research, bounds, note))

    async def _date_prompt_to_forecast(self, question: DateQuestion, prompt: str):
        reasoning = await self._reason(question, prompt)
        percentiles = parse_date(reasoning)
        if percentiles is None:
            instructions = clean_indents(
                f"""
                The text given to you is trying to give a forecast distribution for a date question.
                - This text is trying to answer the question: "{question.question_text}".
                - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
                - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
                """
            )
            dates: list[DatePercentile] = await self._llm_parse(reasoning, list[DatePercentile], instructions)
            percentiles = [Percentile(percentile=d.percentile, value=d.value.timestamp()) for d in dates]
        prediction = NumericDistribution.from_question(
            fit_to_bounds(repair_percentiles(percentiles), question), question
        )
        logger.info(f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}.")
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    async def _aggregate_predictions(self, predictions, question: MetaculusQuestion):
        if isinstance(question, BinaryQuestion) and predictions and all(
            isinstance(p, (int, float)) for p in predictions
        ):
            return pool_binary([float(p) for p in predictions])
        return await super()._aggregate_predictions(predictions, question)


def make_bot(
    n_predictions: int,
    publish: bool,
    effort_cap: str | None = None,
    web_research: bool = True,
    economy: bool = False,
) -> FutureEvalBot:
    use_asknews = asknews_configured()
    # Recorded in each forecast's metadata; run_research picks the sources itself.
    researcher = "asknews/news-summaries" if use_asknews else GeneralLlm(
        model=research_models()[0], temperature=None, timeout=180, allowed_tries=1
    )
    parser = build_parser_llm()
    bot = FutureEvalBot(
        research_reports_per_question=1,
        predictions_per_research_report=max(n_predictions, 1),
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        required_successful_predictions=0.01,  # one surviving model is enough to publish
        llms={
            "default": build_roster(
                n_predictions, effort_cap, economy, ECONOMY_MAX_TOKENS if economy else MAX_TOKENS
            ),
            "summarizer": parser,  # unused (summaries disabled) but must be set
            "researcher": researcher,
            "parser": parser,
        },
    )
    bot.use_asknews = use_asknews
    bot.web_research = web_research
    bot.research_costs = {}
    return bot


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
def season_end(today: dt.date) -> dt.date:
    """
    Roughly when the current season stops opening questions. Metaculus stops
    "a few weeks before" the next season starts (Summer 2026 announcement), so
    the season boundary is a safe upper estimate: assuming too late wastes a
    little accuracy, assuming too early runs the credits dry while questions
    are still opening.
    """
    override = _env("SEASON_END_DATE")
    if override:
        try:
            return dt.date.fromisoformat(override)
        except ValueError:
            logger.warning(f"Ignoring unparsable SEASON_END_DATE={override!r}")
    season = maintenance.season_of(today)
    if season == "fall":
        return dt.date(today.year, 12, 31)
    if season == "spring":
        return dt.date(today.year, 4, 30)
    return dt.date(today.year, 8, 31)


# Never plan as if fewer days remain than this. Near the estimated season end
# the per-question budget would otherwise balloon and spend everything on the
# last few questions, leaving nothing if questions keep opening a while longer.
MIN_DAYS_LEFT = 7


def affordable_per_question(remaining: float, today: dt.date, questions_per_day: float) -> float:
    days_left = max((season_end(today) - today).days, MIN_DAYS_LEFT)
    return remaining / max(days_left * questions_per_day, 1)


@dataclass(frozen=True)
class Plan:
    main: int  # forecasts per seasonal-tournament question
    mini: int  # forecasts per MiniBench question
    effort_cap: str | None  # None = each model's configured reasoning effort
    web_research: bool  # run web search alongside AskNews
    why: str
    economy: bool = False  # previous-generation roster (DEFAULT_ECONOMY)

    @property
    def effort_label(self) -> str:
        """The ledger bucket this plan's spend is filed under."""
        return ledger.bucket(self.economy, self.effort_cap)


# Richest first. The planner takes the first one the credits can sustain until
# the season ends. MiniBench keeps two forecasts longest: it is the tournament
# Metaculus watches when deciding whether to release more credits.
PLANS = [
    # (main, mini, economy roster, effort cap, web research beside AskNews, label)
    (3, 2, False, None, True, "full strength"),
    (2, 2, False, None, True, "two flagship models everywhere"),
    (1, 2, False, None, False, "flagship, MiniBench keeps the ensemble"),
    (1, 1, False, None, False, "one flagship forecast"),
    (1, 2, True, None, False, "previous generation, MiniBench keeps the ensemble"),
    (1, 1, True, None, False, "one previous-generation forecast"),
    (1, 1, True, "medium", False, "minimum viable coverage"),
]
# The plan is re-chosen every five minutes from the live balance, so running
# slightly hot corrects itself; this stops it flapping between two plans.
SPEND_SLACK = 1.1


def plan_for_budget(
    remaining: float | None, today: dt.date | None = None, stats: ledger.Stats | None = None
) -> Plan:
    """
    Decide how much forecasting each question gets, from the credits left, the
    days left in the season, and what forecasts have actually cost.

    Costs and the question rate come from the ledger (measured), or from
    priors taken from production runs until enough has been measured. Savings
    come first from fewer forecasts, then from the previous model generation
    (with a tighter token cap, see ECONOMY_MAX_TOKENS), and only last from
    lower reasoning effort.
    """
    mode = _env("ROSTER_MODE").lower() or "auto"
    if mode == "full":
        return Plan(3, 2, None, True, "ROSTER_MODE=full")
    # Lean and "don't know" use the proven, cheaper previous generation: an
    # outage of the balance API must not triple the spend per question.
    if mode == "lean":
        return Plan(2, 1, None, False, "ROSTER_MODE=lean", economy=True)
    if remaining is None:
        return Plan(2, 1, None, False, "credit balance unknown, assuming lean", economy=True)
    # Spend per question, not per week. Running out in December would mean zero
    # coverage while questions are still resolving, which costs far more than a
    # smaller ensemble does. A question forecast with one model still scores;
    # a question never forecast scores nothing.
    if remaining < 6:
        return Plan(0, 0, None, False, f"${remaining:.2f} left, paused to avoid half-finished runs")
    stats = stats or ledger.Ledger().stats()
    rate = float(_env("EXPECTED_QUESTIONS_PER_DAY") or 0) or stats.questions_per_day
    budget = affordable_per_question(remaining, today or dt.date.today(), rate)
    share = stats.minibench_share
    for main_n, mini_n, economy, cap, web, label in PLANS:
        slots = main_n * (1 - share) + mini_n * share
        key = ledger.bucket(economy, cap)
        # A bucket with no measurement and no prior is treated as unaffordable.
        cost = stats.plan_cost(key, slots)
        if cost <= budget * SPEND_SLACK or label == PLANS[-1][-1]:
            note = (
                f"${remaining:.0f} left, ${budget:.3f} affordable per question at {rate:.1f}/day, "
                f"this plan ~${cost:.3f} ({stats.why}), {label}"
            )
            return Plan(main_n, mini_n, cap, web, note, economy)
    raise AssertionError("unreachable: the last plan is always accepted")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
# Per-process memory across watch cycles: how often each question was tried,
# and which ones already succeeded. Without it a question that fails for a
# persistent reason (or any question in a dry run, where nothing is marked as
# forecast on Metaculus) would be re-forecast every cycle, paying each time.
_attempts: Counter[str] = Counter()
_done: set[str] = set()
# The balance after the last forecast batch. Charges can post a few seconds
# late; whatever left the balance since then belongs to that batch.
_balance_mark: float | None = None


def max_attempts() -> int:
    return int(_env("MAX_ATTEMPTS_PER_QUESTION") or 3)


def question_key(q: MetaculusQuestion) -> str:
    return str(q.id_of_question or q.id_of_post or q.page_url)


def fetch_open_questions(bot: FutureEvalBot, tournament: int | str) -> list[MetaculusQuestion]:
    return bot.metaculus_client.get_all_open_questions_from_tournament(tournament)


def select_questions(
    bot: FutureEvalBot, questions: list[MetaculusQuestion], seen: set[str]
) -> list[MetaculusQuestion]:
    """Drop questions already handled this cycle, already forecast, or out of retries."""
    out = []
    for q in questions:
        key = question_key(q)
        if key in seen:
            continue  # same question reached through another tournament ID
        seen.add(key)
        if key in _done or (bot.skip_previously_forecasted_questions and q.already_forecasted):
            continue
        if _attempts[key] >= max_attempts():
            if _attempts[key] == max_attempts():
                logger.error(f"Giving up on {q.page_url} for this job after {_attempts[key]} failed attempts")
                _attempts[key] += 1  # log once
            continue
        out.append(q)
    return out


async def run(mode: str, publish: bool) -> int:
    # Fresh semaphore bound to this event loop.
    FutureEvalBot._concurrency_limiter = asyncio.Semaphore(FutureEvalBot._max_concurrent_questions)
    FutureEvalBot._web_limiter = asyncio.Semaphore(4)

    # Practice runs on the testing area would distort the measured question
    # rate and costs, so only tournament runs feed the ledger.
    live = mode != "test_questions"
    global _balance_mark
    book = ledger.Ledger.load()
    remaining = maintenance.fetch_credits_remaining()
    if live and _balance_mark is not None and remaining is not None:
        book.add_late_charges(_balance_mark - remaining)
        _balance_mark = None
    plan = plan_for_budget(remaining, dt.date.today(), book.stats())
    logger.info(
        f"Budget plan: {plan.main} forecast(s) per tournament question, {plan.mini} per MiniBench "
        f"question, {'previous-generation' if plan.economy else 'flagship'} models, "
        f"effort cap {plan.effort_cap or 'none'}, web research "
        f"{'on' if plan.web_research else 'fallback only'} ({plan.why})"
    )
    if plan.main == 0 and plan.mini == 0:
        logger.error("Out of credits: skipping this run. Request more through the Metaculus form.")
        return 0

    main_bot = make_bot(plan.main, publish, plan.effort_cap, plan.web_research, plan.economy)
    mini_bot = make_bot(plan.mini, publish, plan.effort_cap, plan.web_research, plan.economy)

    if mode == "test_questions":
        main_bot.skip_previously_forecasted_questions = False
        jobs = [(main_bot, TEST_AREA)]
    else:
        # MiniBench first: it is the cheaper tournament AND the one Metaculus
        # measures when deciding whether to release more credits. If a run is
        # cut short, this is the half we want finished.
        jobs = []
        if plan.mini and _env("SKIP_MINIBENCH").lower() not in ("1", "true", "yes"):
            jobs.append((mini_bot, MINIBENCH))
        if plan.main:
            jobs += [(main_bot, t) for t in main_tournaments(dt.date.today())]

    successes, failures, seen = 0, 0, set()
    for bot, tournament in jobs:
        try:
            open_questions = fetch_open_questions(bot, tournament)
        except Exception as e:  # noqa: BLE001 - e.g. a guessed slug that doesn't exist
            logger.info(f"Skipping tournament {tournament!r}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if live:
            # Count each question under the first tournament it was seen in,
            # so one reachable through two IDs isn't counted twice.
            for q in open_questions:
                if question_key(q) not in seen:
                    book.record_seen(question_key(q), "minibench" if tournament == MINIBENCH else "main")
        questions = select_questions(bot, open_questions, seen)
        if not questions:
            continue
        for q in questions:
            _attempts[question_key(q)] += 1
        # Chain readings: a batch starts from the previous batch's closing
        # balance, so a late charge lands in the next batch, not in a gap.
        before = (_balance_mark if _balance_mark is not None else maintenance.fetch_credits_remaining()) if live else None
        # select_questions already applied the framework's skip filter, so
        # reports line up one-to-one with questions.
        reports = await bot.forecast_questions(questions, return_exceptions=True)
        after = maintenance.fetch_credits_remaining() if live else None
        research_usd, tracked_usd = log_costs(bot, questions, reports)
        if before is not None and after is not None:
            slots = plan.mini if bot is mini_bot else plan.main
            share = research_usd / tracked_usd if tracked_usd > 0 else None
            book.record_spend(plan.effort_label, slots, len(questions), before - after, research_share=share)
            _balance_mark = after
        for q, r in zip(questions, reports):
            if isinstance(r, BaseException):
                failures += 1
                logger.error(f"[{tournament}] question failed: {type(r).__name__}: {str(r)[:500]}")
            else:
                successes += 1
                _done.add(question_key(q))
                logger.info(f"[{tournament}] forecast OK: {q.page_url}")
    used = Counter(
        main_bot.get_llm("default", "llm").usage_log + mini_bot.get_llm("default", "llm").usage_log
    )
    logger.info(f"Run finished: {successes} forecast(s), {failures} failure(s); models used: {dict(used)}")
    if successes or failures:
        left = maintenance.fetch_credits_remaining()
        if remaining is not None and left is not None:
            spent = remaining - left
            per_q = f"${spent / successes:.3f}" if successes else "n/a"
            logger.info(f"Credits: ${left:.2f} left, ${spent:.2f} spent this run, {per_q} per question")

    if live:
        book.save()

    # Fail the workflow (which makes GitHub email you) only for systematic
    # breakage: something failed and nothing succeeded. Isolated failures are
    # retried automatically on the next cycles.
    return 1 if failures and not successes else 0


def log_costs(bot: FutureEvalBot, questions, reports) -> tuple[float, float]:
    """
    Log each question's exact cost, split into research and forecasting
    (from OpenRouter's per-call cost reports). Returns (research, total) USD
    over the questions that produced a report.
    """
    research_total = tracked_total = 0.0
    for q, r in zip(questions, reports):
        if isinstance(r, BaseException):
            continue
        total = float(getattr(r, "price_estimate", 0) or 0)
        research = min(bot.research_costs.get(question_key(q), 0.0), total)
        research_total += research
        tracked_total += total
        logger.info(
            f"Cost for {q.page_url}: ${total:.3f} (research ${research:.3f}, forecasting ${total - research:.3f})"
        )
    return research_total, tracked_total


async def _safe_run(mode: str, publish: bool) -> int:
    try:
        return await run(mode, publish)
    except Exception as e:  # noqa: BLE001 - one bad cycle must not end the watch window
        logger.exception(f"Cycle crashed: {type(e).__name__}: {e}")
        return 1


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
    rc = await _safe_run(mode, publish)
    cycles = 1
    while time.monotonic() + interval_seconds <= end:
        await asyncio.sleep(interval_seconds)
        rc = await _safe_run(mode, publish)
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
        default=float(_env("WATCH_MINUTES") or 0),
        help="keep polling for this many minutes instead of doing one pass",
    )
    ap.add_argument(
        "--interval-seconds",
        type=float,
        default=float(_env("WATCH_INTERVAL_SECONDS") or 300),
        help="how often to look for new questions while watching",
    )
    args = ap.parse_args()
    for var in ("METACULUS_TOKEN", "OPENROUTER_API_KEY"):
        if not os.getenv(var):
            sys.exit(f"Missing required secret: {var}")
    if args.watch_minutes > 0:
        sys.exit(asyncio.run(watch(args.mode, not args.dry_run, args.watch_minutes, args.interval_seconds)))
    sys.exit(asyncio.run(run(args.mode, publish=not args.dry_run)))
