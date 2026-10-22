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
from forecasting_tools.data_models.questions import MetaculusQuestion  # noqa: E402

import maintenance  # noqa: E402
import prompts  # noqa: E402
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
    ("openrouter/openai/gpt-5.6-luna", "high"),
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

# Never cap output below this. Reasoning tokens count toward max_tokens, so a
# low cap at high effort can spend the whole budget thinking and return no
# answer: the call is paid for and the forecast is lost.
MAX_TOKENS = 32000


def _env(name: str) -> str:
    # GitHub passes unset repo variables as "", so a plain getenv default is not enough.
    return os.getenv(name, "").strip()


def research_models() -> list[str]:
    return [m.strip() for m in _env("RESEARCH_MODELS").split(",") if m.strip()] or DEFAULT_RESEARCH


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
    return GeneralLlm(model=model, **params)


def _identity(llm: GeneralLlm) -> tuple[str, str]:
    return llm.model, repr(llm.litellm_kwargs.get("reasoning"))


class RosterLlm(GeneralLlm):
    """
    Looks like one GeneralLlm to the framework, but:
      - rotates the first-choice model across repeated identical prompts, so
        prediction #1 for a question goes to model A, #2 to model B, etc.;
      - on any error, an empty answer, or an answer that fails `validate`,
        falls through the remaining roster, then the backups.
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
    ) -> str:
        key = hashlib.sha256(repr((prompt, system_prompt)).encode()).hexdigest()
        k = self._calls[key]
        self._calls[key] += 1  # incremented before any await: safe under asyncio
        errors = []
        for i, llm in enumerate(self.chain(k)):
            try:
                if system_prompt is None:
                    answer = await llm.invoke(prompt)
                else:
                    answer = await llm.invoke(prompt, system_prompt=system_prompt)
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("empty answer")
                if validate is not None and not validate(answer):
                    raise ValueError(f"answer has no final forecast (ends: {answer[-200:]!r})")
                if i > 0:
                    logger.warning(
                        f"Fallback used: {llm.model} answered after {i} failure(s): {errors}"
                    )
                self.usage_log.append(llm.model)
                return answer
            except Exception as e:  # noqa: BLE001 - any failure means try the next model
                errors.append(f"{llm.model}: {type(e).__name__}: {str(e)[:300]}")
        raise RuntimeError("Every model in the chain failed: " + " | ".join(errors))


def build_roster(n_predictions: int, effort_cap: str | None = None) -> RosterLlm:
    primary_specs = _parse_model_list("FORECAST_MODELS", DEFAULT_PRIMARY)
    backup_specs = _parse_model_list("BACKUP_MODELS", DEFAULT_BACKUPS)
    roster = [make_llm(m, cap_effort(e, effort_cap)) for m, e in primary_specs]
    backups = [make_llm(m, cap_effort(e, effort_cap)) for m, e in backup_specs]
    n = max(n_predictions, 1)
    # "full" mode can ask for more predictions than roster members; repeat in order.
    primaries = [roster[i % len(roster)] for i in range(n)]
    # Roster members left out of rotation are still the best fallbacks we have.
    return RosterLlm(primaries, roster[n:] + backups)


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
EXPECTED_PERCENTILES = (10, 20, 40, 60, 80, 90)


def parse_binary(text: str) -> float | None:
    """The last "Probability: ZZ%" in the text, as a decimal, or None."""
    found = _PROBABILITY.findall(text)
    if not found:
        return None
    value = float(found[-1]) / 100
    return value if 0 <= value <= 1 else None


def parse_multiple_choice(text: str, options: list[str]) -> PredictedOptionList | None:
    """
    One "Option: XX%" line per option, using the exact option names, or None.
    `\\W*:` after the name stops option "1" matching a line for option "10".
    """
    probs = {}
    for option in options:
        pattern = re.compile(
            rf"^[\W_]*{re.escape(option)}[\W_]*?\s*[:=]\s*\**\s*({_NUM})\s*%", re.I | re.M
        )
        found = pattern.findall(text)
        if not found:
            return None
        probs[option] = float(found[-1])
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


def parse_numeric(text: str) -> list[Percentile] | None:
    found = _last_by_percentile(_PERCENTILE_NUMBER.findall(text))
    if not all(p in found for p in EXPECTED_PERCENTILES):
        return None
    return [Percentile(percentile=p / 100, value=float(found[p])) for p in EXPECTED_PERCENTILES]


def parse_date(text: str) -> list[Percentile] | None:
    found = _last_by_percentile(_PERCENTILE_DATE.findall(text))
    if not all(p in found for p in EXPECTED_PERCENTILES):
        return None
    out = []
    for p in EXPECTED_PERCENTILES:
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


def pool_binary(predictions: list[float]) -> float:
    """
    Geometric mean of odds, optionally extremized (BINARY_EXTREMIZE, default 1
    = none). Unlike a median of two, it lets a confident forecaster pull the
    pool toward the tail instead of splitting the difference in probability
    space, which is what log scoring rewards when the confidence is earned.
    """
    factor = float(_env("BINARY_EXTREMIZE") or 1.0)
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
        return lambda text: bool(re.search(rf"Probability\s*\**\s*[:=]?\s*\**\s*{_NUM}", text[-1500:], re.I))
    if isinstance(question, (NumericQuestion, DateQuestion)):
        return lambda text: bool(re.search(r"Percentile\s*90", text[-1500:], re.I))
    return lambda text: bool(re.search(rf"{_NUM}\s*%", text[-3000:]))


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------
class FutureEvalBot(SummerTemplateBot2026):
    _max_concurrent_questions = 2
    _concurrency_limiter = asyncio.Semaphore(2)  # replaced per event loop in run()

    # Set by make_bot. Kept off __init__ because the framework builds each
    # forecast's published config from the __init__ signature.
    use_asknews = False
    # With AskNews configured, also run web search (news alone misses base
    # rates, official data and schedules). Without AskNews, web search is the
    # only source and always runs.
    web_research = True

    # ------------------------------ research ------------------------------
    async def run_research(self, question: MetaculusQuestion) -> str:
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
        prompt = prompts.web_research_prompt(question)
        for model in research_models():
            try:
                llm = GeneralLlm(model=model, temperature=None, timeout=180, allowed_tries=1)
                text = await llm.invoke(prompt)
                if text and text.strip():
                    return text
                logger.warning(f"Web research {model} returned nothing")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Web research {model} failed: {type(e).__name__}: {e}")
        return ""

    # ------------------------------ forecasting ------------------------------
    async def _reason(self, question: MetaculusQuestion, prompt: str) -> str:
        reasoning = await self.get_llm("default", "llm").invoke(prompt, validate=_validator(question))
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
        prediction = NumericDistribution.from_question(repair_percentiles(percentiles), question)
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
        prediction = NumericDistribution.from_question(repair_percentiles(percentiles), question)
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
            "default": build_roster(n_predictions, effort_cap),
            "summarizer": parser,  # unused (summaries disabled) but must be set
            "researcher": researcher,
            "parser": parser,
        },
    )
    bot.use_asknews = use_asknews
    bot.web_research = web_research
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
    """Roughly when the current season stops opening questions."""
    override = _env("SEASON_END_DATE")
    if override:
        try:
            return dt.date.fromisoformat(override)
        except ValueError:
            logger.warning(f"Ignoring unparsable SEASON_END_DATE={override!r}")
    season = maintenance.season_of(today)
    if season == "fall":
        return dt.date(today.year + 1, 1, 6)
    if season == "spring":
        return dt.date(today.year, 5, 6)
    return dt.date(today.year, 9, 1)


def affordable_per_question(remaining: float, today: dt.date) -> float:
    days_left = max((season_end(today) - today).days, 1)
    per_day = float(_env("EXPECTED_QUESTIONS_PER_DAY") or 8)
    return remaining / max(days_left * per_day, 1)


@dataclass(frozen=True)
class Plan:
    main: int  # forecasts per seasonal-tournament question
    mini: int  # forecasts per MiniBench question
    effort_cap: str | None  # None = each model's configured reasoning effort
    web_research: bool  # run web search alongside AskNews
    why: str


def plan_for_budget(remaining: float | None, today: dt.date | None = None) -> Plan:
    """
    Decide how much effort each question gets, from the credits left.

    Metaculus releases credits incrementally: an initial grant, then more if
    MiniBench performance is above average (plus a bonus for open-source bots).
    So when credits are tight, MiniBench keeps the two-model ensemble, because
    that is the tournament that unlocks the rest of the funding, while
    tournament questions drop to one forecast so coverage never hits zero.

    Savings come from fewer forecasts and lower reasoning effort, never from a
    lower max_tokens: a capped reasoning model can run out of tokens before
    writing its answer, which wastes the whole call.
    """
    mode = _env("ROSTER_MODE").lower() or "auto"
    if mode == "full":
        return Plan(3, 2, None, True, "ROSTER_MODE=full")
    if mode == "lean":
        return Plan(2, 1, None, False, "ROSTER_MODE=lean")
    if remaining is None:
        return Plan(2, 1, None, False, "credit balance unknown, assuming lean")
    # Spend per question, not per week. Running out in December would mean zero
    # coverage while questions are still resolving, which costs far more than a
    # smaller ensemble does. A question forecast with one model still scores;
    # a question never forecast scores nothing.
    if remaining < 6:
        return Plan(0, 0, None, False, f"${remaining:.2f} left, paused to avoid half-finished runs")
    budget = affordable_per_question(remaining, today or dt.date.today())
    note = f"${remaining:.0f} left, ${budget:.3f} affordable per question"
    if budget >= 0.40:
        return Plan(3, 2, None, True, note + ", full strength")
    if budget >= 0.22:
        return Plan(2, 2, None, True, note + ", two models everywhere")
    if budget >= 0.10:
        return Plan(1, 2, "medium", False, note + ", MiniBench keeps the ensemble")
    return Plan(1, 1, "medium", False, note + ", minimum viable coverage")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
# Per-process memory across watch cycles: how often each question was tried,
# and which ones already succeeded. Without it a question that fails for a
# persistent reason (or any question in a dry run, where nothing is marked as
# forecast on Metaculus) would be re-forecast every cycle, paying each time.
_attempts: Counter[str] = Counter()
_done: set[str] = set()


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

    remaining = maintenance.fetch_credits_remaining()
    plan = plan_for_budget(remaining, dt.date.today())
    logger.info(
        f"Budget plan: {plan.main} forecast(s) per tournament question, {plan.mini} per MiniBench "
        f"question, effort cap {plan.effort_cap or 'none'}, web research "
        f"{'on' if plan.web_research else 'fallback only'} ({plan.why})"
    )
    if plan.main == 0 and plan.mini == 0:
        logger.error("Out of credits: skipping this run. Request more through the Metaculus form.")
        return 0

    main_bot = make_bot(plan.main, publish, plan.effort_cap, plan.web_research)
    mini_bot = make_bot(plan.mini, publish, plan.effort_cap, plan.web_research)

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
            questions = select_questions(bot, fetch_open_questions(bot, tournament), seen)
        except Exception as e:  # noqa: BLE001 - e.g. a guessed slug that doesn't exist
            logger.info(f"Skipping tournament {tournament!r}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if not questions:
            continue
        for q in questions:
            _attempts[question_key(q)] += 1
        # select_questions already applied the framework's skip filter, so
        # reports line up one-to-one with questions.
        reports = await bot.forecast_questions(questions, return_exceptions=True)
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

    # Fail the workflow (which makes GitHub email you) only for systematic
    # breakage: something failed and nothing succeeded. Isolated failures are
    # retried automatically on the next cycles.
    return 1 if failures and not successes else 0


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
