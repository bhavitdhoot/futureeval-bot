"""
Forecasting and research prompts.

These replace the template's prompts (template_bot.py keeps the originals
untouched, for diffing against upstream). What changed, and why:

- Every prompt states when the question closes and resolves, and how many days
  that is from today. The template asked the model for "the time left until
  the outcome is known" without ever telling it the resolution date.
- An explicit base-rate step. Reference-class reasoning is the single most
  reliable habit of good human forecasters and the template never asked for it.
- A resolution-mechanics step: tournament questions are decided by their exact
  wording, and near-misses resolve No.
- Calibration guidance that cuts both ways: the template's "extra weight on the
  status quo" alone pushes models to hedge, while log scoring punishes both
  hedging on near-certain outcomes and overconfidence.
- Numeric/date questions ask for 5th and 95th percentiles too (the template
  stopped at 10/90 and left the tails to interpolation), remind the model that
  1 in 5 outcomes lands outside its 10th-90th range (LLM intervals are
  typically too narrow), and ask for plain numbers, which lets the bot parse
  most answers locally.
- Market prices, where they exist, are named as the strongest anchor, and the
  research prompt goes looking for them (production research was already
  finding bookmaker odds, with nothing telling the model how to weigh them).
- Research prompts carry today's date, so "latest news" means something to a
  search model, and ask for base rates and scheduled events, not just headlines.
"""

from __future__ import annotations

import datetime as dt

from forecasting_tools import clean_indents
from forecasting_tools.data_models.questions import MetaculusQuestion

NO_RESEARCH = (
    "No research is available for this question. Rely on your own knowledge, "
    "remembering that it ends at your training cutoff, which is before today."
)


def _utc(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=dt.timezone.utc) if value.tzinfo is None else value


def _when(value: dt.datetime | None, now: dt.datetime) -> str | None:
    value = _utc(value)
    if value is None:
        return None
    days = (value - now).total_seconds() / 86400
    rel = f"in {days:.1f} days" if days >= 0 else f"{-days:.1f} days ago"
    return f"{value:%Y-%m-%d %H:%M} UTC ({rel})"


def timeline(question: MetaculusQuestion, now: dt.datetime | None = None) -> str:
    now = _utc(now) or dt.datetime.now(dt.timezone.utc)
    lines = [f"Today: {now:%Y-%m-%d %H:%M} UTC"]
    for label, value in (
        ("Question opened", question.open_time),
        ("Forecasting closes", question.close_time),
        ("Scheduled resolution", question.scheduled_resolution_time),
    ):
        text = _when(value, now)
        if text:
            lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _text(value: str | None) -> str:
    return value.strip() if value and value.strip() else "(none given)"


def _research(research: str) -> str:
    return research.strip() if research and research.strip() else NO_RESEARCH


def _question_block(question: MetaculusQuestion, research: str, extra: str = "") -> str:
    return clean_indents(
        f"""
        QUESTION
        {question.question_text}

        {extra}BACKGROUND
        {_text(question.background_info)}

        RESOLUTION CRITERIA (not yet satisfied when this question opened)
        {_text(question.resolution_criteria)}

        FINE PRINT
        {_text(question.fine_print)}

        TIMELINE
        {timeline(question)}

        RESEARCH FROM YOUR ASSISTANT (may be incomplete, stale or wrong; check the dates)
        {_research(research)}
        """
    )


_PERSONA = (
    "You are an expert superforecaster competing in a forecasting tournament "
    "scored by log score. Only accuracy and calibration matter."
)

_CALIBRATION = clean_indents(
    """
    Calibration reminders:
    - The world usually changes slowly, and question writers tend to ask about changes that sound plausible but mostly don't happen in short windows. Weigh the status quo heavily when little time is left.
    - But don't hedge out of uncertainty you don't have. If the outcome is effectively decided, say so with an extreme probability.
    - Strong, specific, recent evidence beats a base rate; vague commentary does not.
    - If a liquid prediction market (Polymarket, Kalshi) or bookmaker prices the same event, that price is the strongest single anchor available: start from it, after removing the bookmaker's margin, and move away only for a specific reason. Thin markets (e.g. Manifold) and pundit predictions are weaker evidence."""
)


def binary_prompt(question: MetaculusQuestion, research: str, conditional_note: str = "") -> str:
    return clean_indents(
        f"""
        {_PERSONA}

        {_question_block(question, research)}

        Work through these steps in writing:
        1. Resolution mechanics: exactly what must happen, by when, and according to which source, for this to resolve Yes. Note any technicality that could decide it.
        2. Time left: how long until resolution, and what is scheduled to happen before then.
        3. Status quo: what happens if nothing changes? Is the outcome already effectively known?
        4. Base rate: name one or more reference classes and estimate how often such events happen in a window this long.
        5. Evidence: the specific facts that move you away from the base rate, in which direction, and by how much.
        6. Scenarios: the most plausible path to Yes and the most plausible path to No.
        7. Sanity check: would either outcome surprise you? Are you anchored on the question's framing?

        {_CALIBRATION}
        {conditional_note}

        The last line of your answer must be exactly: "Probability: ZZ%" (a number from 0 to 100).
        """
    )


def multiple_choice_prompt(question, research: str, conditional_note: str = "") -> str:
    options = "\n".join(f"- {o}" for o in question.options)
    answer_lines = "\n".join(f"{o}: XX%" for o in question.options)
    return clean_indents(
        f"""
        {_PERSONA}

        {_question_block(question, research, extra=f"OPTIONS{chr(10)}{options}{chr(10)}{chr(10)}")}

        Work through these steps in writing:
        1. Resolution mechanics: exactly how the winning option will be determined, and when.
        2. Status quo: which option wins if nothing changes, and how settled that is.
        3. Base rates: how often the current leader or favourite holds on over a window this long, in comparable situations.
        4. Evidence: what moves each plausible option up or down.
        5. Surprises: a realistic scenario in which an unexpected option wins.

        {_CALIBRATION}
        - Leave a small probability on every option that is not ruled out; a 0% on the option that wins is catastrophic under log scoring.
        {conditional_note}

        End your answer with one line per option, using these exact option names, in this order, with probabilities summing to 100%:
        {answer_lines}
        """
    )


_PERCENTILE_GUIDE = clean_indents(
    """
    Distribution reminders:
    - Percentiles 10 and 90 are not extremes: one outcome in five should land outside them, and one in ten outside percentiles 5 and 95. Forecasters' intervals are usually too narrow, so widen yours for unknown unknowns, especially when the resolution is far away.
    - The distribution need not be symmetric. Put the long tail on the side where surprises are more likely.
    - Values must increase from percentile 5 to percentile 95.
    """
)


def numeric_prompt(
    question, research: str, bounds: tuple[str, str], conditional_note: str = ""
) -> str:
    upper, lower = bounds
    units = question.unit_of_measure or "not stated (infer them from the question)"
    return clean_indents(
        f"""
        {_PERSONA}

        {_question_block(question, research, extra=f"UNITS{chr(10)}{units}{chr(10)}{chr(10)}")}
        {lower}
        {upper}

        Work through these steps in writing:
        1. Resolution mechanics: exactly which number will be used, from which source, and when.
        2. The latest known value and its date.
        3. The outcome if nothing changed, and the outcome if the current trend continued.
        4. Base rates: how much this quantity has typically moved over windows this long in the past.
        5. The expectations of experts, official projections and markets, if any.
        6. A plausible scenario for a surprisingly low outcome, and one for a surprisingly high outcome.

        {_PERCENTILE_GUIDE}
        {conditional_note}

        Give values in the question's units as plain numbers: no thousands separators, no scientific notation, no words like "million".
        End your answer with exactly these eight lines:
        Percentile 5: XX
        Percentile 10: XX
        Percentile 20: XX
        Percentile 40: XX
        Percentile 60: XX
        Percentile 80: XX
        Percentile 90: XX
        Percentile 95: XX
        """
    )


def date_prompt(
    question, research: str, bounds: tuple[str, str], conditional_note: str = ""
) -> str:
    upper, lower = bounds
    return clean_indents(
        f"""
        {_PERSONA}

        {_question_block(question, research)}
        {lower}
        {upper}

        Work through these steps in writing:
        1. Resolution mechanics: exactly what event fixes the date, and according to which source.
        2. What is currently scheduled or announced, and how reliable such schedules have been historically.
        3. The date if the current trend or schedule holds.
        4. Base rates for delays or early arrivals in comparable cases.
        5. A plausible scenario for a surprisingly early date, and one for a surprisingly late date.

        {_PERCENTILE_GUIDE}
        {conditional_note}

        Write dates as YYYY-MM-DD (append THH:MM:SSZ in UTC only if the hour matters), earliest first.
        End your answer with exactly these eight lines:
        Percentile 5: YYYY-MM-DD
        Percentile 10: YYYY-MM-DD
        Percentile 20: YYYY-MM-DD
        Percentile 40: YYYY-MM-DD
        Percentile 60: YYYY-MM-DD
        Percentile 80: YYYY-MM-DD
        Percentile 90: YYYY-MM-DD
        Percentile 95: YYYY-MM-DD
        """
    )


def web_research_prompt(question: MetaculusQuestion) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    return clean_indents(
        f"""
        Today is {now:%Y-%m-%d}. You are the research assistant to a superforecaster. Search the web and report what is most useful for forecasting the question below. Do not make a forecast yourself.

        Cover, with dates and sources:
        - the latest developments directly relevant to the question;
        - the current status quo, and how the question would resolve if it resolved today;
        - scheduled events before the resolution date that could decide it;
        - base rates: how often comparable events have happened, and relevant historical or trend data;
        - prices on prediction markets (Polymarket, Kalshi, Manifold) and bookmakers for this event or closely related ones, with the exact contract, price and date; convert betting odds to implied probabilities;
        - forecasts from experts or official projections, if any.

        Question:
        {question.question_text}

        Resolution criteria:
        {_text(question.resolution_criteria)}

        Fine print:
        {_text(question.fine_print)}

        Timeline:
        {timeline(question, now)}
        """
    )


def asknews_query(question: MetaculusQuestion) -> str:
    """
    AskNews does semantic search over news articles. A long instruction prompt
    (what the template sent) dilutes the match; the question itself is the
    best query.
    """
    return question.question_text.strip()
