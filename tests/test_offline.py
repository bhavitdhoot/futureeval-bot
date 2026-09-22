"""
Offline tests: no network. Model calls are mocked at GeneralLlm.invoke, so the
real forecasting-tools pipeline (research -> predictions -> parsing ->
aggregation) runs end to end on synthetic questions.

Run:  python -m pytest -q tests/
"""

import asyncio
import datetime as dt
import json
import os
import re
import sys

import pytest

os.environ.setdefault("METACULUS_TOKEN", "dummy")
os.environ.setdefault("OPENROUTER_API_KEY", "dummy")
for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY", "FORECAST_MODELS", "ROSTER_MODE"):
    os.environ.pop(k, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting_tools import GeneralLlm  # noqa: E402
from forecasting_tools.data_models.questions import (  # noqa: E402
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

import main as botmain  # noqa: E402

GPT, CLAUDE = botmain.DEFAULT_PRIMARY[0][0], botmain.DEFAULT_PRIMARY[1][0]
PROB = {GPT: 30, CLAUDE: 50}  # each model's "opinion" on the binary question
BROKEN: set[str] = set()      # models that should raise
CALLS: list[str] = []


def _parse_reply(prompt: str) -> str:
    """Pretend to be the parser model: turn reasoning text into the JSON asked for."""
    if "Probability:" in prompt and "prediction_in_decimal" in prompt:
        p = int(re.findall(r"Probability:\s*(\d+)%", prompt)[-1])
        return json.dumps({"prediction_in_decimal": p / 100})
    if "Percentile 10" in prompt or "percentile" in prompt.lower() and "value" in prompt:
        pts = re.findall(r"Percentile (\d+):\s*([\d.]+)", prompt)
        return json.dumps([{"percentile": int(a) / 100, "value": float(b)} for a, b in pts[-6:]])
    opts = re.findall(r'"?(Red|Green|Blue)"?:\s*(\d+)%', prompt)
    if opts:
        return json.dumps({"predicted_options": [
            {"option_name": o, "probability": int(p) / 100} for o, p in opts[-3:]]})
    return "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>"


async def fake_invoke(self, prompt, system_prompt=None):
    CALLS.append(self.model)
    if self.model in BROKEN:
        raise RuntimeError(f"simulated outage for {self.model}")
    text = prompt if isinstance(prompt, str) else str(prompt)
    if "convert text into structured data" in text or "python parsable type" in text:
        return _parse_reply(text)
    if "search-preview" in self.model:
        return "Recent news: nothing decisive has happened. Status quo favours No."
    p = PROB.get(self.model, 40)
    if "Percentile 10:" in text:  # numeric prompt
        base = 100 if self.model == GPT else 120
        return "Reasoning...\n" + "\n".join(
            f"Percentile {q}: {base + i * 10}" for i, q in enumerate([10, 20, 40, 60, 80, 90]))
    if "Option_A" in text or "Red" in text and "Blue" in text:
        return "Reasoning...\nRed: 20%\nGreen: 30%\nBlue: 50%"
    return f"Reasoning about the question...\nProbability: {p}%"


@pytest.fixture(autouse=True)
def patch_llm(monkeypatch):
    CALLS.clear()
    BROKEN.clear()
    orig = GeneralLlm.invoke

    async def dispatcher(self, prompt, system_prompt=None):
        if isinstance(self, botmain.RosterLlm):
            return await orig(self, prompt, system_prompt) if False else await botmain.RosterLlm.invoke(self, prompt, system_prompt)
        return await fake_invoke(self, prompt, system_prompt)

    monkeypatch.setattr(GeneralLlm, "invoke", dispatcher)
    yield


def binary_q():
    return BinaryQuestion(
        question_text="Will X happen before 2027?", id_of_post=1, id_of_question=1,
        page_url="https://www.metaculus.com/questions/1/", resolution_criteria="Resolves Yes if X.",
        fine_print="", background_info="Some background.", already_forecasted=False,
    )


def run(coro):
    return asyncio.run(coro)


def test_ensemble_uses_both_vendors_and_aggregates():
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(binary_q()))
    forecasters = [c for c in CALLS if c in (GPT, CLAUDE)]
    assert set(forecasters) == {GPT, CLAUDE}, forecasters
    # framework aggregates binary predictions (median of 0.30 and 0.50 = 0.40)
    assert abs(report.prediction - 0.40) < 1e-6, report.prediction


def test_fallback_when_one_vendor_is_down():
    BROKEN.add(GPT)
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert abs(report.prediction - 0.50) < 1e-6  # Claude answered both slots
    assert GPT in CALLS  # it was tried first, then skipped


def test_backups_when_both_primaries_down():
    BROKEN.update({GPT, CLAUDE})
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert 0 < report.prediction < 1
    assert botmain.DEFAULT_BACKUPS[0][0] in CALLS


def test_parser_fallback():
    BROKEN.add(botmain.DEFAULT_PARSERS[0])
    bot = botmain.make_bot(1, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert abs(report.prediction - 0.30) < 1e-6
    assert botmain.DEFAULT_PARSERS[1] in CALLS


def test_everything_down_raises_instead_of_publishing_garbage():
    BROKEN.update({GPT, CLAUDE} | {m for m, _ in botmain.DEFAULT_BACKUPS})
    bot = botmain.make_bot(2, publish=False)
    with pytest.raises(BaseException):
        run(bot.forecast_question(binary_q()))


def test_research_fallback_then_none():
    BROKEN.add(botmain.RESEARCH_FALLBACKS[0])
    bot = botmain.make_bot(1, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert 0 < report.prediction < 1  # still forecasts without research


def test_multiple_choice():
    q = MultipleChoiceQuestion(
        question_text="Which colour?", options=["Red", "Green", "Blue"], id_of_post=2,
        id_of_question=2, page_url="https://www.metaculus.com/questions/2/",
        resolution_criteria="", fine_print="", background_info="", already_forecasted=False,
    )
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(q))
    probs = {o.option_name: o.probability for o in report.prediction.predicted_options}
    assert abs(sum(probs.values()) - 1) < 1e-6 and probs["Blue"] > probs["Red"], probs


def test_numeric():
    q = NumericQuestion(
        question_text="How many?", id_of_post=3, id_of_question=3,
        page_url="https://www.metaculus.com/questions/3/", resolution_criteria="", fine_print="",
        background_info="", unit_of_measure="units", upper_bound=300, lower_bound=0,
        open_upper_bound=True, open_lower_bound=False, already_forecasted=False,
    )
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(q))
    cdf = report.prediction.cdf
    vals = [p.percentile for p in cdf]
    assert len(cdf) > 100 and all(b >= a for a, b in zip(vals, vals[1:]))


def test_env_override_of_models(monkeypatch):
    monkeypatch.setenv("FORECAST_MODELS", "openrouter/openai/some-new-model:xhigh,openrouter/anthropic/other")
    roster = botmain.build_roster(2)
    assert [m.model for m in roster.primaries] == ["openrouter/openai/some-new-model", "openrouter/anthropic/other"]
    assert roster.primaries[0].litellm_kwargs["reasoning"] == {"effort": "xhigh"}
    assert "reasoning" not in roster.primaries[1].litellm_kwargs


def test_full_mode_repeats_roster():
    roster = botmain.build_roster(3)
    assert [m.model for m in roster.primaries] == [GPT, CLAUDE, GPT]


# ---------------- tournament discovery & maintenance ----------------
import maintenance  # noqa: E402

EXISTING = {33121, "fall-futureeval-2026", "minibench", "spring-futureeval-2027"}


def fake_probe(t):
    if t in EXISTING or str(t) in {str(x) for x in EXISTING}:
        return 3
    raise RuntimeError("HTTP 400")


def test_season_slugs_roll_over():
    assert maintenance.season_slugs(dt.date(2026, 9, 22))[:2] == ["fall-futureeval-2026", "fall-aib-2026"]
    assert "spring-futureeval-2027" in maintenance.season_slugs(dt.date(2026, 12, 20))
    assert maintenance.season_slugs(dt.date(2027, 1, 10))[0] == "spring-futureeval-2027"
    assert maintenance.season_slugs(dt.date(2027, 6, 1))[0] == "summer-futureeval-2027"


def test_discover_keeps_only_real_tournaments():
    t = maintenance.discover(dt.date(2026, 9, 22), fake_probe)
    assert 33121 in t["main"] and "fall-futureeval-2026" in t["main"]
    assert "fall-aib-2026" not in t["main"] and t["current_season_found"] and t["minibench_ok"]


def test_check_alerts_when_next_season_missing():
    # Feb 2027 with no spring tournament discoverable -> alert
    t = maintenance.discover(dt.date(2027, 2, 10), lambda x: 0 if x in (33121, "minibench") else fake_probe("nope"))
    assert not t["current_season_found"]
    problems = maintenance.check(t, dt.date(2027, 2, 10), credits_remaining=500)
    assert any("No spring 2027" in p for p in problems)
    # but not during the season's first month (tournaments open partway through it)
    assert maintenance.check(t, dt.date(2027, 1, 5), credits_remaining=500) == []


def test_check_alerts_on_low_credits_only():
    t = maintenance.discover(dt.date(2026, 10, 1), fake_probe)
    assert maintenance.check(t, dt.date(2026, 10, 1), credits_remaining=300) == []
    assert maintenance.check(t, dt.date(2026, 10, 1), credits_remaining=None) == []
    assert any("credits are low" in p for p in maintenance.check(t, dt.date(2026, 10, 1), credits_remaining=10))


def test_empty_env_vars_behave_like_unset(monkeypatch):
    # GitHub Actions passes unset repository variables as empty strings.
    for var in ("CREDIT_FLOOR_USD", "EXTRA_TOURNAMENTS", "ROSTER_MODE", "SKIP_MINIBENCH", "FORECAST_MODELS"):
        monkeypatch.setenv(var, "")
    t = maintenance.discover(dt.date(2026, 10, 1), fake_probe)
    assert maintenance.check(t, dt.date(2026, 10, 1), credits_remaining=300) == []
    assert any("credits are low" in p for p in maintenance.check(t, dt.date(2026, 10, 1), credits_remaining=10))
    assert [m.model for m in botmain.build_roster(2).primaries] == [GPT, CLAUDE]


def test_main_uses_fresh_targets_file(tmp_path, monkeypatch):
    f = tmp_path / "TARGETS.json"
    monkeypatch.setattr(maintenance, "TARGETS_FILE", f)
    f.write_text(json.dumps({"updated": "2027-01-20", "main": ["spring-futureeval-2027"]}))
    ts = botmain.main_tournaments(dt.date(2027, 1, 21))
    assert "spring-futureeval-2027" in ts and 33121 in ts
    # stale file -> only known IDs plus ONE guessed slug (keeps runs fast)
    ts_stale = botmain.main_tournaments(dt.date(2027, 3, 1))
    assert ts_stale.count("spring-futureeval-2027") == 1 and "spring-aib-2027" not in ts_stale
    # during Fall 2026 no guessing at all
    f.unlink()
    assert all(not isinstance(t, str) for t in botmain.main_tournaments(dt.date(2026, 10, 1)))


def test_run_exit_code_only_on_systematic_failure(monkeypatch):
    async def all_fail(self, tournament, return_exceptions=True):
        return [RuntimeError("boom")]
    async def mixed(self, tournament, return_exceptions=True):
        class R:  # minimal stand-in for a ForecastReport
            question = binary_q()
        return [RuntimeError("boom"), R()]
    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_on_tournament", all_fail)
    assert run(botmain.run("tournament", publish=False)) == 1
    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_on_tournament", mixed)
    assert run(botmain.run("tournament", publish=False)) == 0
