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

import ledger  # noqa: E402
import main as botmain  # noqa: E402
import maintenance  # noqa: E402

# With OpenRouter's catalog unreadable (as in these offline tests), each slot
# resolves to its last, known-good candidate.
GPT, CLAUDE, GEMINI = (spec.split("|")[-1] for spec, _ in botmain.DEFAULT_PRIMARY)
PROB = {GPT: 30, CLAUDE: 50}  # each model's "opinion" on the binary question
BROKEN: set[str] = set()      # models that should raise
TRUNCATED: set[str] = set()   # models that stop before their final answer
CALLS: list[str] = []
STYLE = {"numeric_commas": False}


def _parse_reply(prompt: str) -> str:
    """Pretend to be the parser model: turn reasoning text into the JSON asked for."""
    if "Probability:" in prompt and "prediction_in_decimal" in prompt:
        p = int(re.findall(r"Probability:\s*(\d+)%", prompt)[-1])
        return json.dumps({"prediction_in_decimal": p / 100})
    if "Percentile 10" in prompt or "percentile" in prompt.lower() and "value" in prompt:
        pts = re.findall(r"Percentile (\d+):\s*([\d.,]+)", prompt)
        return json.dumps([{"percentile": int(a) / 100, "value": float(b.replace(",", ""))} for a, b in pts[-6:]])
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
    if ":online" in self.model:
        return "Recent news: nothing decisive has happened. Status quo favours No."
    if self.model in TRUNCATED:
        return "Let me think step by step about the base rate and"
    p = PROB.get(self.model, 40)
    if "Percentile 10:" in text:  # numeric prompt
        base = 100 if self.model == GPT else 120
        fmt = "{:,}" if STYLE["numeric_commas"] else "{}"
        return "Reasoning...\n" + "\n".join(
            f"Percentile {q}: {fmt.format((base + i * 10) * (1000 if STYLE['numeric_commas'] else 1))}"
            for i, q in enumerate([10, 20, 40, 60, 80, 90]))
    if "Option_A" in text or "Red" in text and "Blue" in text:
        return "Reasoning...\nRed: 20%\nGreen: 30%\nBlue: 50%"
    return f"Reasoning about the question...\nProbability: {p}%"


@pytest.fixture(autouse=True)
def offline_catalog(monkeypatch):
    monkeypatch.setattr(maintenance, "fetch_available_models", lambda: None)
    monkeypatch.setattr(botmain, "_catalog_read", False)
    monkeypatch.setattr(botmain, "_catalog_cache", None)


@pytest.fixture(autouse=True)
def isolated_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("LEDGER_PATH", str(tmp_path / "ledger.json"))
    monkeypatch.delenv("EXPECTED_QUESTIONS_PER_DAY", raising=False)


@pytest.fixture(autouse=True)
def patch_llm(monkeypatch):
    CALLS.clear()
    BROKEN.clear()
    TRUNCATED.clear()
    STYLE["numeric_commas"] = False
    botmain._attempts.clear()
    botmain._done.clear()
    botmain._balance_mark = None
    for var in ("BINARY_EXTREMIZE", "PARSER_MODELS", "RESEARCH_MODELS", "BACKUP_MODELS"):
        monkeypatch.delenv(var, raising=False)

    async def dispatcher(self, prompt, system_prompt=None, **kwargs):
        if isinstance(self, botmain.RosterLlm):
            return await botmain.RosterLlm.invoke(self, prompt, system_prompt, **kwargs)
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
    # geometric mean of odds of 0.30 and 0.50, slightly below their arithmetic mean
    assert abs(report.prediction - botmain.pool_binary([0.3, 0.5])) < 1e-9, report.prediction
    assert 0.39 < report.prediction < 0.40


def test_fallback_when_one_vendor_is_down():
    BROKEN.add(GPT)
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert abs(report.prediction - 0.50) < 1e-6  # Claude answered both slots
    assert GPT in CALLS  # it was tried first, then skipped


def test_third_vendor_then_backups_when_primaries_down():
    BROKEN.update({GPT, CLAUDE})
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert 0 < report.prediction < 1
    assert set(bot.get_llm("default", "llm").usage_log) == {GEMINI}
    BROKEN.add(GEMINI)
    CALLS.clear()
    report = run(botmain.make_bot(2, publish=False).forecast_question(binary_q()))
    assert "openrouter/openai/gpt-5.5" in CALLS and 0 < report.prediction < 1


def test_binary_parsed_locally_without_parser_calls():
    bot = botmain.make_bot(1, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert abs(report.prediction - 0.30) < 1e-6
    assert not set(CALLS) & set(botmain.DEFAULT_PARSERS)


def test_parser_fallback():
    # "1,000,000" is not a plain number, so the LLM parser is needed, and its
    # first model is down.
    STYLE["numeric_commas"] = True
    BROKEN.add(botmain.DEFAULT_PARSERS[0])
    bot = botmain.make_bot(1, publish=False)
    report = run(bot.forecast_question(numeric_q(upper=300_000)))
    assert botmain.DEFAULT_PARSERS[1] in CALLS
    # the parsed thousands landed in the right place: P(X <= 100,000) is about 10%
    nearest = min(report.prediction.cdf, key=lambda c: abs(c.value - 100_000))
    assert abs(nearest.percentile - 0.10) < 0.02, nearest


def test_everything_down_raises_instead_of_publishing_garbage():
    BROKEN.update({GPT, CLAUDE, GEMINI} | {m for m, _ in botmain.DEFAULT_BACKUPS})
    bot = botmain.make_bot(2, publish=False)
    with pytest.raises(BaseException):
        run(bot.forecast_question(binary_q()))


def test_research_fallback_then_none():
    BROKEN.update(botmain.research_models())
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


def numeric_q(upper=300):
    return NumericQuestion(
        question_text="How many?", id_of_post=3, id_of_question=3,
        page_url="https://www.metaculus.com/questions/3/", resolution_criteria="", fine_print="",
        background_info="", unit_of_measure="units", upper_bound=upper, lower_bound=0,
        open_upper_bound=True, open_lower_bound=False, already_forecasted=False,
    )


def test_numeric():
    bot = botmain.make_bot(2, publish=False)
    report = run(bot.forecast_question(numeric_q()))
    cdf = report.prediction.cdf
    vals = [p.percentile for p in cdf]
    assert len(cdf) > 100 and all(b >= a for a, b in zip(vals, vals[1:]))


def test_env_override_of_models(monkeypatch):
    monkeypatch.setenv("FORECAST_MODELS", "openrouter/openai/some-new-model:xhigh,openrouter/anthropic/other")
    roster = botmain.build_roster(2)
    assert [m.model for m in roster.primaries] == ["openrouter/openai/some-new-model", "openrouter/anthropic/other"]
    assert roster.primaries[0].litellm_kwargs["reasoning"] == {"effort": "xhigh"}
    assert "reasoning" not in roster.primaries[1].litellm_kwargs


def test_full_mode_uses_three_vendors():
    roster = botmain.build_roster(3)
    assert [m.model for m in roster.primaries] == [GPT, CLAUDE, GEMINI]
    assert [m.model for m in botmain.build_roster(4).primaries][3] == GPT  # repeats past three


# ---------------- tournament discovery & maintenance ----------------

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


# ---------------- budget planning ----------------
OCT = dt.date(2026, 10, 11)   # 81 days of season left
DEC = dt.date(2026, 12, 15)   # 16 days left


def test_budget_scales_with_time_left(monkeypatch):
    monkeypatch.delenv("ROSTER_MODE", raising=False)
    monkeypatch.delenv("EXPECTED_QUESTIONS_PER_DAY", raising=False)
    monkeypatch.delenv("SEASON_END_DATE", raising=False)
    # Same balance, different runway: late in the season it can spend freely.
    # With October 2026 production costs (~$0.10 per high-effort forecast) and
    # 8 questions a day, $78 in mid-October buys one high-effort forecast, two
    # on MiniBench...
    tight = botmain.plan_for_budget(78, OCT)
    assert (tight.main, tight.mini, tight.effort_cap) == (1, 2, None)
    # ...and $40 only a medium-effort one.
    lean = botmain.plan_for_budget(40, OCT)
    assert (lean.main, lean.mini, lean.effort_cap) == (1, 1, "medium")
    assert botmain.plan_for_budget(78, DEC).main >= 2
    # A big top-up unlocks full strength even with a long season left.
    full = botmain.plan_for_budget(800, OCT)
    assert (full.main, full.mini, full.effort_cap, full.web_research) == (3, 2, None, True)
    # Nearly empty always pauses rather than dying mid-question.
    empty = botmain.plan_for_budget(2, DEC)
    assert (empty.main, empty.mini) == (0, 0)
    unknown = botmain.plan_for_budget(None, OCT)
    assert (unknown.main, unknown.mini, unknown.effort_cap) == (2, 1, None)
    # While credits are tight, MiniBench never gets less effort than the main
    # tournament (at full strength the main tournament deliberately gets more).
    for bal in (78, 20):
        plan = botmain.plan_for_budget(bal, OCT)
        assert plan.mini >= plan.main


def test_season_end_rolls_over_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("SEASON_END_DATE", raising=False)
    assert botmain.season_end(dt.date(2026, 10, 11)) == dt.date(2026, 12, 31)
    assert botmain.season_end(dt.date(2027, 2, 1)) == dt.date(2027, 4, 30)
    assert botmain.season_end(dt.date(2027, 6, 1)) == dt.date(2027, 8, 31)
    monkeypatch.setenv("SEASON_END_DATE", "2027-03-01")
    assert botmain.season_end(dt.date(2027, 2, 1)) == dt.date(2027, 3, 1)
    monkeypatch.setenv("SEASON_END_DATE", "not-a-date")
    assert botmain.season_end(dt.date(2026, 10, 11)) == dt.date(2026, 12, 31)


def test_end_of_season_does_not_blow_the_budget(monkeypatch):
    # One day before the assumed end, the budget is still spread over a week
    # of questions, not the last day's eight...
    assert botmain.affordable_per_question(56, dt.date(2026, 12, 30), 8) == 1.0
    # ...and the same once an overridden end date has passed but questions keep coming.
    monkeypatch.setenv("SEASON_END_DATE", "2026-12-20")
    assert botmain.affordable_per_question(56, dt.date(2026, 12, 28), 8) == 1.0


def test_retired_models_are_reported(monkeypatch):
    monkeypatch.delenv("FORECAST_MODELS", raising=False)
    configured = maintenance.configured_model_ids()
    assert "openai/gpt-6.1-sol|openai/gpt-6-sol|openai/gpt-5.6-sol" in configured
    assert all("openrouter/" not in m and ":online" not in m for m in configured)
    available = {c for slot in configured for c in slot.split("|")} - {"openai/gpt-5.6-sol"}
    assert maintenance.missing_models(available, configured) == ["openai/gpt-5.6-sol"]  # the backup slot
    # a multi-candidate slot only fails when none of its candidates is listed
    available -= {"openai/gpt-6.1-sol", "openai/gpt-6-sol"}
    assert "openai/gpt-6.1-sol|openai/gpt-6-sol|openai/gpt-5.6-sol" in maintenance.missing_models(available, configured)
    assert maintenance.missing_models(None, configured) == []      # unknown, never alarms
    t = maintenance.discover(dt.date(2026, 10, 11), fake_probe)
    problems = maintenance.check(t, dt.date(2026, 10, 11), 300, ["openai/gpt-5.6-sol"])
    assert any("no longer listed" in p for p in problems)
    assert maintenance.check(t, dt.date(2026, 10, 11), 300, []) == []


def test_roster_mode_overrides_budget(monkeypatch):
    monkeypatch.setenv("ROSTER_MODE", "full")
    plan = botmain.plan_for_budget(5, OCT)
    assert (plan.main, plan.mini, plan.effort_cap) == (3, 2, None)
    monkeypatch.setenv("ROSTER_MODE", "lean")
    plan = botmain.plan_for_budget(5, OCT)
    assert (plan.main, plan.mini, plan.effort_cap) == (2, 1, None)


def test_tight_budget_lowers_effort_not_max_tokens():
    roster = botmain.build_roster(2, effort_cap="medium")
    for m in roster.primaries + roster.backups:
        assert m.litellm_kwargs["max_tokens"] == botmain.MAX_TOKENS
        assert m.litellm_kwargs["reasoning"] == {"effort": "medium"}
    # a cap never raises a lower configured effort
    assert botmain.cap_effort("low", "medium") == "low"
    assert botmain.cap_effort(None, "medium") is None


def test_minibench_runs_before_main_tournament(monkeypatch):
    monkeypatch.delenv("ROSTER_MODE", raising=False)
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: 100.0)
    order = []

    def record(bot, tournament):
        order.append(tournament)
        return []

    monkeypatch.setattr(botmain, "fetch_open_questions", record)
    assert run(botmain.run("tournament", publish=False)) == 0
    assert order[0] == botmain.MINIBENCH and len(order) > 1


def test_no_forecasting_when_credits_are_gone(monkeypatch):
    monkeypatch.delenv("ROSTER_MODE", raising=False)
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: 1.0)
    called = []

    def record(bot, tournament):
        called.append(tournament)
        return []

    monkeypatch.setattr(botmain, "fetch_open_questions", record)
    assert run(botmain.run("tournament", publish=False)) == 0
    assert called == []


# ---------------- watch loop ----------------
def test_watch_polls_until_window_ends(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(botmain.time, "monotonic", lambda: clock["t"])

    async def fake_sleep(sec):
        clock["t"] += sec

    monkeypatch.setattr(botmain.asyncio, "sleep", fake_sleep)
    cycles = {"n": 0}

    async def fake_run(mode, publish):
        cycles["n"] += 1
        return 0

    monkeypatch.setattr(botmain, "run", fake_run)
    # 30 minute window, polling every 5 minutes -> first pass plus 6 more
    assert run(botmain.watch("tournament", False, 30, 300)) == 0
    assert cycles["n"] == 7


def test_watch_returns_last_exit_code(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(botmain.time, "monotonic", lambda: clock["t"])

    async def fake_sleep(sec):
        clock["t"] += sec

    monkeypatch.setattr(botmain.asyncio, "sleep", fake_sleep)
    codes = iter([0, 0, 1])

    async def fake_run(mode, publish):
        return next(codes)

    monkeypatch.setattr(botmain, "run", fake_run)
    assert run(botmain.watch("tournament", False, 10, 300)) == 1


def binary_q_n(n):
    q = binary_q()
    q.id_of_question = q.id_of_post = n
    q.page_url = f"https://www.metaculus.com/questions/{n}/"
    return q


def test_run_exit_code_only_on_systematic_failure(monkeypatch):
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: 500.0)
    monkeypatch.setattr(botmain, "fetch_open_questions", lambda bot, t: [binary_q_n(1), binary_q_n(2)])

    async def all_fail(self, questions, return_exceptions=True):
        return [RuntimeError("boom") for _ in questions]

    async def mixed(self, questions, return_exceptions=True):
        return [RuntimeError("boom"), object()]

    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_questions", all_fail)
    assert run(botmain.run("tournament", publish=False)) == 1
    botmain._attempts.clear()
    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_questions", mixed)
    assert run(botmain.run("tournament", publish=False)) == 0


# ---------------- robustness and accuracy upgrades ----------------
def test_truncated_answer_moves_down_the_chain():
    # GPT spends its whole budget thinking and never writes "Probability:".
    TRUNCATED.add(GPT)
    bot = botmain.make_bot(1, publish=False)
    report = run(bot.forecast_question(binary_q()))
    assert abs(report.prediction - 0.50) < 1e-6  # Claude's answer, not a parse of garbage
    assert CALLS.count(GPT) == 1


def test_single_prediction_keeps_other_flagship_as_first_fallback():
    roster = botmain.build_roster(1)
    assert [m.model for m in roster.chain(0)][:2] == [GPT, CLAUDE]
    BROKEN.add(GPT)
    bot = botmain.make_bot(1, publish=False)
    run(bot.forecast_question(binary_q()))
    assert bot.get_llm("default", "llm").usage_log == [CLAUDE]


def test_fallback_chain_has_no_repeats():
    roster = botmain.build_roster(4)  # [GPT, CLAUDE, GEMINI, GPT]
    for k in range(3):
        models = [m.model for m in roster.chain(k)]
        assert len(models) == len(set(models)), models


def test_model_list_keeps_openrouter_variants(monkeypatch):
    monkeypatch.setenv("FORECAST_MODELS", "openrouter/openai/x:online,openrouter/anthropic/y:HIGH,z")
    assert botmain._parse_model_list("FORECAST_MODELS", []) == [
        ("openrouter/openai/x:online", None), ("openrouter/anthropic/y", "high"), ("z", None)]


def test_local_parsers():
    assert botmain.parse_binary("Probability: 20%\n...\n**Probability: 7.5%**") == 0.075
    assert botmain.parse_binary("I think it's likely.") is None
    assert botmain.parse_binary("The prior probability: 10% seems low.") is None
    mc = botmain.parse_multiple_choice(
        "- 1: 10%\n- 10: 20%\n\nFinal:\n**1**: 30%\n10: 50%\n100: 20%", ["1", "10", "100"])
    assert {o.option_name: round(o.probability, 3) for o in mc.predicted_options} == {
        "1": 0.3, "10": 0.5, "100": 0.2}
    assert botmain.parse_multiple_choice("Red: 50%", ["Red", "Blue"]) is None  # missing option
    text = "\n".join(f"Percentile {p}: {v}" for p, v in zip([10, 20, 40, 60, 80, 90], [1, 2, 3, 4, 5, 6]))
    assert [x.value for x in botmain.parse_numeric(text)] == [1, 2, 3, 4, 5, 6]
    assert botmain.parse_numeric(text.replace(": 6", ": 6 million")) is None  # needs the LLM parser
    dates = "\n".join(f"Percentile {p}: 2027-0{i + 1}-15" for i, p in enumerate([10, 20, 40, 60, 80, 90]))
    parsed = botmain.parse_date(dates)
    assert parsed[0].value == dt.datetime(2027, 1, 15, tzinfo=dt.timezone.utc).timestamp()


def test_reversed_percentiles_are_repaired_not_discarded():
    from forecasting_tools import Percentile
    reversed_ = [Percentile(percentile=p / 100, value=v) for p, v in zip([10, 20, 40, 60, 80, 90], [60, 50, 40, 30, 20, 10])]
    assert [p.value for p in botmain.repair_percentiles(reversed_)] == [10, 20, 30, 40, 50, 60]


def test_binary_pooling(monkeypatch):
    assert abs(botmain.pool_binary([0.3, 0.3]) - 0.3) < 1e-9
    assert botmain.pool_binary([0.02, 0.3]) < 0.12 < (0.02 + 0.3) / 2  # confident forecaster counts
    monkeypatch.setenv("BINARY_EXTREMIZE", "1.5")
    assert botmain.pool_binary([0.8, 0.8]) > 0.8


def test_prompts_carry_the_resolution_date():
    q = binary_q()
    q.scheduled_resolution_time = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=10)
    import prompts
    text = prompts.binary_prompt(q, "")
    assert "Scheduled resolution" in text and ("in 10.0 days" in text or "in 9.9 days" in text)
    assert prompts.NO_RESEARCH in text and "Base rate" in text


def test_asknews_gets_the_question_and_empty_results_fall_back(monkeypatch):
    monkeypatch.setenv("ASKNEWS_API_KEY", "dummy")
    queries = []

    async def no_articles(self, query):
        queries.append(query)
        return "Here are the relevant news articles:\n\nNo articles were found.\n\n"

    monkeypatch.setattr(botmain.AskNewsSearcher, "get_formatted_news_async", no_articles)
    bot = botmain.make_bot(1, publish=False, web_research=False)
    research = run(bot.run_research(binary_q()))
    assert queries == ["Will X happen before 2027?"]
    assert "## Web research" in research and "AskNews" not in research


def test_asknews_and_web_research_combined(monkeypatch):
    monkeypatch.setenv("ASKNEWS_API_KEY", "dummy")

    async def articles(self, query):
        return "Here are the relevant news articles:\n\n**X is likely**"

    monkeypatch.setattr(botmain.AskNewsSearcher, "get_formatted_news_async", articles)
    research = run(botmain.make_bot(1, publish=False, web_research=True).run_research(binary_q()))
    assert "Recent news (AskNews)" in research and "Web research" in research
    CALLS.clear()
    research = run(botmain.make_bot(1, publish=False, web_research=False).run_research(binary_q()))
    assert "Web research" not in research and not any(":online" in c for c in CALLS)


def test_same_question_in_two_tournaments_is_forecast_once(monkeypatch):
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: 500.0)
    monkeypatch.setattr(botmain, "fetch_open_questions", lambda bot, t: [binary_q_n(7)])
    forecast = []

    async def record(self, questions, return_exceptions=True):
        forecast.extend(questions)
        return [object() for _ in questions]

    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_questions", record)
    run(botmain.run("tournament", publish=False))
    run(botmain.run("tournament", publish=False))  # next cycle: already done in this job
    assert len(forecast) == 1


def test_persistently_failing_question_is_capped(monkeypatch):
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: 500.0)
    monkeypatch.setattr(botmain, "fetch_open_questions", lambda bot, t: [binary_q_n(8)])
    tries = []

    async def fail(self, questions, return_exceptions=True):
        tries.extend(questions)
        return [RuntimeError("boom") for _ in questions]

    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_questions", fail)
    for _ in range(6):
        run(botmain.run("tournament", publish=False))
    assert len(tries) == botmain.max_attempts() == 3


def test_watch_survives_a_crashing_cycle(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(botmain.time, "monotonic", lambda: clock["t"])

    async def fake_sleep(sec):
        clock["t"] += sec

    monkeypatch.setattr(botmain.asyncio, "sleep", fake_sleep)
    outcomes = iter([RuntimeError("Metaculus API down"), 0, 0])

    async def flaky_run(mode, publish):
        o = next(outcomes)
        if isinstance(o, Exception):
            raise o
        return o

    monkeypatch.setattr(botmain, "run", flaky_run)
    assert run(botmain.watch("tournament", False, 10, 300)) == 0


def test_model_check_honours_parser_override(monkeypatch):
    monkeypatch.setenv("PARSER_MODELS", "openrouter/openai/custom-parser")
    configured = maintenance.configured_model_ids()
    assert "openai/custom-parser" in configured and "openai/gpt-4o-mini" not in configured


# ---------------- spending ledger and measured budget plan ----------------
NOW = dt.datetime(2026, 10, 29, 12, tzinfo=dt.timezone.utc)


def fresh_book(days_old=10):
    return ledger.Ledger(created=(NOW - dt.timedelta(days=days_old)).isoformat())


def test_ledger_measures_rate_share_and_cost():
    book = fresh_book(days_old=3)
    # 3 days, 4 main questions/day plus 1 MiniBench question a day
    for d in range(3):
        for h in range(4):
            book.record_seen(f"m{d}{h}", "main", NOW - dt.timedelta(days=d, hours=h * 3))
        book.record_seen(f"b{d}", "minibench", NOW - dt.timedelta(days=d, hours=1))
    book.record_seen("m00", "main", NOW)  # seen again: not double counted
    for i in range(5):
        book.record_spend(ledger.UNCAPPED, 1, 1, 0.20, NOW - dt.timedelta(hours=i))
    stats = book.stats(NOW)
    assert stats.measured
    assert abs(stats.questions_per_day - 15 / 3) < 1e-9
    assert abs(stats.minibench_share - 3 / 15) < 1e-9
    assert abs(stats.slot_cost[ledger.UNCAPPED] - 0.20) < 1e-9
    # medium had no data, so it is scaled from uncapped at the priors' ratio
    assert abs(stats.slot_cost[ledger.MEDIUM] - 0.20 * 0.6) < 1e-9


def test_backlog_on_a_fresh_ledger_is_not_counted_as_arrivals():
    book = ledger.Ledger(created=(NOW - dt.timedelta(days=2)).isoformat())
    for i in range(30):  # already open when the ledger started
        book.record_seen(f"old{i}", "minibench", NOW - dt.timedelta(days=2) + dt.timedelta(minutes=1))
    for i in range(16):  # genuinely new
        book.record_seen(f"new{i}", "main", NOW - dt.timedelta(hours=i * 3))
    assert abs(book.stats(NOW).questions_per_day - 8) < 1e-9


def test_ledger_uses_totals_so_balance_lag_washes_out():
    book = ledger.Ledger()
    # the balance lagged: one batch looked free, the next paid for both
    for usd in (0.0, 0.24, 0.12, 0.12):
        book.record_spend(ledger.UNCAPPED, 2, 1, usd, NOW)
    assert abs(book.stats(NOW).slot_cost[ledger.UNCAPPED] - 0.06) < 1e-9


def test_ledger_priors_until_enough_data():
    stats = ledger.Ledger().stats(NOW)
    assert not stats.measured and stats.questions_per_day == ledger.PRIOR_QUESTIONS_PER_DAY
    book = ledger.Ledger()
    book.record_seen("q", "main", NOW - dt.timedelta(hours=5))  # under 2 days of history
    book.record_spend(ledger.UNCAPPED, 1, 1, 9.99, NOW)  # one slot is not a measurement
    stats = book.stats(NOW)
    assert stats.questions_per_day == ledger.PRIOR_QUESTIONS_PER_DAY
    assert stats.slot_cost == ledger.PRIOR_SLOT_COST


def test_ledger_round_trip_prune_and_corruption(tmp_path):
    path = tmp_path / "l.json"
    book = ledger.Ledger()
    book.record_seen("old", "main", NOW - dt.timedelta(days=40))
    book.record_seen("new", "main", NOW)
    book.record_spend(ledger.UNCAPPED, 1, 1, 0.1, NOW)
    book.record_spend(ledger.UNCAPPED, 1, 0, 0.1, NOW)  # nothing forecast: ignored
    book.save(path, NOW)
    again = ledger.Ledger.load(path)
    assert list(again.seen) == ["new"] and len(again.spend) == 1 and again.created == book.created
    path.write_text("{not json")
    assert ledger.Ledger.load(path).seen == {}  # corrupt file never stops forecasting
    # structurally damaged entries are dropped on load, so stats() can't crash later
    path.write_text(json.dumps({
        "seen": {"a": "x", "b": {"tournament": "main"}, "c": {"tournament": "main", "first_seen": NOW.isoformat()}},
        "spend": [{"ts": "bad"}, 5, {"ts": NOW.isoformat(), "effort": "uncapped", "slots": 1, "questions": 1, "usd": 0.1}],
        "created": "nonsense",
    }))
    damaged = ledger.Ledger.load(path)
    assert list(damaged.seen) == ["c"] and len(damaged.spend) == 1
    damaged.stats(NOW)
    path.write_text("[1, 2]")
    assert ledger.Ledger.load(path).spend == []


def test_plan_follows_measured_costs():
    cheap = ledger.Stats(8.0, 0.25, {ledger.UNCAPPED: 0.03, ledger.MEDIUM: 0.02}, True)
    pricey = ledger.Stats(8.0, 0.25, {ledger.UNCAPPED: 0.40, ledger.MEDIUM: 0.25}, True)
    assert botmain.plan_for_budget(52.6, dt.date(2026, 10, 29), cheap).main == 3
    plan = botmain.plan_for_budget(52.6, dt.date(2026, 10, 29), pricey)
    assert (plan.main, plan.mini, plan.effort_cap) == (1, 1, "medium")  # never below coverage
    # fewer questions than assumed means more per question
    quiet = ledger.Stats(1.0, 0.25, {ledger.UNCAPPED: 0.10, ledger.MEDIUM: 0.06}, True)
    assert botmain.plan_for_budget(52.6, dt.date(2026, 10, 29), quiet).main >= 2


def test_question_rate_override(monkeypatch):
    stats = ledger.Stats(1.0, 0.25, {ledger.UNCAPPED: 0.10, ledger.MEDIUM: 0.06}, True)
    monkeypatch.setenv("EXPECTED_QUESTIONS_PER_DAY", "40")
    assert botmain.plan_for_budget(52.6, dt.date(2026, 10, 29), stats).effort_cap == "medium"


def test_run_feeds_the_ledger(monkeypatch):
    balance = {"usd": 50.0}
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: balance["usd"])
    monkeypatch.setattr(
        botmain, "fetch_open_questions",
        lambda bot, t: [binary_q_n(21)] if t == botmain.MINIBENCH else [binary_q_n(22), binary_q_n(23)])

    async def forecast(self, questions, return_exceptions=True):
        balance["usd"] -= 0.1 * len(questions)
        return [object() for _ in questions]

    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_questions", forecast)
    run(botmain.run("tournament", publish=False))
    book = ledger.Ledger.load()
    assert {k: v["tournament"] for k, v in book.seen.items()} == {"21": "minibench", "22": "main", "23": "main"}
    assert [round(s["usd"], 3) for s in book.spend] == [0.1, 0.2]
    assert sum(s["questions"] for s in book.spend) == 3


def test_practice_runs_do_not_feed_the_ledger(monkeypatch):
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: 50.0)
    monkeypatch.setattr(botmain, "fetch_open_questions", lambda bot, t: [binary_q_n(31)])

    async def forecast(self, questions, return_exceptions=True):
        return [object() for _ in questions]

    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_questions", forecast)
    run(botmain.run("test_questions", publish=False))
    assert not ledger.ledger_path().exists()


def test_numeric_tails_are_used_when_given():
    text = "\n".join(f"Percentile {p}: {v}" for p, v in zip([5, 10, 20, 40, 60, 80, 90, 95], range(1, 9)))
    assert [p.percentile for p in botmain.parse_numeric(text)] == [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
    from forecasting_tools import NumericDistribution
    dist = NumericDistribution.from_question(botmain.parse_numeric(text.replace(": ", ": 1")), numeric_q())
    assert len(dist.get_cdf()) > 100


# ---------------- regressions from the code review ----------------
def test_late_charges_are_added_to_the_batch_they_belong_to(monkeypatch):
    balance = {"usd": 50.0}
    monkeypatch.setattr(maintenance, "fetch_credits_remaining", lambda: balance["usd"])
    questions = iter([[binary_q_n(41)], [], []])
    monkeypatch.setattr(botmain, "fetch_open_questions", lambda bot, t: next(questions, []))

    async def forecast(self, qs, return_exceptions=True):
        balance["usd"] -= 0.06  # posted in time
        return [object() for _ in qs]

    monkeypatch.setattr(botmain.FutureEvalBot, "forecast_questions", forecast)
    monkeypatch.setenv("SKIP_MINIBENCH", "true")
    run(botmain.run("tournament", publish=False))
    balance["usd"] -= 0.04  # posted after the batch's closing reading
    run(botmain.run("tournament", publish=False))
    assert [round(s["usd"], 3) for s in ledger.Ledger.load().spend] == [0.10]


def test_two_validation_misses_cost_two_calls_not_the_whole_chain():
    TRUNCATED.update({GPT, CLAUDE})
    roster = botmain.build_roster(2)
    validate = botmain._validator(binary_q())
    answer = run(roster.invoke("prompt", validate=validate))
    assert answer.startswith("Let me think")  # first answer, handed on to the parser
    assert CALLS == [GPT, CLAUDE]


def test_differently_worded_final_answer_is_not_rejected():
    validate = botmain._validator(binary_q())
    assert validate("...so my final probability (Yes) is 35%.")
    assert not validate("Let me think step by step about the base rate and")
    assert botmain._validator(numeric_q())("90th percentile: 120")


def test_rotation_follows_the_question_not_the_prompt_text():
    roster = botmain.build_roster(2)
    first = run(roster.invoke("Today: 11:59 ...", rotation_key="forecast:7"))
    second = run(roster.invoke("Today: 12:00 ...", rotation_key="forecast:7"))
    assert CALLS == [GPT, CLAUDE], (first, second)


def test_multiple_choice_names_match_exactly():
    text = "Final:\n<10: 20%\n10: 30%\n>10: 50%"
    mc = botmain.parse_multiple_choice(text, ["<10", "10", ">10"])
    assert {o.option_name: round(o.probability, 3) for o in mc.predicted_options} == {"<10": 0.2, "10": 0.3, ">10": 0.5}
    mc = botmain.parse_multiple_choice("1. **Red**: 20%\n2. Green: 30%\n- `Blue`: 50%", ["Red", "Green", "Blue"])
    assert mc is not None


def test_bad_extremize_setting_does_not_block_forecasts(monkeypatch):
    monkeypatch.setenv("BINARY_EXTREMIZE", "1.2x")
    assert abs(botmain.pool_binary([0.3, 0.3]) - 0.3) < 1e-9


# ---------------- runtime model resolution ----------------
def test_newest_listed_candidate_wins(monkeypatch):
    monkeypatch.setattr(maintenance, "fetch_available_models",
                        lambda: {"openai/gpt-6-sol", "openai/gpt-5.6-sol", "anthropic/claude-opus-5.5"})
    assert botmain.resolve_model(botmain.DEFAULT_PRIMARY[0][0]) == "openrouter/openai/gpt-6-sol"
    assert botmain.resolve_model(botmain.DEFAULT_PRIMARY[1][0]) == "openrouter/anthropic/claude-opus-5.5"
    # nothing listed: fall back to the known-good last candidate
    assert botmain.resolve_model(botmain.DEFAULT_PRIMARY[2][0]) == GEMINI
    assert botmain.resolve_model("openrouter/x/y") == "openrouter/x/y"


def test_catalog_is_read_once_per_process(monkeypatch):
    reads = []
    monkeypatch.setattr(maintenance, "fetch_available_models", lambda: reads.append(1) or {"openai/gpt-6-sol"})
    botmain.build_roster(3)
    botmain.build_roster(3)
    assert len(reads) == 1


def test_research_candidates_resolve_with_online_suffix(monkeypatch):
    monkeypatch.setattr(maintenance, "fetch_available_models", lambda: {"openai/new-search"})
    monkeypatch.setenv("RESEARCH_MODELS", "openrouter/openai/new-search:online|openrouter/openai/old:online")
    assert botmain.research_models() == ["openrouter/openai/new-search:online"]
