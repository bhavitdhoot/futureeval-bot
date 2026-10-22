# FutureEval forecasting bot

An autonomous forecasting bot for the [Metaculus FutureEval bot tournaments](https://www.metaculus.com/futureeval/): the $50k seasonal tournament (three seasons a year) and the bi-weekly $1k MiniBench. It runs entirely on GitHub Actions, with no server and no human in the loop, as the tournament rules require.

New here? Setup is in [SETUP.md](SETUP.md). The few things that recur each season are in [SEASON_CHECKLIST.md](SEASON_CHECKLIST.md).

## How the bot works

This section is the methodology description Metaculus asks prize winners to provide. The code in this repository is the full implementation.

The bot is built on the official [Metaculus template bot](https://github.com/Metaculus/metac-bot-template) and its [forecasting-tools](https://github.com/Metaculus/forecasting-tools) framework. For every open question it does one research pass, then asks several frontier models to forecast independently, then aggregates.

**Research.** The bot sends the question to AskNews news search, and in parallel asks a web-search model for recent developments, the status quo, scheduled events, base rates and any market or expert forecasts. News alone misses base rates and official data, so the two are combined. When credits are tight, web search runs only if AskNews finds nothing. If both fail, the bot forecasts without research rather than skipping the question.

**Prompts** (`prompts.py`). These are rewritten from the template's. Each prompt states when the question closes and resolves and how many days away that is (the template asked models to estimate "time left" without giving them the date). The model works through resolution mechanics, time left, the status quo, a reference-class base rate, the evidence, and scenarios on both sides. Calibration guidance cuts both ways: weight the status quo, but don't hedge on outcomes that are effectively decided. Numeric and date prompts remind the model that one outcome in five should land outside its 10th–90th percentile range.

**Forecasting** uses a two-vendor roster: `openai/gpt-5.6-sol` and `anthropic/claude-opus-4.8`, both at high reasoning effort, routed through OpenRouter. How many forecasts each question gets depends on the credit balance (see "Rough credit use" below). Model choice and reasoning effort were the features most associated with rank in Metaculus's own Spring 2026 analysis: the template prompt with a frontier high-reasoning model placed 18th of 173, while the same prompt at standard effort scored less than half as well.

**Parsing.** Final answers are parsed directly from the requested format ("Probability: 23%", "Percentile 10: 4.5"). An LLM parser is only used when an answer strays from that format. Percentiles listed in the wrong order are sorted rather than discarded.

**Aggregation.** Binary forecasts are pooled with the geometric mean of odds, which has outperformed the median and the mean of probabilities on Metaculus questions. Multiple choice uses the mean probability per option. Numeric and date questions use the median of the forecasters' cumulative distributions.

Reliability is treated as part of accuracy, because a skipped question scores zero:

- Every model call has a fallback chain. If a model errors, or returns an empty or truncated answer with no final forecast, the call goes to the other vendor's model, then to backup models (`gpt-5.5`, `claude-opus-4.7`, `gpt-5.6-luna`).
- The parser has its own fallback chain.
- A question publishes as long as at least one model succeeds.
- A forecast run keeps polling for new questions every 5 minutes. A failed question is retried on later polls, up to 3 attempts per run, so a question that fails for a persistent reason can't drain credits.
- A daily job discovers the live tournament IDs, so new seasons are picked up without code changes. A question reachable through two tournament IDs is only forecast once.

There is no human involvement in individual forecasts. The bot is not re-run on open questions after seeing its output, and no forecast is edited by hand.

## Files

| File | What it does |
|---|---|
| `main.py` | Entry point. Model roster, fallbacks, research, parsing, aggregation, budget plan, which tournaments to forecast on. |
| `prompts.py` | Forecasting and research prompts. |
| `template_bot.py`, `bot_helpers.py` | Metaculus's template, unchanged (kept for comparison with upstream; `main.py` overrides its prompts). |
| `maintenance.py` | Daily: finds live tournaments (writes `TARGETS.json`), checks credit balance and model availability, raises an alert if you need to act. |
| `.github/workflows/forecast.yaml` | Runs the bot, polling for new questions. |
| `.github/workflows/daily.yaml` | Runs maintenance and keeps the schedule alive. |
| `.github/workflows/test.yaml` | Manual: offline tests, then a live forecast on the practice tournament. |
| `tests/` | Offline tests (mocked models, real framework pipeline). |

## Settings

Everything below is optional. Set these as repository Variables (Settings, then Secrets and variables, then Actions, then the Variables tab). Leave them unset to use the defaults.

| Variable | Default | Use it when |
|---|---|---|
| `ROSTER_MODE` | `auto` (sized to the credit balance) | Set to `full` if you have plenty of credits (three forecasts per main question, two per MiniBench question), or `lean` for a fixed two and one. |
| `SKIP_MINIBENCH` | unset | Set to `true` to save credits for the main tournament. |
| `FORECAST_MODELS` | `openrouter/openai/gpt-5.6-sol:high,openrouter/anthropic/claude-opus-4.8:high` | New flagship models come out. Format is `slug:effort`, comma-separated. |
| `BACKUP_MODELS` | `gpt-5.5`, `claude-opus-4.7`, `gpt-5.6-luna` (all high) | Rarely. |
| `PARSER_MODELS` | `gpt-4.1-mini`, `gpt-5.6-luna`, `gpt-4o-mini` | Rarely. |
| `RESEARCH_MODELS` | `gpt-5.6-terra:online`, `gpt-5.6-sol:online` | A web-search model is retired. |
| `EXTRA_TOURNAMENTS` | unset | Metaculus renames a tournament and the daily check emails you. Paste its ID here. |
| `CREDIT_FLOOR_USD` | `25` | You want the low-credit warning earlier or later. |
| `EXPECTED_QUESTIONS_PER_DAY` | `8` | The budget plan over- or under-spends because the real question rate differs. |
| `SEASON_END_DATE` | season's usual end | A season ends on an unusual date (`YYYY-MM-DD`). |
| `BINARY_EXTREMIZE` | `1` (off) | Only with evidence from resolved questions that the pooled forecasts are underconfident; `1.2` pushes them 20% further from 50% in log-odds. |
| `MAX_ATTEMPTS_PER_QUESTION` | `3` | Rarely. |
| `WATCH_MINUTES`, `WATCH_INTERVAL_SECONDS` | `300`, `300` | Rarely. |

The donated Metaculus OpenRouter key only works with OpenAI, Anthropic and Google models, so keep the roster within those three.

## Rough credit use

By default the bot sizes its effort to the credits left and the days left in the season, so the money lasts the whole season. Lots of credits buys three forecasts per main question and two per MiniBench question, plus web research alongside AskNews. As credits tighten, it drops to fewer forecasts, then to medium reasoning effort, and keeps MiniBench at two forecasts as long as it can, because MiniBench results decide whether Metaculus releases more credits. Below $6 it pauses rather than leave runs half finished. Each run logs its plan and the actual spend per question.

In `lean` mode, expect roughly $0.50 per seasonal-tournament question and $0.25 per MiniBench question. That works out to very roughly $300 to $400 per four-month season. These figures are estimates based on another bot's measured per-model costs, so watch the balance in your first weeks. The daily health check warns you by email when the balance drops below the floor.

## Credits

Built on [metac-bot-template](https://github.com/Metaculus/metac-bot-template) and [forecasting-tools](https://github.com/Metaculus/forecasting-tools) by Metaculus. Model choices and the donated-key provider limits were informed by the public operations notes of [nostreambot](https://github.com/No-Stream/metaculus-bot) (MIT). No code was copied from it.
