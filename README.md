# FutureEval forecasting bot

An autonomous forecasting bot for the [Metaculus FutureEval bot tournaments](https://www.metaculus.com/futureeval/): the $50k seasonal tournament (three seasons a year) and the bi-weekly $1k MiniBench. It runs entirely on GitHub Actions, with no server and no human in the loop, as the tournament rules require.

New here? Setup is in [SETUP.md](SETUP.md). The few things that recur each season are in [SEASON_CHECKLIST.md](SEASON_CHECKLIST.md).

## How the bot works

This section is the methodology description Metaculus asks prize winners to provide. The code in this repository is the full implementation.

The bot is built on the official [Metaculus template bot](https://github.com/Metaculus/metac-bot-template). Its forecasting prompts are used unchanged (`template_bot.py`). For every open question it does one research pass, then asks several frontier models to forecast independently, then aggregates.

Research comes from AskNews news summaries. If AskNews is unavailable, it falls back to a search-enabled OpenAI model, and if that fails too it forecasts without research rather than skipping the question.

Forecasting uses a two-vendor roster: `openai/gpt-5.6-sol` and `anthropic/claude-opus-4.8`, both at high reasoning effort, routed through OpenRouter. Each seasonal-tournament question gets one forecast from each model. MiniBench questions get one forecast from the OpenAI model, which keeps credit use down on lower-value questions. The forecasts are combined with the forecasting-tools framework's standard aggregation (median for binary questions, normalized median for multiple choice, mixture of distributions for numeric).

The design choice behind this comes from Metaculus's own Spring 2026 analysis. Model choice and reasoning effort were the features most associated with rank. The template prompt with a frontier high-reasoning model placed 18th of 173, while the same prompt at standard effort scored less than half as well.

Reliability is treated as part of accuracy, because a skipped question scores zero:

- Every model call has a fallback chain. If a model errors, the call goes to the other vendor's model, then to backup models (`gpt-5.5`, `claude-opus-4.7`, `gpt-5.2`).
- The parser that converts reasoning into numbers has its own fallback chain.
- A question publishes as long as at least one model succeeds.
- Failed questions are retried automatically on the next run, 20 minutes later.
- A daily job discovers the live tournament IDs, so new seasons are picked up without code changes.

There is no human involvement in individual forecasts. The bot is not re-run on open questions after seeing its output, and no forecast is edited by hand.

## Files

| File | What it does |
|---|---|
| `main.py` | Entry point. Model roster, fallbacks, resilient research, which tournaments to forecast on. |
| `template_bot.py`, `bot_helpers.py` | Metaculus's template, unchanged (prompts and question-type handling). |
| `maintenance.py` | Daily: finds live tournaments (writes `TARGETS.json`), checks credit balance, raises an alert if you need to act. |
| `.github/workflows/forecast.yaml` | Runs the bot every 20 minutes. |
| `.github/workflows/daily.yaml` | Runs maintenance once a day and keeps the schedule alive. |
| `.github/workflows/test.yaml` | Manual: offline tests, then a live forecast on the practice tournament. |
| `tests/` | Offline tests (mocked models, real framework pipeline). |

## Settings

Everything below is optional. Set these as repository Variables (Settings, then Secrets and variables, then Actions, then the Variables tab). Leave them unset to use the defaults.

| Variable | Default | Use it when |
|---|---|---|
| `ROSTER_MODE` | `lean` | Set to `full` if you have plenty of credits (three forecasts per main question, two per MiniBench question). |
| `SKIP_MINIBENCH` | unset | Set to `true` to save credits for the main tournament. |
| `FORECAST_MODELS` | `openrouter/openai/gpt-5.6-sol:high,openrouter/anthropic/claude-opus-4.8:high` | New flagship models come out. Format is `slug:effort`, comma-separated. |
| `BACKUP_MODELS` | `gpt-5.5`, `claude-opus-4.7`, `gpt-5.2` (all high) | Rarely. |
| `PARSER_MODELS` | `gpt-4.1-mini`, `gpt-5.6-luna`, `gpt-4o-mini` | Rarely. |
| `EXTRA_TOURNAMENTS` | unset | Metaculus renames a tournament and the daily check emails you. Paste its ID here. |
| `CREDIT_FLOOR_USD` | `25` | You want the low-credit warning earlier or later. |

The donated Metaculus OpenRouter key only works with OpenAI, Anthropic and Google models, so keep the roster within those three.

## Rough credit use

In `lean` mode, expect roughly $0.50 per seasonal-tournament question and $0.25 per MiniBench question. That works out to very roughly $300 to $400 per four-month season. These figures are estimates based on another bot's measured per-model costs, so watch the balance in your first weeks. The daily health check warns you by email when the balance drops below the floor.

## Credits

Built on [metac-bot-template](https://github.com/Metaculus/metac-bot-template) and [forecasting-tools](https://github.com/Metaculus/forecasting-tools) by Metaculus. Model choices and the donated-key provider limits were informed by the public operations notes of [nostreambot](https://github.com/No-Stream/metaculus-bot) (MIT). No code was copied from it.
