# Seasonal checklist

Seasons start in January, May and September and run about four months. Around each season boundary, set aside about 30 minutes for the items below. Everything else is automatic.

## Every season (about 30 minutes)

**1. Fill in the bot survey. This one is not optional.** Metaculus sends a survey to every participant after a season's questions resolve. No survey means no prize, for that season and the next. For "how does your bot work", paste or link the "How the bot works" section of README.md.

**2. Renew your LLM credits.** Use the same form as setup (https://forms.gle/aQdYMq9Pisrf1v7d8) to request credits for the new season. If Metaculus issues a new key, replace the `OPENROUTER_API_KEY` secret.

**3. Renew AskNews.** Metaculus says bot accounts must renew their AskNews allocation each season, using the same process as setup. Skip this if you didn't set up AskNews.

**4. Optional: refresh the models.** When OpenAI or Anthropic release a new flagship, set the repository variable `FORECAST_MODELS` to the new slugs (format in README.md), before the new season's first questions. Check the exact slugs on openrouter.ai/models. Old models keep working for a while and backups cover retirements, so this is an accuracy upgrade, not a repair. If you'd rather not, ask Claude to look up the current slugs for you.

## If you win a prize

Metaculus verifies prize winners' identity, nationality and residency, and may ask for a short demo or explanation of the bot (README.md covers the explanation). Taxes and fees are the recipient's responsibility.

Before you send payment details, ask Metaculus two questions: how they can pay a UAE resident, and whether any tax will be withheld from the prize. Their bot-tournament contact is ben [at] metaculus [dot] com. Payouts arrive a couple of months after a season ends, once questions resolve. For example, Spring 2026 (January to April) paid out in June and July.

## What each alert email means

GitHub emails you when a workflow fails. The error message is at the bottom of the run's log.

**"OpenRouter credits are low"** (Daily maintenance): request more credits through the form. To stretch what's left, set the variable `SKIP_MINIBENCH` to `true`, and make sure `ROSTER_MODE` isn't `full`.

**"No [season] [year] bot tournament found"** (Daily maintenance): Metaculus probably renamed the new season's tournament. Find it on metaculus.com/aib; the ID is the number in its URL or on the page. Add it as the repository variable `EXTRA_TOURNAMENTS`. This only fires from the second month of a season, because tournaments open partway through their first month.

**"Forecast on tournament questions" failed** (every question in a run failed). Look for these in the log:

- `401` or an authentication error mentioning Metaculus: your `METACULUS_TOKEN` is wrong or was revoked.
- `402`, `403` or "credit" mentioning OpenRouter: the credits ran out. See above.
- "Every model in the chain failed" with `404`: model slugs were retired. Update `FORECAST_MODELS`.
- Anything else: copy the log into a chat with Claude.

If failure emails pile up, pause the bot first (Actions, then Forecast on tournament questions, then the "..." menu, then Disable workflow), fix the cause, then re-enable it.

## The one rule to keep in mind

The rules forbid a human in the loop. Don't run the bot on an open tournament question, look at its answer, and then change something so it answers that same question differently. Improving the bot between questions is allowed. Nudging specific forecasts is not. If you leave it alone, you're fine.
