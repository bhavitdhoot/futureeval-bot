# One-time setup

Budget about an hour of your time, spread over a few days because step 2 involves waiting for Metaculus. Do step 2 first for that reason.

Fall 2026 tournament questions start opening on Monday, September 28, 2026. Every question the bot misses scores zero, so the sooner it's live the better. Joining a week or two late costs a little, not the season. Don't pay for credits while you wait; the whole point is $0.

## 1. Metaculus account and bot token (10 minutes)

1. Sign up at [metaculus.com](https://www.metaculus.com).
2. Go to Settings, then **My Forecasting Bots**, then **Create a Bot**.
3. Pick the bot's username. It shows on public leaderboards, so choose something you're happy to have public. Don't put "v2" or similar in it: that marks a secondary, non-prize-eligible bot. You're allowed one prize-eligible bot.
4. Copy the token it gives you. This is your `METACULUS_TOKEN`. Keep it private.

## 2. Free LLM credits (5 minutes, then wait)

Fill in the participation and free-credits form: https://forms.gle/aQdYMq9Pisrf1v7d8

Metaculus sends you an OpenRouter API key with donated credits on it. That key is your `OPENROUTER_API_KEY`. It only works with OpenAI, Anthropic and Google models, which is what this bot uses.

In the form, mention you're entering the Fall 2026 seasonal tournament and MiniBench, and that the bot is open source (you can link your GitHub repo once step 4 is done).

## 3. AskNews research (optional, 10 minutes)

AskNews gives bot accounts a free news allocation. The bot works without it (it falls back to a search model paid from your credits), but AskNews is what Metaculus's own best bots use.

Create an account at [asknews.app](https://asknews.app), then request the free bot allocation using the instructions on the Metaculus [bot resources page](https://www.metaculus.com/notebooks/38928/futureeval-resources-page/). Their process asks for your bot name, the email you registered with AskNews, your name, and a social profile link. You'll end up with either a client ID and secret, or a single API key.

## 4. GitHub repository (15 minutes)

1. Create a new **public** repository, for example `futureeval-bot`, with no README (this folder has one). It must be public. Public repos get unlimited free Actions minutes, and a run every 20 minutes would blow through the private-repo allowance. Your keys stay secret either way: GitHub encrypts secrets and never shows them, even in public repos.
2. From this unzipped folder, push the code:

   ```bash
   cd futureeval-bot
   git init
   git add .
   git commit -m "FutureEval bot"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/futureeval-bot.git
   git push -u origin main
   ```

3. In the repo, open **Settings**, then **Actions**, then **General**. Under **Workflow permissions**, choose **Read and write permissions** and save. The daily job needs this to commit its tournament list and heartbeat.

## 5. Add your keys as secrets (5 minutes)

In the repo: **Settings**, then **Secrets and variables**, then **Actions**, then **New repository secret**. Names must match exactly.

| Secret | Value |
|---|---|
| `METACULUS_TOKEN` | from step 1 |
| `OPENROUTER_API_KEY` | from step 2 |
| `ASKNEWS_CLIENT_ID` and `ASKNEWS_SECRET` | from step 3, if you got a client ID and secret |
| `ASKNEWS_API_KEY` | from step 3, if you got a single key instead |

## 6. Turn it on and test it (15 minutes)

1. Open the **Actions** tab and click the button to enable workflows.
2. Run **Test bot** (Actions, then Test bot, then Run workflow). It runs the offline tests, then forecasts on Metaculus's practice tournament. Afterwards, open your bot's Metaculus profile and check the forecasts landed.
3. Run **Daily maintenance** once by hand. When it finishes, the repo should contain a `TARGETS.json` that lists `33121` (the Fall 2026 tournament) and shows `"current_season_found": true`. The run should end green.
4. Check GitHub will email you when something breaks: your GitHub **Settings**, then **Notifications**, then **Actions**. Make sure email notifications for failed workflows are on.

That's it. The forecast workflow is already scheduled and runs every 20 minutes on its own.

## What "done" looks like

Once live, you shouldn't need to look at it. GitHub emails you only if:

- the daily health check fails (credits running low, or the next season's tournament can't be found), or
- a forecasting run fails completely (no question succeeded).

Isolated failures on single questions are retried automatically and don't email you. [SEASON_CHECKLIST.md](SEASON_CHECKLIST.md) explains what each email means and what to do.

## Optional: run the tests locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests/
```

To run the bot locally, copy `.env.template` to `.env`, fill in your keys, and run `python main.py --mode test_questions`. Never commit `.env`.
