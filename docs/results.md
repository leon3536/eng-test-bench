# Harness bench results

Final, 3 Nov 2026, reruns included. Claude Code through the gateway added 6 Nov 2026; Droid added 13 Nov 2026.

2,225 trials, $358 in model fees, 12 answer lookups counted as fails (3 Nov results, 41 runs); Droid adds 243 trials, $46.23 and 1 answer lookup. The 6 Nov gateway experiment ($16.37 across its five arms) is reported in its own section and not in these totals.

- Eleven agent harnesses ran the same 27 IT, e-commerce and SaaS tasks (terminal-bench 2.0) on seven models (nine columns: Opus at three efforts), one attempt per task (two where a trial was rerun; a task's result is the average of its trials). Droid, the eleventh harness, was added on 13 Nov 2026.
- Every ratio divides by a reference on the same tasks: 0.80× means 20% less than the reference.
- The overall reference is **Claude Code on Opus 5.5 medium**.
- Harnesses are compared with **Claude Code** on Claude models and with **Goose** on the other models, at the same model and effort.
- Models are compared with the same harness on **Opus 5.5 medium**.

## Executive insights

1. **Model choice correlates more with task success; harness choice correlates more with cost.** On Claude models every harness except Droid passed 25–27 of 27 tasks, a spread too small to tell apart at this sample size; Droid passed 22–25, with most of its misses on the three server tasks (see insight 7). Cost varied widely: at Opus medium, Pi cost 31% less than Claude Code, Goose 21% less, and Cline 60% more.
2. **Sonnet 5.5 stood out among the mid-tier models on these terminal-bench tasks.** Across the 8 harnesses that ran every model in the 3 Nov results, Sonnet passed 209 of 216 (97%), against 92–94% for Kimi K3, GPT-6.1 Terra, DeepSeek and Grok, and 80% for GLM. It was also the fastest (1.6 min, the median of the 8 harnesses' median trial times, against 2.1–5.6) and tied with Terra as the cheapest (2.6¢, measured the same way). The pass gap to the next models is a few tasks; the cost and speed gaps are larger. For scale, Opus medium passed 211 of 216 at about three times Sonnet's cost.
3. **Claude Code was the fastest harness at Opus 5.5 medium, not the cheapest.** At Opus medium the other harnesses took 1.3–2.2× as long as Claude Code (agent run time, task by task); for Droid (1.40×) no difference is also possible. At Opus low and high and on Sonnet, Goose took 0.90–1.01× Claude Code's time and Pi 0.85–1.19×. Where people wait on the agent, that speed may justify its cost; for unattended batch work, Pi and Goose reached similar pass counts (within one task) for less.
4. **DeepSeek and GLM cost less than Opus 5.5 medium on every harness (for GLM on Kilo, mini-swe-agent and Droid, no difference is also possible), and each open model tended to do best on a particular harness.** Kimi K3 cost more than Opus medium on 7 of the 10 harnesses that ran both (1.30–1.75×) and less on Goose, Cline and Terminus-2 (0.53–0.97×). The top harness differed by model: Kimi K3 on Pi (27 of 27, among the cheapest and fastest), DeepSeek V4.1 Flash on OpenCode (26 of 27 at the lowest median cost, 2.2¢; the next cheapest harnesses that also passed 26 were Claude Code through the gateway at 2.3¢ and Terminus-2 at 2.6¢), GLM 5.3 on Cline (24 of 27). GLM passed the fewest tasks of any model on every harness (20 to 24; on Droid it tied Sonnet and Grok at 22). Most of these margins are one to four tasks, so treat each pairing as a starting point to confirm, not a settled ranking.
5. **Grok 4.7 was the most expensive option in this test, and the slowest on most harnesses.** On Goose it cost 1.9× Opus medium and took 3.3× as long, with 26 vs 27 passes; GLM took longer on mini-swe-agent, DeepSeek on Strands, and GLM, DeepSeek and Kimi K3 on Droid; on Terminus-2 GLM matched it (2.16× each). Some of the slowness coincided with an overloaded test machine; the cost gap held after the reruns.
6. **Some agents with internet access looked up the benchmark's answers.** In 13 trials (one of them under Droid), 9 of them on DeepSeek, the agent downloaded the benchmark's own answer key and used it; we count those as fails. Published scores for internet-connected agents may be inflated, and the same behaviour in production means fetching and running unvetted code, so internet access warrants a deliberate policy.
7. **Droid's handling of the agent's background processes correlated with failures on tasks that ask the agent to leave a service running.** Droid (added 13 Nov) failed `kv-store-grpc` in all 9 model columns, while the other harnesses passed it 94% of the time. On exit, the Droid program stopped the background processes the agent had started through it; 20 of Droid's 30 failed trials were on the three such tasks, and in 17 of those 20 its log records stopping the server the task needed. On the other 24 tasks Droid passed as many as the reference harness, or more, in 8 of 9 model columns. If an agent is expected to leave a service running, check that the harness does not stop it on exit.

## Tasks passed (of 27)

| Harness | Opus 5.5 low | Opus 5.5 medium | Opus 5.5 high | Sonnet 5.5 | Kimi K3 | GPT-6.1 Terra | DeepSeek V4.1 Flash | GLM 5.3 | Grok 4.7 |
|---|---|---|---|---|---|---|---|---|---|
| Claude Code | 25/27 | 27/27 | 26/27 | 25/27 | 25/27† | 26/27† | 26/27† | · | · |
| Pi | 26/27 ★ | 26/27 | 27/27 ★ | 26/27 ⭐ | 27/27 | 25/27 | 26/27 ⊘ ★ | 22/27 | 24/27 ⊘ |
| Goose | 27/27 ⭐ | 27/27 | 27/27 ★ | 26/27 ★ | 26/27 | 25/27 | 23/27 | 21/27 | 26/27 |
| OpenCode | 26/27 | 26/27 | 27/27 | 26/27 ★ | 26/27 | 25/27 | 26/27 ⊘2 ⭐ | 22/27 | 26/27 ⊘ |
| Kilo | 27/27 | 27/27 | 27/27 | 26/27 | 24/27 | 26/27 | 26/27 ⊘ | 20/27 | 22/27 |
| Cline | 26/27 | 27/27 | 27/27 | 26/27 | 26/27 | 25/27 | 25/27 | 24/27 | 26/27 ⊘ |
| Terminus-2 | 25/27 | 27/27 | 27/27 | 26/27 | 25/27 | 26/27 | 26/27 | 23/27 | 24/27 ⊘ |
| mini-swe-agent | 27/27 ★ | 26/27 | 26/27 | 26/27 ★ | 26/27 | 26/27 | 24/27 ⊘2 ★ | 20/27 | 27/27 |
| Strands | 27/27 ★ | 25/27 | 27/27 | 27/27 ★ | 24/27 | 26/27 | 24/27 ⊘ | 21/27 | 24/27 |
| Droid | 24/27 ★ | 24/27 | 25/27 | 22/27 | 26/27 | 23/27 | 25/27 ⊘ | 22/27 | 22/27 |
| DeepSeek Harness | · | · | · | · | · | · | 24/27 ⊘ ★ | · | · |

⭐ local optimum: beats both its column's reference harness and its own harness on Opus 5.5 medium.
★ beats one of the two. "Beats" means no fewer passes, cost and time within 5% or better, and at least 5% cheaper
or faster. ⊘ marks answer lookups (the agent downloaded the benchmark's own answers), counted as fails. · not run.
† Claude Code through the LiteLLM gateway, not a direct run (see the gateway section).

## Median cost per trial

| Harness | Opus 5.5 low | Opus 5.5 medium | Opus 5.5 high | Sonnet 5.5 | Kimi K3 | GPT-6.1 Terra | DeepSeek V4.1 Flash | GLM 5.3 | Grok 4.7 |
|---|---|---|---|---|---|---|---|---|---|
| Claude Code | 7.6¢ | 8.4¢ | $0.11 | 3.6¢ | $0.13† | 2.3¢† | 2.3¢† | · | · |
| Pi | 5.6¢ ★ | 5.8¢ | 7.8¢ ★ | 2.1¢ ⭐ | 6.8¢ | 1.7¢ | 3.1¢ ★ | 2.6¢ | $0.20 |
| Goose | 6.0¢ ⭐ | 7.4¢ | 8.8¢ ★ | 2.3¢ ★ | 7.0¢ | 1.6¢ | 3.1¢ | 2.6¢ | $0.13 |
| OpenCode | $0.10 | 9.3¢ | $0.10 | 2.6¢ ★ | $0.14 | 3.6¢ | 2.2¢ ⭐ | 7.3¢ | $0.24 |
| Kilo | $0.17 | 7.9¢ | $0.15 | 2.9¢ | $0.15 | 4.7¢ | 3.9¢ | 7.7¢ | $0.19 |
| Cline | 7.7¢ | $0.12 | $0.14 | 3.9¢ | 8.8¢ | 2.9¢ | 5.0¢ | 5.6¢ | $0.32 |
| Terminus-2 | 7.3¢ | $0.11 | $0.12 | 4.1¢ | 6.0¢ | 1.8¢ | 2.6¢ | 4.9¢ | $0.13 |
| mini-swe-agent | 5.4¢ ★ | 7.5¢ | 8.9¢ | 2.6¢ ★ | $0.14 | 2.4¢ | 2.7¢ ★ | 10.0¢ | $0.21 |
| Strands | 8.5¢ ★ | 8.6¢ | $0.12 | 2.5¢ ★ | 9.5¢ | 7.8¢ | 6.1¢ | 5.1¢ | $0.15 |
| Droid | 5.8¢ ★ | 8.9¢ | 9.1¢ | 3.1¢ | $0.11 | 3.4¢ | 2.7¢ | 7.3¢ | $0.13 |
| DeepSeek Harness | · | · | · | · | · | · | 2.9¢ ★ | · | · |

## Median minutes per trial

Whole trial, median over trials: harness install, the agent's run and grading. Insights 3 and 5 use the
"Time ÷ reference" ratios below instead (insight 2 uses these medians), which pair tasks and count only the
agent's own run time, leaving out trials during which the test machine's CPU was busy (90% or more for at least a
quarter of the agent's run). Install time differs by cell, so compare these minutes with care. On Opus 5.5 medium,
Claude Code (14 of 27 trials), Pi (27 of 27), mini-swe-agent (18 of 27) and Strands (17 of 27) include trials from
the first runs (29–30 Oct 2026), before pre-baked images, when installing the harness took a median of 5.1, 1.7, 1.7
and 1.0 minutes per trial. Elsewhere, Cline's install took about 1–2 minutes even from a pre-baked image, the
DeepSeek Harness's about 2.3 minutes and Droid's about 0.6–0.8 minutes (both installed fresh in every trial).

| Harness | Opus 5.5 low | Opus 5.5 medium | Opus 5.5 high | Sonnet 5.5 | Kimi K3 | GPT-6.1 Terra | DeepSeek V4.1 Flash | GLM 5.3 | Grok 4.7 |
|---|---|---|---|---|---|---|---|---|---|
| Claude Code | 1.0 | 3.6 | 1.4 | 1.1 | 1.8† | 1.3† | 2.0† | · | · |
| Pi | 1.7 ★ | 4.1 | 1.9 ★ | 1.6 ⭐ | 2.4 | 2.1 | 2.7 ★ | 4.5 | 8.1 |
| Goose | 1.0 ⭐ | 2.2 | 1.9 ★ | 1.1 ★ | 2.3 | 1.5 | 2.5 | 2.7 | 4.6 |
| OpenCode | 1.8 | 3.2 | 1.8 | 1.7 ★ | 3.5 | 2.8 | 3.1 ⭐ | 4.6 | 5.7 |
| Kilo | 2.0 | 1.3 | 2.0 | 1.6 | 2.7 | 3.7 | 3.0 | 3.4 | 5.5 |
| Cline | 2.4 | 3.8 | 2.7 | 2.5 | 3.1 | 3.1 | 4.9 | 3.0 | 6.3 |
| Terminus-2 | 1.9 | 2.1 | 3.0 | 1.9 | 3.3 | 1.8 | 2.6 | 4.7 | 5.0 |
| mini-swe-agent | 1.5 ★ | 3.1 | 1.7 | 1.2 ★ | 3.0 | 2.1 | 2.5 ★ | 11.0 | 8.4 |
| Strands | 1.2 ★ | 2.2 | 1.4 | 1.2 ★ | 2.8 | 2.0 | 4.5 | 3.1 | 4.3 |
| Droid | 2.6 ★ | 2.2 | 2.7 | 1.6 | 4.5 | 3.3 | 3.9 | 3.9 | 2.7 |
| DeepSeek Harness | · | · | · | · | · | · | 5.1 ★ | · | · |

## Harnesses on the same model

90% ranges over tasks in brackets. ⭐ on a harness: it beat the reference. ⭐ on the reference: nothing in the group beat it.
– in a time column: fewer than 8 shared tasks had a trial on both sides that ran while the test machine's CPU was not busy, so no time ratio is given.

**Opus 5.5 low**, reference Claude Code (25/27)

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 26 vs 25 | 0.74× (0.69–0.80) | 1.19× (1.05–1.34) |
| Goose ⭐ | 27 vs 25 | 0.81× (0.76–0.87) | 1.01× (0.86–1.18) |
| OpenCode | 26 vs 25 | 1.29× (1.20–1.39) | 1.71× (1.45–2.02) |
| Kilo | 27 vs 25 | 2.04× (1.81–2.30) | 2.09× (1.72–2.53) |
| Cline | 26 vs 25 | 1.14× (1.01–1.29) | 1.27× (1.10–1.47) |
| Terminus-2 | 25 vs 25 | 0.88× (0.77–1.02) | 2.37× (1.86–3.01) |
| mini-swe-agent | 27 vs 25 | 0.71× (0.63–0.80) | 1.72× (1.44–2.06) |
| Strands | 27 vs 25 | 1.01× (0.92–1.10) | 1.55× (1.16–2.05) |
| Droid | 24 vs 25 | 0.79× (0.69–0.92) | 1.43× (0.95–2.15) |

**Opus 5.5 medium**, reference Claude Code (27/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 26 vs 27 | 0.69× (0.63–0.76) | 1.53× (1.36–1.73) |
| Goose | 27 vs 27 | 0.79× (0.72–0.86) | 1.32× (1.12–1.54) |
| OpenCode | 26 vs 27 | 0.95× (0.85–1.07) | 2.22× (1.94–2.55) |
| Kilo | 27 vs 27 | 0.98× (0.86–1.11) | 1.32× (1.16–1.51) |
| Cline | 27 vs 27 | 1.60× (1.34–1.92) | 1.63× (1.33–1.99) |
| Terminus-2 | 27 vs 27 | 1.17× (0.97–1.41) | 2.16× (1.73–2.71) |
| mini-swe-agent | 26 vs 27 | 0.86× (0.76–0.97) | 1.91× (1.51–2.40) |
| Strands | 25 vs 27 | 0.92× (0.84–1.01) | 1.63× (1.35–1.97) |
| Droid | 24 vs 27 | 1.04× (0.91–1.18) | 1.40× (0.92–2.15) |

**Opus 5.5 high**, reference Claude Code (26/27)

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi ⭐ | 27 vs 26 | 0.72× (0.65–0.80) | 1.01× (0.82–1.23) |
| Goose ⭐ | 27 vs 26 | 0.79× (0.70–0.88) | 0.96× (0.80–1.14) |
| OpenCode | 27 vs 26 | 0.95× (0.84–1.07) | 1.21× (1.00–1.47) |
| Kilo | 27 vs 26 | 1.14× (1.02–1.28) | 1.23× (1.03–1.47) |
| Cline | 27 vs 26 | 1.35× (1.14–1.60) | 1.11× (0.96–1.29) |
| Terminus-2 | 27 vs 26 | 1.05× (0.95–1.17) | 1.89× (1.52–2.33) |
| mini-swe-agent | 26 vs 26 | 0.90× (0.78–1.04) | 1.60× (1.26–2.01) |
| Strands | 27 vs 26 | 0.93× (0.82–1.04) | 1.35× (1.04–1.74) |
| Droid | 25 vs 26 | 0.97× (0.86–1.08) | 1.12× (0.80–1.56) |

**Sonnet 5.5**, reference Claude Code (25/27)

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi ⭐ | 26 vs 25 | 0.63× (0.58–0.69) | 0.85× (0.72–0.99) |
| Goose ⭐ | 26 vs 25 | 0.62× (0.57–0.67) | 0.90× (0.82–0.98) |
| OpenCode | 26 vs 25 | 0.76× (0.68–0.84) | 1.50× (1.30–1.73) |
| Kilo | 26 vs 25 | 0.86× (0.77–0.97) | 1.60× (1.32–1.93) |
| Cline | 26 vs 25 | 1.27× (1.10–1.47) | 1.03× (0.92–1.16) |
| Terminus-2 | 26 vs 25 | 0.99× (0.85–1.14) | 2.64× (1.89–3.70) |
| mini-swe-agent | 26 vs 25 | 0.74× (0.67–0.82) | 1.52× (1.32–1.76) |
| Strands | 27 vs 25 | 0.72× (0.66–0.79) | 1.32× (1.18–1.47) |
| Droid | 22 vs 25 | 1.03× (0.86–1.23) | 1.04× (0.83–1.30) |

**Kimi K3**, reference Goose (26/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Claude Code | 25 vs 26 | 1.83× (1.57–2.13) | 1.12× (0.76–1.64) |
| Pi | 27 vs 26 | 1.19× (1.03–1.39) | 1.12× (0.93–1.34) |
| OpenCode | 26 vs 26 | 1.77× (1.50–2.09) | 1.50× (1.31–1.72) |
| Kilo | 24 vs 26 | 2.16× (1.84–2.53) | 1.38× (1.10–1.73) |
| Cline | 26 vs 26 | 1.31× (1.18–1.44) | 1.14× (1.01–1.29) |
| Terminus-2 | 25 vs 26 | 0.82× (0.72–0.92) | 1.53× (1.35–1.73) |
| mini-swe-agent | 26 vs 26 | 1.97× (1.68–2.30) | 1.94× (1.57–2.40) |
| Strands | 24 vs 26 | 1.63× (1.39–1.91) | 1.70× (1.40–2.06) |
| Droid | 26 vs 26 | 1.77× (1.56–2.01) | 1.30× (1.00–1.69) |

**GPT-6.1 Terra**, reference Goose (25/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Claude Code | 26 vs 25 | 1.67× (1.46–1.91) | 1.50× (1.09–2.06) |
| Pi | 25 vs 25 | 1.14× (0.98–1.32) | 1.39× (1.23–1.58) |
| OpenCode | 25 vs 25 | 1.86× (1.61–2.16) | 2.17× (1.81–2.60) |
| Kilo | 26 vs 25 | 2.78× (2.44–3.16) | 2.49× (2.08–2.99) |
| Cline | 25 vs 25 | 1.54× (1.34–1.76) | 1.61× (1.38–1.88) |
| Terminus-2 | 26 vs 25 | 0.94× (0.82–1.07) | 1.34× (1.15–1.57) |
| mini-swe-agent | 26 vs 25 | 1.14× (0.96–1.36) | 1.96× (1.55–2.48) |
| Strands | 26 vs 25 | 3.82× (3.30–4.43) | 2.08× (1.71–2.54) |
| Droid | 23 vs 25 | 1.97× (1.72–2.25) | 1.87× (1.43–2.45) |

**DeepSeek V4.1 Flash**, reference Goose (23/27)

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Claude Code | 26 vs 23 | 0.78× (0.63–0.96) | 1.11× (0.77–1.59) |
| Pi ⭐ | 26 vs 23 | 0.90× (0.72–1.13) | 0.70× (0.56–0.87) |
| OpenCode ⭐ | 26 vs 23 | 0.82× (0.63–1.07) | 0.83× (0.62–1.10) |
| Kilo | 26 vs 23 | 1.18× (0.95–1.48) | 1.08× (0.84–1.39) |
| Cline | 25 vs 23 | 1.48× (1.24–1.75) | 1.18× (0.93–1.51) |
| Terminus-2 | 26 vs 23 | 0.91× (0.74–1.12) | 1.22× (0.97–1.53) |
| mini-swe-agent ⭐ | 24 vs 23 | 0.82× (0.67–1.00) | 1.05× (0.85–1.29) |
| Strands | 24 vs 23 | 1.89× (1.52–2.35) | 2.21× (1.68–2.92) |
| Droid | 25 vs 23 | 0.83× (0.68–1.00) | 1.15× (0.78–1.68) |
| DeepSeek Harness ⭐ | 24 vs 23 | 0.93× (0.77–1.13) | 1.00× (0.83–1.22) |

**GLM 5.3**, reference Goose (21/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 22 vs 21 | 1.13× (0.73–1.75) | 1.10× (0.73–1.67) |
| OpenCode | 22 vs 21 | 2.09× (1.52–2.88) | 1.01× (0.62–1.64) |
| Kilo | 20 vs 21 | 2.72× (2.03–3.64) | 1.22× (0.88–1.69) |
| Cline | 24 vs 21 | 2.41× (1.67–3.49) | 0.92× (0.62–1.36) |
| Terminus-2 | 23 vs 21 | 1.67× (1.12–2.51) | 1.83× (1.30–2.58) |
| mini-swe-agent | 20 vs 21 | 1.83× (1.16–2.89) | 2.57× (1.81–3.64) |
| Strands | 21 vs 21 | 1.33× (0.68–2.59) | 1.17× (0.69–1.98) |
| Droid | 22 vs 21 | 2.84× (2.05–3.94) | 0.96× (0.63–1.47) |

**Grok 4.7**, reference Goose (26/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 24 vs 26 | 1.44× (1.17–1.77) | – |
| OpenCode | 26 vs 26 | 1.51× (1.22–1.87) | – |
| Kilo | 22 vs 26 | 1.51× (1.12–2.05) | – |
| Cline | 26 vs 26 | 1.29× (1.03–1.61) | – |
| Terminus-2 | 24 vs 26 | 0.88× (0.70–1.12) | 0.78× (0.57–1.06) |
| mini-swe-agent | 27 vs 26 | 1.05× (0.81–1.36) | 1.22× (0.87–1.72) |
| Strands | 24 vs 26 | 1.01× (0.76–1.33) | – |
| Droid | 22 vs 26 | 1.08× (0.77–1.50) | – |

## Models on the same harness

**Claude Code**, reference Claude Code on Opus 5.5 medium (27/27) ⭐

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 25 vs 27 | 0.81× (0.74–0.87) | 0.73× (0.65–0.82) |
| Opus 5.5 high | 26 vs 27 | 1.27× (1.16–1.38) | 1.41× (1.18–1.68) |
| Sonnet 5.5 | 25 vs 27 | 0.38× (0.33–0.44) | 0.74× (0.57–0.96) |
| Kimi K3 | 25 vs 27 | 1.40× (1.14–1.71) | 2.05× (1.44–2.93) |
| GPT-6.1 Terra | 26 vs 27 | 0.34× (0.26–0.45) | 1.60× (1.11–2.32) |
| DeepSeek V4.1 Flash | 26 vs 27 | 0.24× (0.19–0.29) | 2.89× (2.13–3.93) |

**Pi**, reference Pi on Opus 5.5 medium (26/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 26 vs 26 | 0.87× (0.79–0.95) | 0.57× (0.48–0.66) |
| Opus 5.5 high | 27 vs 26 | 1.32× (1.18–1.48) | 0.93× (0.74–1.16) |
| Sonnet 5.5 ⭐ | 26 vs 26 | 0.35× (0.30–0.40) | 0.41× (0.31–0.53) |
| Kimi K3 | 27 vs 26 | 1.32× (1.15–1.52) | 1.42× (1.14–1.77) |
| GPT-6.1 Terra | 25 vs 26 | 0.34× (0.28–0.42) | 0.97× (0.77–1.22) |
| DeepSeek V4.1 Flash | 26 vs 26 | 0.40× (0.32–0.50) | 1.08× (0.86–1.34) |
| GLM 5.3 | 22 vs 26 | 0.51× (0.32–0.81) | 1.91× (1.29–2.81) |
| Grok 4.7 | 24 vs 26 | 3.19× (2.18–4.67) | 3.95× (1.75–8.91) |

**Goose**, reference Goose on Opus 5.5 medium (27/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 27 vs 27 | 0.83× (0.72–0.96) | 0.56× (0.47–0.67) |
| Opus 5.5 high | 27 vs 27 | 1.26× (1.12–1.42) | 1.03× (0.85–1.23) |
| Sonnet 5.5 | 26 vs 27 | 0.30× (0.26–0.35) | 0.51× (0.40–0.64) |
| Kimi K3 | 26 vs 27 | 0.97× (0.84–1.12) | 1.48× (1.19–1.83) |
| GPT-6.1 Terra | 25 vs 27 | 0.26× (0.22–0.31) | 0.81× (0.65–1.02) |
| DeepSeek V4.1 Flash | 23 vs 27 | 0.38× (0.31–0.48) | 1.81× (1.35–2.44) |
| GLM 5.3 | 21 vs 27 | 0.39× (0.27–0.57) | 2.17× (1.38–3.42) |
| Grok 4.7 | 26 vs 27 | 1.94× (1.46–2.57) | 3.33× (1.72–6.43) |

**OpenCode**, reference OpenCode on Opus 5.5 medium (26/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 26 vs 26 | 1.09× (0.95–1.26) | 0.56× (0.45–0.71) |
| Opus 5.5 high | 27 vs 26 | 1.26× (1.08–1.47) | 0.77× (0.64–0.92) |
| Sonnet 5.5 ⭐ | 26 vs 26 | 0.30× (0.26–0.35) | 0.50× (0.40–0.62) |
| Kimi K3 | 26 vs 26 | 1.42× (1.19–1.70) | 1.31× (0.99–1.74) |
| GPT-6.1 Terra | 25 vs 26 | 0.40× (0.34–0.48) | 1.04× (0.84–1.29) |
| DeepSeek V4.1 Flash ⭐ | 26 vs 26 | 0.26× (0.20–0.33) | 0.89× (0.72–1.11) |
| GLM 5.3 | 22 vs 26 | 0.68× (0.49–0.94) | 1.30× (0.89–1.89) |
| Grok 4.7 | 26 vs 26 | 2.42× (1.79–3.28) | 2.06× (1.02–4.20) |

**Kilo**, reference Kilo on Opus 5.5 medium (27/27) ⭐

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 27 vs 27 | 1.68× (1.36–2.06) | 1.16× (0.88–1.52) |
| Opus 5.5 high | 27 vs 27 | 1.48× (1.30–1.68) | 1.31× (1.14–1.50) |
| Sonnet 5.5 | 26 vs 27 | 0.33× (0.28–0.39) | 0.89× (0.70–1.14) |
| Kimi K3 | 24 vs 27 | 1.68× (1.42–1.99) | 2.03× (1.61–2.57) |
| GPT-6.1 Terra | 26 vs 27 | 0.58× (0.47–0.72) | 2.01× (1.54–2.62) |
| DeepSeek V4.1 Flash | 26 vs 27 | 0.37× (0.30–0.44) | 1.97× (1.54–2.50) |
| GLM 5.3 | 20 vs 27 | 0.86× (0.64–1.16) | 2.63× (1.89–3.66) |
| Grok 4.7 | 22 vs 27 | 2.36× (1.85–3.00) | 3.51× (1.95–6.30) |

**Cline**, reference Cline on Opus 5.5 medium (27/27) ⭐

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 26 vs 27 | 0.57× (0.50–0.65) | 0.57× (0.49–0.67) |
| Opus 5.5 high | 27 vs 27 | 1.06× (0.93–1.22) | 0.97× (0.81–1.15) |
| Sonnet 5.5 | 26 vs 27 | 0.30× (0.26–0.34) | 0.47× (0.40–0.55) |
| Kimi K3 | 26 vs 27 | 0.62× (0.54–0.72) | 1.36× (1.16–1.60) |
| GPT-6.1 Terra | 25 vs 27 | 0.20× (0.17–0.23) | 1.06× (0.87–1.29) |
| DeepSeek V4.1 Flash | 25 vs 27 | 0.28× (0.23–0.33) | 1.76× (1.47–2.10) |
| GLM 5.3 | 24 vs 27 | 0.47× (0.35–0.62) | 1.62× (1.21–2.17) |
| Grok 4.7 | 26 vs 27 | 1.23× (0.88–1.71) | 1.98× (1.12–3.49) |

**Terminus-2**, reference Terminus-2 on Opus 5.5 medium (27/27) ⭐

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 25 vs 27 | 0.61× (0.51–0.73) | 0.80× (0.61–1.05) |
| Opus 5.5 high | 27 vs 27 | 1.14× (0.91–1.41) | 1.23× (1.00–1.51) |
| Sonnet 5.5 | 26 vs 27 | 0.32× (0.27–0.38) | 0.90× (0.65–1.25) |
| Kimi K3 | 25 vs 27 | 0.53× (0.45–0.64) | 1.37× (1.09–1.73) |
| GPT-6.1 Terra | 26 vs 27 | 0.16× (0.13–0.21) | 0.66× (0.53–0.83) |
| DeepSeek V4.1 Flash | 26 vs 27 | 0.24× (0.18–0.31) | 1.31× (1.01–1.69) |
| GLM 5.3 | 23 vs 27 | 0.44× (0.33–0.60) | 2.16× (1.46–3.18) |
| Grok 4.7 | 24 vs 27 | 1.15× (0.79–1.67) | 2.16× (1.37–3.40) |

**mini-swe-agent**, reference mini-swe-agent on Opus 5.5 medium (26/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 27 vs 26 | 0.67× (0.59–0.76) | 0.66× (0.54–0.81) |
| Opus 5.5 high | 26 vs 26 | 1.33× (1.17–1.52) | 1.18× (1.03–1.35) |
| Sonnet 5.5 ⭐ | 26 vs 26 | 0.33× (0.27–0.40) | 0.59× (0.47–0.74) |
| Kimi K3 | 26 vs 26 | 1.75× (1.52–2.02) | 1.98× (1.66–2.35) |
| GPT-6.1 Terra | 26 vs 26 | 0.27× (0.24–0.32) | 1.10× (0.93–1.30) |
| DeepSeek V4.1 Flash | 24 vs 26 | 0.29× (0.23–0.36) | 1.30× (1.05–1.60) |
| GLM 5.3 | 20 vs 26 | 0.66× (0.38–1.14) | 3.85× (2.76–5.37) |
| Grok 4.7 | 27 vs 26 | 1.87× (1.34–2.61) | 3.22× (2.07–5.03) |

**Strands**, reference Strands on Opus 5.5 medium (25/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 27 vs 25 | 0.88× (0.74–1.04) | 0.70× (0.54–0.89) |
| Opus 5.5 high | 27 vs 25 | 1.27× (1.12–1.45) | 1.17× (0.94–1.45) |
| Sonnet 5.5 ⭐ | 27 vs 25 | 0.30× (0.25–0.35) | 0.60× (0.46–0.78) |
| Kimi K3 | 24 vs 25 | 1.35× (1.13–1.61) | 2.03× (1.65–2.49) |
| GPT-6.1 Terra | 26 vs 25 | 0.85× (0.67–1.08) | 1.37× (1.09–1.71) |
| DeepSeek V4.1 Flash | 24 vs 25 | 0.62× (0.47–0.81) | 3.28× (2.54–4.24) |
| GLM 5.3 | 21 vs 25 | 0.45× (0.27–0.74) | 2.05× (1.49–2.83) |
| Grok 4.7 | 24 vs 25 | 1.67× (1.14–2.43) | 2.61× (1.38–4.95) |

**Droid**, reference Droid on Opus 5.5 medium (24/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 24 vs 24 | 0.62× (0.53–0.72) | 0.76× (0.65–0.90) |
| Opus 5.5 high | 25 vs 24 | 1.18× (1.05–1.31) | 1.18× (1.03–1.34) |
| Sonnet 5.5 | 22 vs 24 | 0.37× (0.29–0.48) | 0.50× (0.30–0.83) |
| Kimi K3 | 26 vs 24 | 1.30× (1.11–1.53) | 1.79× (1.32–2.43) |
| GPT-6.1 Terra | 23 vs 24 | 0.39× (0.32–0.48) | 1.47× (1.14–1.90) |
| DeepSeek V4.1 Flash | 25 vs 24 | 0.24× (0.19–0.30) | 1.81× (1.06–3.08) |
| GLM 5.3 | 22 vs 24 | 0.85× (0.67–1.07) | 2.25× (1.61–3.13) |
| Grok 4.7 | 22 vs 24 | 1.58× (1.11–2.26) | 1.64× (0.93–2.88) |

## Claude Code through a LiteLLM gateway (6 Nov 2026)

A second, smaller experiment ran Claude Code 2.1.312 through a self-hosted LiteLLM gateway on the same 27 tasks, one
attempt per task. The gateway sits between the harness and the model provider, so these runs show what the route
correlated with, not what the model did. Ratios divide by the reference in the fifth column (0.80× means 20% less),
using only the tasks both arms ran; cost and time ratios compare summed cost and summed trial minutes (so Terra's 2.13× here is 1.67× task by task in the comparison tables), not the medians shown, and the
medians here are per task (each task's latest trial), where the tables above use per-trial medians. "Translated"
means the gateway converts Claude Code's Anthropic-format requests into the provider's own API; "native" or
"pass-through" means the provider accepts Anthropic-format requests directly. Rows compared with Goose change the
harness as well as the route. The Claude Code cells marked † in the tables above come from the DeepSeek (native),
Kimi and Terra rows here; the control and the translated DeepSeek route appear only here.

| Arm (all Claude Code) | Passed | Cost, 27 tasks | Median minutes | Reference | Total cost ÷ reference | Total time ÷ reference |
|---|---|---|---|---|---|---|
| Opus 5.5 medium (control) | 26 of 27 | $5.86 | 2.3 | Claude Code direct, Opus 5.5 medium: 27 of 27 | 1.48× | 0.78× |
| DeepSeek V4.1 Flash, native route | 26 of 27 | $1.09 | 2.0 | Goose, DeepSeek via Fireworks: 23 of 27 | 0.78× | 0.78× |
| DeepSeek V4.1 Flash, translated route | 23 of 27 | $1.49 | 2.6 | Same model, native route: 26 of 27 | 1.37× | 1.51× |
| Kimi K3, pass-through to Fireworks | 25 of 27 | $5.92 | 1.6 | Goose, Kimi K3 via Fireworks: 26 of 27 | 1.83× | 1.08× |
| GPT-6.1 Terra, translated to the Responses API | 26 of 27 | $2.01 | 1.3 | Goose, Terra via OpenAI: 25 of 27 | 2.13× | 1.24× |

What the table showed:

- **The control kept Opus within one trial of direct.** Through the gateway Claude Code passed 26 of 27 against 27 of
  27 direct, and every trial used Opus 5.5 only. The miss was `sanitize-git-repo` (reward 0, not a timeout); the direct
  Claude Code run on that task passed. The gateway run cost 1.48× the direct run, and its uncached input was about
  2.1× higher (696k against 333k tokens) while total prompt and cache-read tokens were similar. The cause was not
  established.
- **Translation correlated with four timeouts on DeepSeek.** The translated route passed 23 of 27 and the native
  route 26 of 27, with the same model behind both. All four translated misses (`cobol-modernization`, `mailman`,
  `financial-document-processor`, `headless-terminal`) were agent timeouts, and the translated route took 1.51× the
  total minutes.
- **Kimi K3 passed only on the pass-through route.** The translated route failed on Fireworks rejecting
  `thinking_blocks` (two early trials were excluded). After the gateway forwarded to Fireworks' own Anthropic
  endpoint, Kimi passed 25 of 27: one plain fail (`sqlite-with-gcov`)
  and one API error (`financial-document-processor`).
- **Terra's cost first read $11.41 and was $2.01.** LiteLLM 1.83.7 did not report cached tokens on the Responses-API
  translation, so the bench priced about 9.9M cached prompt tokens as fresh input. The gateway's own spend log
  (564 calls, all matched to trials) gave the corrected cost, and the ledger now carries both figures.
  Terra passed 26 of 27 (miss: `sanitize-git-repo`).

Notes: Harbor pointed every Claude Code model alias and its subagent model at each arm's own model; the Opus control put
back Haiku, Sonnet and Claude Code's default subagent choice, as the direct arm uses them. DeepSeek and Kimi ran at
high effort, Terra and Opus at medium. Costs are token counts at list prices, except Terra's, which come from the gateway's spend log. A trial that hit the agent time limit counts by its tests, as in the rest of this report: Kimi and Terra each
had two timeouts (`kv-store-grpc`, `pypi-server`) whose tests passed afterwards, and the four translated DeepSeek timeouts
failed their tests. Three Opus control trials failed
at setup because the container could not resolve the package mirrors (`apt-get update` failed before the agent
started); they were rerun on the same arm and all three passed. With one attempt per task, a one-task difference is
too small to read much into.

## Droid (13 Nov 2026)

Factory's Droid 1.0.0 ran headless (`droid exec --auto high`) on the same 27 tasks in all nine model columns (seven models, Opus at three efforts), one attempt per task: 243 trials, $46.23 at list prices. Its cells are in the matrix and comparisons above. Three of the tasks ask the agent to leave a server running for the tests (`kv-store-grpc`, `hf-model-inference`, `pypi-server`). Droid's failures on those three were on the tests that need the server still running, and in most of them the Droid program's log shows it stopping the agent's server on exit, so the table shows Droid's passes with and without those three tasks. Ratios divide by the reference harness on the same model over all 27 tasks (0.80× means 20% less).

| Model (all Droid) | Passed, 27 tasks | Passed, 24 tasks without a server | Reference, same 24 tasks | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|---|---|
| Opus 5.5 low | 24 of 27 | 24 of 24 | Claude Code: 22 of 24 | 0.79× | 1.43× |
| Opus 5.5 medium | 24 of 27 | 24 of 24 | Claude Code: 24 of 24 | 1.04× | 1.40× |
| Opus 5.5 high | 25 of 27 | 24 of 24 | Claude Code: 23 of 24 | 0.97× | 1.12× |
| Sonnet 5.5 | 22 of 27 | 22 of 24 | Claude Code: 22 of 24 | 1.03× | 1.04× |
| Kimi K3 | 26 of 27 | 24 of 24 | Goose: 23 of 24 | 1.77× | 1.30× |
| GPT-6.1 Terra | 23 of 27 | 22 of 24 | Goose: 22 of 24 | 1.97× | 1.87× |
| DeepSeek V4.1 Flash | 25 of 27 | 23 of 24 | Goose: 20 of 24 | 0.83× | 1.15× |
| GLM 5.3 | 22 of 27 | 22 of 24 | Goose: 19 of 24 | 2.84× | 0.96× |
| Grok 4.7 | 22 of 27 | 21 of 24 | Goose: 22 of 24 | 1.08× | – |

- **In 17 of the 20 failed server-task trials, the Droid program's log shows it stopping the server the task needed.** 20 of Droid's 30 failed trials were on the three server tasks: `kv-store-grpc` 0 of 9, `hf-model-inference` 4 of 9, `pypi-server` 3 of 9. The other harnesses passed these tasks about 94% of the time. In each of the 20, the tests that need the running server failed and the checks on the agent's files passed (`pypi-server` has only the live-server test). In 17 trials the agent started the server through the Droid program's background-command option and declared the task finished. The Droid program then waited 5 minutes and added one reminder to the conversation, which listed a dev server as something to kill ("If you do not need it (for example a dev server …), kill it") and said "the session stops waiting for it and finishes"; it did not say the Droid program would stop the process itself. On exit, the Droid program sent SIGKILL to the background processes started through that option, the server included, as its log records. In the other 3 trials the agent started the server with a plain `nohup … &`, and the server was gone when the tests ran. In the 7 passing server-task trials the log shows no such stop; 6 of them started the server detached (`setsid nohup … &`), often after a first start that did not survive.
- **Without the three server tasks, Droid passed as many as the reference harness, or more, in 8 of 9 model columns.** On the other 24 tasks Droid passed all 24 on Opus at every effort and on Kimi K3, and on GLM 5.3 it passed 22 of 24, more than any other harness on that model (16 to 21). The exception was Grok 4.7 (21 against Goose's 22). Most of these margins are one or two tasks.
- **The other 10 failures were the agent's own.** Nine were wrong work the agent declared finished (`mteb-retrieve` 4, `sanitize-git-repo` 2, `openssl-selfsigned-cert`, `db-wal-recovery`, `sparql-university`). One was an answer lookup: the agent on DeepSeek used the Droid program's built-in web search and fetch tools to download the benchmark's solution for `mteb-retrieve`, which counts as a fail.
- **Read Droid's time with the server tasks in mind.** Each server-task trial that used the background option includes the Droid program's 5-minute wait, and the time ratios in the table (all 27 tasks) include those waits. The Droid program also hides secret values in the agent's tool output; the two `sanitize-git-repo` failures happened with this in place, and Opus passed the task with it.

Droid's adapter supplies each model through a custom-model setting with the provider's own API key; no Factory account is used. Opus and Sonnet ran at medium effort by default (Opus also at low and high), the open models at high, Terra and Grok at medium. 38 Droid trials were lost to our side and rerun on the same arm: 25 when our Anthropic key ran out of credit and 13 when Docker answered too slowly for Harbor's start-up check. Neither group is in the 243 trials or the results: the credit failures are listed as exclusions, and the Docker ones never started the agent.

## Method

terminal-bench 2.0 on Harbor 0.25.0. In the 3 Nov results, 37 trials (2%) that may have failed for reasons on our side were rerun (Droid's 38 reruns are in the Droid section), with 10 passes from the same period rerun as a control (9 passed again); both attempts count. Open models (DeepSeek, GLM, Kimi) ran through Fireworks at high effort; Terra and
Grok at medium. Claude Code ran directly only on Claude models; its Kimi K3, Terra and DeepSeek results (†) came through a LiteLLM gateway (see the gateway section). On the other models Goose is the reference harness. Droid was added on 13 Nov 2026 with the same tasks and models (see the Droid section). DeepSeek
Harness runs only on DeepSeek, so it has no model comparison. A pass is the task's own tests passing. Medians are per
trial. Pass differences of one or two tasks are within noise at this sample size.

The interactive version of this report is [report.html](report.html) (open it in a browser).
