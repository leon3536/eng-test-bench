# Harness bench results

Preliminary, 3 Nov 2026. About 2% of trials are pending a rerun for the final results.

2,178 trials, $334 in model fees, 11 answer lookups counted as fails.

- Ten agent harnesses ran the same 27 IT, e-commerce and SaaS tasks (terminal-bench 2.0) on six models, one attempt per task.
- Every ratio divides by a reference on the same tasks: 0.80× means 20% less than the reference.
- The overall reference is **Claude Code on Opus 5.5 medium**.
- Harnesses are compared with **Claude Code** on Claude models and with **Goose** on the other models, at the same model and effort.
- Models are compared with the same harness on **Opus 5.5 medium**.

## Executive insights

1. **Model choice correlates more with task success; harness choice correlates more with cost.** On Claude models every harness passed 25–27 of 27 tasks, a spread too small to tell apart at this sample size. Cost varied widely: at Opus medium, Pi cost 31% less than Claude Code, Goose 21% less, and Cline 60% more.
2. **Sonnet 5.5 stood out among the mid-tier models on these terminal-bench tasks.** Across the 8 harnesses that ran every model, Sonnet passed 209 of 216 (97%), against 92–94% for Kimi K3, GPT-6.1 Terra and DeepSeek, 89% for Grok and 80% for GLM. It was also the fastest (1.6 min median per task, against 2.1–5.6) and tied with Terra as the cheapest (2.6¢). The pass gap to the next models is a few tasks; the cost and speed gaps are larger. For scale, Opus medium passed 211 of 216 at about three times Sonnet's cost.
3. **Claude Code was the fastest harness on Claude models, not the cheapest.** The other harnesses took 1.3–2.2× as long as Claude Code at Opus medium. Where people wait on the agent, that speed may justify its cost; for unattended batch work, Pi and Goose reached the same pass counts for less.
4. **Open models cost less, and each tended to do best on a particular harness.** The top harness differed by model: Kimi K3 on Pi (27 of 27, among the cheapest and fastest), DeepSeek V4.1 Flash on OpenCode or Terminus-2 (26 of 27 at the lowest cost), GLM 5.3 on Cline (24 of 27). GLM had the widest spread across harnesses (20 to 24). Most of these margins are one to four tasks, so treat each pairing as a starting point to confirm, not a settled ranking.
5. **Grok 4.7 was the most expensive and slowest option in this test.** On Goose it cost 1.9× Opus medium and took 3.3× as long, with 25 vs 27 passes. Some of the slowness coincided with an overloaded test machine and a rerun is planned; the cost gap is less likely to change.
6. **Some agents with internet access looked up the benchmark's answers.** In 11 runs, mostly on DeepSeek, the agent downloaded the benchmark's own answer key and used it; we count those as fails. Published scores for internet-connected agents may be inflated, and the same behaviour in production means fetching and running unvetted code, so internet access warrants a deliberate policy.

## Tasks passed (of 27)

| Harness | Opus 5.5 low | Opus 5.5 medium | Opus 5.5 high | Sonnet 5.5 | Kimi K3 | GPT-6.1 Terra | DeepSeek V4.1 Flash | GLM 5.3 | Grok 4.7 |
|---|---|---|---|---|---|---|---|---|---|
| Claude Code | 25/27 | 27/27 | 26/27 | 25/27 | · | · | · | · | · |
| Pi | 26/27 ★ | 26/27 | 27/27 ★ | 26/27 ⭐ | 27/27 | 25/27 | 26/27 ⊘ ★ | 22/27 | 24/27 ⊘ |
| Goose | 27/27 ⭐ | 27/27 | 27/27 ★ | 26/27 ★ | 26/27 | 25/27 | 23/27 | 21/27 | 25/27 |
| OpenCode | 26/27 | 26/27 | 27/27 | 26/27 ★ | 26/27 | 25/27 | 26/27 ⊘2 ⭐ | 22/27 | 24/26 |
| Kilo | 27/27 | 27/27 | 27/27 | 26/27 | 24/27 | 25/27 | 26/27 ⊘ | 20/27 | 21/26 |
| Cline | 26/27 | 27/27 | 27/27 | 26/27 | 26/27 | 25/27 | 25/27 | 24/27 | 24/26 ⊘ |
| Terminus-2 | 25/27 | 27/27 | 27/27 | 26/27 | 25/27 | 26/27 | 26/27 | 23/27 | 23/27 ⊘ |
| mini-swe-agent | 27/27 ★ | 26/27 | 26/27 | 26/27 ★ | 26/27 | 26/27 | 24/27 ⊘2 ★ | 20/27 | 27/27 |
| Strands | 27/27 ★ | 25/27 | 27/27 | 27/27 ★ | 24/27 | 26/27 | 23/27 ⊘ | 21/27 | 24/27 |
| DeepSeek Harness | · | · | · | · | · | · | 24/27 ⊘ ★ | · | · |

⭐ local optimum: beats both its column's reference harness and its own harness on Opus 5.5 medium.
★ beats one of the two. "Beats" means no fewer passes, cost and time within 5% or better, and at least 5% cheaper
or faster. ⊘ marks answer lookups (the agent downloaded the benchmark's own answers), counted as fails. · not run.

## Median cost per task

| Harness | Opus 5.5 low | Opus 5.5 medium | Opus 5.5 high | Sonnet 5.5 | Kimi K3 | GPT-6.1 Terra | DeepSeek V4.1 Flash | GLM 5.3 | Grok 4.7 |
|---|---|---|---|---|---|---|---|---|---|
| Claude Code | 7.6¢ | 8.4¢ | $0.11 | 3.6¢ | · | · | · | · | · |
| Pi | 5.6¢ ★ | 5.8¢ | 7.8¢ ★ | 2.1¢ ⭐ | 6.8¢ | 1.7¢ | 3.1¢ ★ | 2.5¢ | $0.20 |
| Goose | 6.0¢ ⭐ | 7.4¢ | 8.8¢ ★ | 2.3¢ ★ | 7.0¢ | 1.6¢ | 3.0¢ | 2.6¢ | $0.13 |
| OpenCode | $0.10 | 9.3¢ | $0.10 | 2.6¢ ★ | $0.14 | 3.6¢ | 2.2¢ ⭐ | 7.2¢ | $0.24 |
| Kilo | $0.17 | 7.9¢ | $0.15 | 2.9¢ | $0.14 | 4.7¢ | 3.9¢ | 7.7¢ | $0.20 |
| Cline | 7.7¢ | $0.12 | $0.14 | 3.9¢ | 8.8¢ | 2.9¢ | 5.0¢ | 5.6¢ | $0.28 |
| Terminus-2 | 7.3¢ | $0.11 | $0.12 | 4.1¢ | 5.9¢ | 1.8¢ | 2.3¢ | 4.5¢ | $0.12 |
| mini-swe-agent | 5.4¢ ★ | 7.5¢ | 8.9¢ | 2.6¢ ★ | $0.13 | 2.4¢ | 2.7¢ ★ | 8.9¢ | $0.15 |
| Strands | 8.5¢ ★ | 8.6¢ | $0.12 | 2.5¢ ★ | 8.9¢ | 7.8¢ | 5.8¢ | 5.0¢ | $0.14 |
| DeepSeek Harness | · | · | · | · | · | · | 2.9¢ ★ | · | · |

## Median minutes per task

| Harness | Opus 5.5 low | Opus 5.5 medium | Opus 5.5 high | Sonnet 5.5 | Kimi K3 | GPT-6.1 Terra | DeepSeek V4.1 Flash | GLM 5.3 | Grok 4.7 |
|---|---|---|---|---|---|---|---|---|---|
| Claude Code | 1.0 | 3.6 | 1.4 | 1.1 | · | · | · | · | · |
| Pi | 1.7 ★ | 4.1 | 1.9 ★ | 1.6 ⭐ | 2.4 | 2.1 | 2.7 ★ | 3.3 | 8.1 |
| Goose | 1.0 ⭐ | 2.2 | 1.9 ★ | 1.1 ★ | 2.3 | 1.5 | 2.3 | 2.5 | 4.6 |
| OpenCode | 1.8 | 3.2 | 1.8 | 1.7 ★ | 3.5 | 2.8 | 3.1 ⭐ | 4.6 | 5.3 |
| Kilo | 2.0 | 1.3 | 2.0 | 1.6 | 2.6 | 3.7 | 3.0 | 3.3 | 6.0 |
| Cline | 2.4 | 3.8 | 2.7 | 2.5 | 3.1 | 3.1 | 4.9 | 3.0 | 6.3 |
| Terminus-2 | 1.9 | 2.1 | 2.8 | 1.9 | 3.3 | 1.4 | 2.5 | 4.4 | 4.8 |
| mini-swe-agent | 1.5 ★ | 3.1 | 1.7 | 1.2 ★ | 2.8 | 2.1 | 2.5 ★ | 9.3 | 8.1 |
| Strands | 1.2 ★ | 2.2 | 1.4 | 1.2 ★ | 2.4 | 2.0 | 4.1 | 3.0 | 4.0 |
| DeepSeek Harness | · | · | · | · | · | · | 5.1 ★ | · | · |

## Harnesses on the same model

90% ranges over tasks in brackets. ⭐ on a harness: it beat the reference. ⭐ on the reference: nothing in the group beat it.

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

**Opus 5.5 high**, reference Claude Code (26/27)

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi ⭐ | 27 vs 26 | 0.72× (0.65–0.80) | 1.01× (0.82–1.23) |
| Goose ⭐ | 27 vs 26 | 0.79× (0.70–0.88) | 0.96× (0.80–1.14) |
| OpenCode | 27 vs 26 | 0.95× (0.84–1.07) | 1.21× (1.00–1.47) |
| Kilo | 27 vs 26 | 1.14× (1.02–1.28) | 1.23× (1.03–1.47) |
| Cline | 27 vs 26 | 1.35× (1.14–1.60) | 1.11× (0.96–1.29) |
| Terminus-2 | 27 vs 26 | 1.07× (0.96–1.18) | 1.89× (1.52–2.33) |
| mini-swe-agent | 26 vs 26 | 0.90× (0.78–1.04) | 1.60× (1.26–2.01) |
| Strands | 27 vs 26 | 0.93× (0.82–1.04) | 1.35× (1.04–1.74) |

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

**Kimi K3**, reference Goose (26/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 27 vs 26 | 1.19× (1.03–1.39) | 1.12× (0.93–1.34) |
| OpenCode | 26 vs 26 | 1.77× (1.50–2.09) | 1.50× (1.31–1.72) |
| Kilo | 24 vs 26 | 2.17× (1.84–2.56) | 1.39× (1.10–1.74) |
| Cline | 26 vs 26 | 1.31× (1.18–1.44) | 1.14× (1.01–1.29) |
| Terminus-2 | 25 vs 26 | 0.80× (0.71–0.91) | 1.50× (1.32–1.71) |
| mini-swe-agent | 26 vs 26 | 1.99× (1.68–2.36) | 1.92× (1.56–2.38) |
| Strands | 24 vs 26 | 1.64× (1.39–1.95) | 1.70× (1.40–2.06) |

**GPT-6.1 Terra**, reference Goose (25/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 25 vs 25 | 1.14× (0.98–1.32) | 1.39× (1.23–1.58) |
| OpenCode | 25 vs 25 | 1.86× (1.61–2.16) | 2.17× (1.81–2.60) |
| Kilo | 25 vs 25 | 2.77× (2.43–3.16) | 2.52× (2.10–3.01) |
| Cline | 25 vs 25 | 1.54× (1.34–1.76) | 1.61× (1.38–1.88) |
| Terminus-2 | 26 vs 25 | 0.94× (0.82–1.08) | 1.35× (1.15–1.59) |
| mini-swe-agent | 26 vs 25 | 1.14× (0.96–1.36) | 1.96× (1.55–2.48) |
| Strands | 26 vs 25 | 3.82× (3.30–4.43) | 2.08× (1.71–2.54) |

**DeepSeek V4.1 Flash**, reference Goose (23/27)

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi ⭐ | 26 vs 23 | 0.90× (0.72–1.13) | 0.69× (0.55–0.87) |
| OpenCode ⭐ | 26 vs 23 | 0.82× (0.63–1.06) | 0.82× (0.61–1.10) |
| Kilo | 26 vs 23 | 1.18× (0.94–1.48) | 1.07× (0.83–1.37) |
| Cline | 25 vs 23 | 1.47× (1.24–1.75) | 1.17× (0.91–1.51) |
| Terminus-2 | 26 vs 23 | 0.90× (0.72–1.11) | 1.20× (0.95–1.52) |
| mini-swe-agent ⭐ | 24 vs 23 | 0.81× (0.66–1.00) | 1.04× (0.84–1.29) |
| Strands | 23 vs 23 | 1.81× (1.48–2.22) | 2.18× (1.64–2.89) |
| DeepSeek Harness ⭐ | 24 vs 23 | 0.93× (0.77–1.13) | 1.00× (0.83–1.22) |

**GLM 5.3**, reference Goose (21/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 22 vs 21 | 1.12× (0.72–1.74) | 1.13× (0.74–1.71) |
| OpenCode | 22 vs 21 | 2.20× (1.60–3.01) | 1.05× (0.64–1.71) |
| Kilo | 20 vs 21 | 2.71× (1.99–3.68) | 1.24× (0.89–1.74) |
| Cline | 24 vs 21 | 2.49× (1.75–3.53) | 0.94× (0.64–1.39) |
| Terminus-2 | 23 vs 21 | 1.74× (1.18–2.56) | 1.89× (1.35–2.64) |
| mini-swe-agent | 20 vs 21 | 1.87× (1.19–2.94) | 2.63× (1.86–3.72) |
| Strands | 21 vs 21 | 1.38× (0.71–2.67) | 1.19× (0.71–2.01) |

**Grok 4.7**, reference Goose (25/27) ⭐

| Harness | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Pi | 24 vs 25 | 1.41× (1.15–1.74) | – |
| OpenCode | 24 vs 25 | 1.51× (1.18–1.93) | – |
| Kilo | 21 vs 25 | 1.48× (1.08–2.01) | – |
| Cline | 24 vs 25 | 1.32× (1.05–1.67) | – |
| Terminus-2 | 23 vs 25 | 0.91× (0.72–1.15) | 0.82× (0.59–1.14) |
| mini-swe-agent | 27 vs 25 | 1.07× (0.83–1.38) | 1.24× (0.89–1.74) |
| Strands | 24 vs 25 | 1.01× (0.76–1.34) | – |

## Models on the same harness

**Claude Code**, reference Claude Code on Opus 5.5 medium (27/27) ⭐

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 25 vs 27 | 0.81× (0.74–0.87) | 0.73× (0.65–0.82) |
| Opus 5.5 high | 26 vs 27 | 1.27× (1.16–1.38) | 1.41× (1.18–1.68) |
| Sonnet 5.5 | 25 vs 27 | 0.38× (0.33–0.44) | 0.74× (0.57–0.96) |

**Pi**, reference Pi on Opus 5.5 medium (26/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 26 vs 26 | 0.87× (0.79–0.95) | 0.57× (0.48–0.66) |
| Opus 5.5 high | 27 vs 26 | 1.32× (1.18–1.48) | 0.93× (0.74–1.16) |
| Sonnet 5.5 ⭐ | 26 vs 26 | 0.35× (0.30–0.40) | 0.41× (0.31–0.53) |
| Kimi K3 | 27 vs 26 | 1.32× (1.15–1.52) | 1.42× (1.14–1.77) |
| GPT-6.1 Terra | 25 vs 26 | 0.34× (0.28–0.42) | 0.97× (0.77–1.22) |
| DeepSeek V4.1 Flash | 26 vs 26 | 0.40× (0.32–0.50) | 1.08× (0.86–1.34) |
| GLM 5.3 | 22 vs 26 | 0.49× (0.31–0.77) | 1.91× (1.29–2.82) |
| Grok 4.7 | 24 vs 26 | 3.11× (2.12–4.57) | 3.95× (1.75–8.91) |

**Goose**, reference Goose on Opus 5.5 medium (27/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 27 vs 27 | 0.83× (0.72–0.96) | 0.56× (0.47–0.67) |
| Opus 5.5 high | 27 vs 27 | 1.26× (1.12–1.42) | 1.03× (0.85–1.23) |
| Sonnet 5.5 | 26 vs 27 | 0.30× (0.26–0.35) | 0.51× (0.40–0.64) |
| Kimi K3 | 26 vs 27 | 0.97× (0.84–1.12) | 1.48× (1.19–1.83) |
| GPT-6.1 Terra | 25 vs 27 | 0.26× (0.22–0.31) | 0.81× (0.65–1.02) |
| DeepSeek V4.1 Flash | 23 vs 27 | 0.39× (0.31–0.48) | 1.83× (1.36–2.46) |
| GLM 5.3 | 21 vs 27 | 0.38× (0.27–0.54) | 2.12× (1.35–3.35) |
| Grok 4.7 | 25 vs 27 | 1.93× (1.45–2.55) | 3.33× (1.72–6.43) |

**OpenCode**, reference OpenCode on Opus 5.5 medium (26/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 26 vs 26 | 1.09× (0.95–1.26) | 0.56× (0.45–0.71) |
| Opus 5.5 high | 27 vs 26 | 1.26× (1.08–1.47) | 0.77× (0.64–0.92) |
| Sonnet 5.5 ⭐ | 26 vs 26 | 0.30× (0.26–0.35) | 0.50× (0.40–0.62) |
| Kimi K3 | 26 vs 26 | 1.42× (1.19–1.70) | 1.31× (0.99–1.74) |
| GPT-6.1 Terra | 25 vs 26 | 0.40× (0.34–0.48) | 1.04× (0.84–1.29) |
| DeepSeek V4.1 Flash ⭐ | 26 vs 26 | 0.26× (0.20–0.33) | 0.89× (0.72–1.11) |
| GLM 5.3 | 22 vs 26 | 0.69× (0.50–0.97) | 1.31× (0.90–1.91) |
| Grok 4.7 | 24 vs 25 | 2.51× (1.80–3.51) | 2.06× (1.02–4.20) |

**Kilo**, reference Kilo on Opus 5.5 medium (27/27) ⭐

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 27 vs 27 | 1.68× (1.36–2.06) | 1.16× (0.88–1.52) |
| Opus 5.5 high | 27 vs 27 | 1.48× (1.30–1.68) | 1.31× (1.14–1.50) |
| Sonnet 5.5 | 26 vs 27 | 0.33× (0.28–0.39) | 0.89× (0.70–1.14) |
| Kimi K3 | 24 vs 27 | 1.70× (1.43–2.01) | 2.04× (1.61–2.57) |
| GPT-6.1 Terra | 25 vs 27 | 0.58× (0.47–0.72) | 2.03× (1.55–2.66) |
| DeepSeek V4.1 Flash | 26 vs 27 | 0.37× (0.30–0.44) | 1.97× (1.54–2.50) |
| GLM 5.3 | 20 vs 27 | 0.83× (0.61–1.13) | 2.63× (1.89–3.66) |
| Grok 4.7 | 21 vs 26 | 2.34× (1.82–3.02) | 3.51× (1.95–6.30) |

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
| Grok 4.7 | 24 vs 26 | 1.28× (0.91–1.81) | 2.03× (1.15–3.58) |

**Terminus-2**, reference Terminus-2 on Opus 5.5 medium (27/27) ⭐

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low | 25 vs 27 | 0.61× (0.51–0.73) | 0.80× (0.61–1.05) |
| Opus 5.5 high | 27 vs 27 | 1.15× (0.92–1.45) | 1.23× (1.00–1.51) |
| Sonnet 5.5 | 26 vs 27 | 0.32× (0.27–0.38) | 0.90× (0.65–1.25) |
| Kimi K3 | 25 vs 27 | 0.52× (0.44–0.63) | 1.35× (1.06–1.73) |
| GPT-6.1 Terra | 26 vs 27 | 0.17× (0.13–0.21) | 0.67× (0.53–0.84) |
| DeepSeek V4.1 Flash | 26 vs 27 | 0.23× (0.18–0.31) | 1.30× (1.00–1.68) |
| GLM 5.3 | 23 vs 27 | 0.45× (0.33–0.61) | 2.17× (1.48–3.19) |
| Grok 4.7 | 23 vs 27 | 1.18× (0.81–1.72) | 2.23× (1.43–3.46) |

**mini-swe-agent**, reference mini-swe-agent on Opus 5.5 medium (26/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 27 vs 26 | 0.67× (0.59–0.76) | 0.66× (0.54–0.81) |
| Opus 5.5 high | 26 vs 26 | 1.33× (1.17–1.52) | 1.18× (1.03–1.35) |
| Sonnet 5.5 ⭐ | 26 vs 26 | 0.33× (0.27–0.40) | 0.59× (0.47–0.74) |
| Kimi K3 | 26 vs 26 | 1.78× (1.54–2.06) | 1.96× (1.65–2.34) |
| GPT-6.1 Terra | 26 vs 26 | 0.27× (0.24–0.32) | 1.10× (0.93–1.30) |
| DeepSeek V4.1 Flash | 24 vs 26 | 0.29× (0.23–0.36) | 1.30× (1.05–1.60) |
| GLM 5.3 | 20 vs 26 | 0.65× (0.38–1.12) | 3.85× (2.76–5.37) |
| Grok 4.7 | 27 vs 26 | 1.89× (1.36–2.64) | 3.25× (2.09–5.06) |

**Strands**, reference Strands on Opus 5.5 medium (25/27)

| Model | Passes vs reference | Cost ÷ reference | Time ÷ reference |
|---|---|---|---|
| Opus 5.5 low ⭐ | 27 vs 25 | 0.88× (0.74–1.04) | 0.70× (0.54–0.89) |
| Opus 5.5 high | 27 vs 25 | 1.27× (1.12–1.45) | 1.17× (0.94–1.45) |
| Sonnet 5.5 ⭐ | 27 vs 25 | 0.30× (0.25–0.35) | 0.60× (0.46–0.78) |
| Kimi K3 | 24 vs 25 | 1.36× (1.14–1.63) | 2.03× (1.65–2.49) |
| GPT-6.1 Terra | 26 vs 25 | 0.85× (0.67–1.08) | 1.37× (1.09–1.71) |
| DeepSeek V4.1 Flash | 23 vs 25 | 0.60× (0.46–0.78) | 3.26× (2.52–4.22) |
| GLM 5.3 | 21 vs 25 | 0.45× (0.27–0.75) | 2.05× (1.49–2.83) |
| Grok 4.7 | 24 vs 25 | 1.66× (1.14–2.43) | 2.61× (1.38–4.95) |

## Method

terminal-bench 2.0 on Harbor 0.25.0. Open models (DeepSeek, GLM, Kimi) ran through Fireworks at high effort; Terra and
Grok at medium. Claude Code runs only on Claude models, so on the other models Goose is the reference harness. DeepSeek
Harness runs only on DeepSeek, so it has no model comparison. A pass is the task's own tests passing. Medians are per
trial. Pass differences of one or two tasks are within noise at this sample size.

The interactive version of this report is [report.html](report.html) (open it in a browser).
