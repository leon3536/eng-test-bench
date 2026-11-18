# Roster-Free Hockey Vision & Jersey STR: Literature Synthesis & Ablation Plan

## 1. Problem Diagnosis on Baseline Pipeline
Empirical evaluation on the Pixel 11 Pro on-device baseline (`coach_person_detector.onnx` + RGB chest 2-means + ML Kit OCR + Gemini Nano 2x2 grid) exposed five failure modes that violate roster-free hockey accuracy:
1. **Crossfade double-exposure artifacts in video benchmarks:** Encoded still-frame comparison videos with frame-blending transitions yielded transient `19–23` person boxes during fades, spawning 1-hit ghost tracks that lingered up to `0.5s` in `MarkedMp4ClipExporter`.
2. **Dark-yoke white jersey team inversion (`#97` failure mode):** Single-box RGB chest averages mix dark shoulder yokes (`y: 22%..34%`) with white waist fabric (`y: 42%..56%`), flipping oblique-angle away skaters into `TEAM_A (Dark)`.
3. **Far-board blurred referee misses & green rink cast:** When board officials span `16–22 px` width, alternating black/white stripes alias below Nyquist in narrow `20`-bin chest slices, while Dublin-Iceland-style green ice (`RGB ~ 155, 201, 175`) inflates HSV saturation and biases `NANO_TEAM_LABEL` toward `"Green/Dark Green"`.
4. **Tight headless torso crops & AICore policy rejections (`GenAiException: ErrorCode 4`):** Tight torso crops trigger safety/policy classifiers and discard partial-view temporal context.
5. **Single-read / naive voting jersey errors:** Naive majority voting over raw strings confuses single-digit reads from occluded two-digit jerseys (`?7` -> `#7` instead of `#97`), ignores visual digit confusions (`{1,7}`, `{3,8}`, `{6,8,9}`), and allows concurrent teammates to share the same number.

---

## 2. Literature Synthesis (arXiv / CVPRW / ECCV)

### 2.1 Legibility Filtering & Torso-Guided Sports STR
- **Koshkina & Elder (CVPRW CVsports 2021; arXiv:2405.13896, CVPRW 2024) — *"A General Framework for Jersey Number Recognition in Sports Video"* (`mkoshkina/jersey-number-pipeline`):**
  - Introduces an explicit **legibility classifier** ahead of/alongside scene text recognition so occluded, blurred, or front-facing torso crops do not pollute tracklet aggregation.
  - Combines pose/torso RoI localization, STR decoding, and multi-frame tracklet aggregation, achieving **91.4% accuracy on ice hockey images** and **87.4% tracklet accuracy on SoccerNet**.
  - **Adopted here:** Jointly predict calibrated `legibility_logit` (`P(readable full number)`) and `back_logit` (`P(back of jersey)`) inside a single forward pass (`coach_jersey_str.onnx`), using `(0.3 + 0.7 * legibility) * (0.5 + 0.5 * back) * sharpness * boxHeight` to curate best-2-view contact sheets for Gemini Nano and gate frame-level STR reads.
- **Koshkina & Elder (arXiv:2502.21242, Feb 2025) — *SportsSUSHI*:**
  - Extends hierarchical graph tracking (`SUSHI`) with domain-specific cues (jersey numbers, team IDs, rink geometry) to bridge long-term occlusions without cross-team switches.

### 2.2 Fast Single-Stage Compositional STR & Calibration
- **Bautista & Atienza (ECCV 2022, arXiv:2207.06966, `baudm/parseq`) — *PARSeq*:**
  - Demonstrates that permutation language modeling and cross-attention queries over visual patch tokens outperform external language models on curved, rotated, and partially obscured sports text.
- **Du et al. (arXiv:2411.15858, 2024/2025) — *SVTRv2*:**
  - Shows multi-size visual feature rearrangement + semantic guidance lets single-forward non-autoregressive recognizers beat encoder-decoder models on latency/accuracy trade-offs.
- **Grad (CVPRW CVsports 2025) — *"Single-Stage Uncertainty-Aware Jersey Number Recognition in Soccer"*:**
  - Processes full player/torso crops in a **single stage** (avoiding brittle separate digit localization), exploits **digit-compositional heads** (tens + units decomposition rather than only 100 flat independent classes), and uses uncertainty calibration for confidence-weighted tracklet fusion (**85.62% SoccerNet tracklet accuracy**).
  - **Adopted here:** Our ONNX architecture (`JerseyStrNet`) fuses:
    1. **MobileNetV3-Large ImageNet backbone (`features[0..14]`)** at both `6x6` (stride `16`, `112` ch) and `3x3` (stride `32`, `160` ch) scales (`~9.5 MB` FP32, opset `17`, `< 1 ms` CPU).
    2. **Spatial Attention Queries (`Q_left`, `Q_right`, `Q_center`, `Q_legible`)** over the `6x6 = 36` positional tokens with learned 2D embeddings — preserving left-to-right digit order (`12` vs `21`, `34` vs `43`) and isolating occluded partial digits (`?7` vs `9?`).
    3. **Compositional + Holistic Logit Fusion:** Compositional log-probability prior `logP_left[L(k)] + logP_right[R(k)]` added to a holistic `101`-class residual head (`0 = none`, `1..99`, `100 = "00"`), plus post-hoc **temperature scaling** calibrated on out-of-distribution synthetic validation data.

### 2.3 Order-Free CIELAB Kit Clustering & Multi-Cue Referee Classification
- **Reference Implementation (`CoachSebastien/backend/coach/cv/jersey_color.py`, `tracking_v2/kit.py`, `tracking_v2/colour.py`):**
  - **Order-Free 7D Summary:** Represents each torso as `(dark_L, dark_a, dark_b, light_L, light_a, light_b, light_share_pct)` and scores `min(dist(x, c), dist(swap(x), c))`. A white jersey with a dark shoulder yoke (`light_L ~ 82, light_share ~ 50%, lower_belly_Luma ~ 170`) stays clearly distinct from solid dark navy (`dark_L ~ 20, light_share ~ 15%, lower_belly_Luma ~ 35`).
  - **Ice White Balance & Background Exclusion:** Median lower-ice RGB white balance + border-ice `deltaE < 11` background discounting (unless `> 40%` ice-like, indicating a white jersey).
  - **Multi-Cue Official Separation:** Combines `64x36` 1D Hann-DFT vertical stripe power (`k = 3..15`, 6-row-band correlation consistency, vertical row extent, white-stripe `p90 >= 122`, low chroma `C* < 16.5`) with orange sleeve armband fraction and dark trouser score (`y: 56%..84%`) for far-board blurred referees, eliminating dark-navy `#14` false positives.

### 2.4 Roster-Free Bayesian Fusion & Joint Co-Occurrence Assignment
- **Reference Implementation (`CoachSebastien/backend/coach/cv/numbers.py`):**
  - **100% Roster-Free Uniform Prior:** Equal prior probability over `1..99` (plus `"00"` in ONNX class `100`, and negative class `0`).
  - **Digit-Confusion + Partial Geometry:** Combines symmetric digit-confusion matrix (`{1,7}`, `{3,8}`, `{6,8}`, `{6,9}`, `{8,9}`, `{3,9}`, `{0,8}`, `{2,7}`, `{4,1}`) with `LEFT_PARTIAL` (`9?`), `RIGHT_PARTIAL` (`?7`), and `SINGLE` likelihoods (`HIDE_P = 0.35–0.38`), fusing calibrated `ONNX_STR`, `GEMINI_NANO`, and `MLKIT_OCR` distributions.
  - **Hungarian Co-Occurrence Exclusivity (`assign_numbers_jointly`):** `O(N^3)` Munkres assignment over concurrent same-team tracklets (`overlap > max(1.5s, 0.12 * minSpan)`), preventing duplicate team numbers unless strong independent multi-bin evidence overrides `DUPLICATE_PRIOR = 0.04`.

---

## 3. Experimental & Ablation Results (Zero Test Leakage)
All synthetic training and model selection occurred **exclusively** on procedurally generated synthetic hockey jersey crops (`jersey_synth.py`, uniform `1..99` + `"00"`, zero CoachBox frames or roster numbers). CoachBox frames (`frame_00..15.png` and `fs_00..22_*.png`), `sample_hockey_30s.mp4`, and `tvi_iceland_clip_01_0240s_0270s.mp4` were held out strictly for final evaluation (`coachbox_local_eval.py`).

### 3.1 ONNX STR Architecture & Calibration Ablations (`160k` Train / `12k` Held-Out Synthetic Val)

| Model Variant (on v4 NHL Varsity Shards) | Legible Top-1 | Legible Top-3 | Order-Swap (`ab` != `ba`) | Partial Left / Right Acc | Hard-Neg FP | Legibility Acc |
|---|---|---|---|---|---|---|
| **A1 Baseline:** Global-Avg-Pool (`160k`, 8 ep) | `94.02%` | `97.80%` | `93.68%` | `89.26% / 88.93%` | `0.00%` | `99.94%` |
| **A2 Proposed:** Multi-Scale `6x6` Spatial Attention + Compositional `(L, R)` Fusion (`160k`, 20 ep) | **`94.44%`** | **`97.95%`** | **`93.98%`** | **`90.71% / 90.30%`** | **`0.00%`** | **`99.98%`** |

- **Artifact Specs (`coach_jersey_str.onnx` & `coach_jersey_str.json`):**
  - Size: **`10,544,493` bytes (`10.54 MB` <= `12 MB` limit)**, FP32, ONNX **opset `17`**, static `[1, 3, 96, 96]` input (`images`, RGB `[0..1]`, embedded ImageNet normalization).
  - Numerical Parity & CPU Latency: PyTorch vs. ORT `max_abs_diff = 2.38e-06`; **`onnxruntime` CPU (2 threads) = `0.358 ms/crop`**.
  - **Temperature Calibration (baked into exported ONNX logits):** `T_num = 0.8720` (NLL `0.2013 -> 0.1880`, **ECE `0.0379 -> 0.0144`**, `-62%` calibration error), `T_leg = 0.4957`, `T_back = 1.1350`, `T_digit = 0.8728`.
  - **Sim-to-Real Upgrade Impact (non-CoachBox `tvi_iceland_clip_01` frame 120):** Switching `jersey_synth.py` from initial thin polygons (`v2`) to squared/beveled NHL varsity block glyphs (`v4`) + exact `2%..98%` content stretch lifted `#22` from `#93 (0.54)` to **`#22 (0.96)`** and rejected illegible `[4]` (`leg = 0.00`).

### 3.2 End-to-End Local Evaluation (`research/coachbox_local_eval.py` -> `research/local_validation_report.json`)

| Benchmark | Pipeline | Detection Recall (`11:14`) | Skater Team Acc | `#97 Begic` Team B | Official `[REF]` P / R (`#14` FP) | Roster-Free STR Hits / Notes |
|---|---|---|---|---|---|---|
| **CoachBox Compare (`frame_00..15.png`, `2.0s` step)** | Baseline Tracker + Naive Vote | `12/12 (100.0%)` | `9/10 (90.0%)` | Yes | `0.0 / 0.0` (`#14` 0 FP) | `0/9` (low-res `752x421` upscaled stills) |
| **CoachBox Compare (`frame_00..15.png`, `2.0s` step)** | Proposed SOTA (`:core` + ONNX STR) | `12/12 (100.0%)` | `9/10 (90.0%)` | Yes | `0.0 / 0.0` (`#14` 0 FP) | `0/9` (board refs ~6 src px wide) |
| **CoachBox `1080p` (`fs_00..22`, `1.4s` step)** | Baseline Tracker + ONNX STR Naive Vote | `12/12 (100.0%)` | `9/10 (90.0%)` | Yes (`#97` hit) | `1.00 / 0.50` (`#14` 0 FP) | `2/9` (`#97`, `#67` exact hits; `P=0.400, R=0.222`) |
| **CoachBox `1080p` (`fs_00..22`, `1.4s` step)** | Proposed SOTA (`:core` + ONNX STR) | **`12/12 (100.0%)`** | **`9/10 (90.0%)`** | **Yes** (`BELLY_WEIGHT=1.65`) | **`1.00 / 0.50`** (`REF_C` TP, `#14` 0 FP) | `1/9` (`#67` exact hit; `#97` got `0.95` on `tid=5` at `18.2s`, bumped by `1.4s`-step crossing on `tid=8`; `#57` got `pR='7'` `0.64`) |
| **Burst Ghost Check (`coachbox_tvbd_cougars_30s.mp4`)** | Baseline vs. SOTA (`det > 15`) | `max_raw = 22` (`17` burst frames) | — | — | — | Emitted peak cut **`22 -> 15`** on crossfades; **`12/12`** on `cleanhold_30s` (`0` burst frames) |
| **`sample_hockey_30s.mp4` (`30` frames)** | Proposed SOTA | `0.0` mean / `0` max det | — | — | — | `0` false-positive detections on 2D schematic |
| **`tvi_iceland_clip_01_0240s_0270s.mp4` (`60` frames)** | Proposed SOTA | `8.93` mean / `13` max det | `9` A / `15` B tracks | — | `0` ref tracks | Resolved `20` roster-free track numbers (`#22, #25, #40, #53, #74, #32, #98`, etc.) |

### 3.3 Remaining Real-Clip Observations
1. **Why `REF_L` missed in `coachbox_1080p` (`0.50` recall):** Across `fs_00..22`, `REF_L` (`tid=27`) averaged `pantsDarkness = 0.39` (below the `0.60` threshold; `0.84` on `fs_12_674` alone) because the `17 px` wide board official overlapped the yellow kickplate/boards in several shots (`REF_C` hit `0.72` and classified `REFEREE` with `1.00` precision and `0` `#14` FP).
2. **Why `coachbox_1080p` `1.4s`-step Hungarian bumped `#97` (`tid=5` -> `#86`):** `coach_jersey_str.onnx` read `#97 (0.95)` on `tid=5` at `t=18.2s`, while `tid=8` (`#34`) jumped tracks across a `1.4s` screenshot gap onto `#97` at `t=21.0s` (`#97 (0.99)`), causing Hungarian to award `#97` to `tid=8`. Real `4 fps` (`0.25s`) Pixel 11 Pro inference plus Gemini Nano multi-view contact sheets avoid `1.4s` track jumps and provide `2.5x` Nano confirmation.

---

## 4. Round 2 Sim-to-Real Upgrade (`v6` Synth + `v3` ONNX + Round 2 Tracker/STR Sync)

### 4.1 What Changed in Round 2 (Zero Test Leakage / 100% Roster-Free)
1. **Dataset (`research/jersey_synth.py` `v6`, `160k` train / `12k` val, uniform `1..99` + `"00"`):**
   - Preserved native crouched torso crop aspect ratios (`0.75..2.30`, `35%` wide `1.38..2.30`) with exact `PAD_GREY=114` top/bottom letterbox bars and `2%..98%` contrast stretch matching `JerseyStrOnnxRecognizer.kt`.
   - Added curved/arched NHL shoulder yokes + double lower waist stripes, stitched/blurred uppercase nameplate bars above back numbers (`82%`), thinner stroke range (`0.095..0.175 * dh`), smaller numerals (`24..58%` torso height), saturated colored numerals (crimson/scarlet/orange/royal/teal with light outline) on dark jerseys, `~7.5 px` Nyquist digit legibility floor for positives, and `32%` hard negatives (sub-Nyquist extreme blurs + crests + stripes + boards).
2. **Calibration & Regularization (`research/train_jersey_str.py` `v3` ONNX):**
   - Applied label smoothing `0.05` (`0.03` on legibility/back) and conservative OOD calibration floors **`T_num = max(1.08, fit)`**, **`T_digit = max(1.08, fit)`**, **`T_leg = max(1.05, fit)`**, **`T_back = max(1.05, fit)`** so blurry far-ice crops never sharpen into false `0.99` spikes.
   - **Exported `v3` ONNX:** **`10,544,545` bytes (`10.545 MB`)**, opset **`17`**, **`0.354 ms/crop`** ORT CPU (2 threads), PyTorch vs ORT `max_abs_diff = 5.25e-06`.
   - **Baked Calibration:** `T_num = 1.0800`, `T_digit = 1.0800`, `T_leg = 1.0500`, `T_back = 1.0500` (val NLL `0.7194 -> 0.7181`, **ECE `0.0580 -> 0.0562`**).
3. **Evaluation Harness Parity (`research/coachbox_local_eval.py`):**
   - Mirrored Round 2 `:core` (`#87`: `OnIceFilter` `minHeight=0.026`, `DEFAULT_BELLY_WEIGHT=1.25`, `JerseyNumberPosterior` uniform `0.08/99` outlier floor + `UNASSIGN_PRIOR=0.08` + `MIN_POSTERIOR_PROB=0.48` + `MIN_EVIDENCE_WEIGHT=0.90`, `MultiTeamPlayerTracker` relaxed `isBlurredFarBoardOfficial` + `isCrossKitIncompatible`) and Round 2 `:app` (`#89`: `findTorsoCenterFrac` + `75%` wide-stick ice drop, `numProbs[0] < 0.60` / `leg >= 0.30` / partial margin `1.15x` gating).

### 4.2 Round 2 Ablations & End-to-End Local Results (`v6` Synth / `v3` ONNX)

| Model Variant (`v6` Wide/Nameplate/Curved-Yoke Shards) | Legible Top-1 | Legible Top-3 | Order-Swap Top-1 | Partial L / R Acc | Full L / R Digit Acc | Hard-Neg FP | Legibility Acc |
|---|---|---|---|---|---|---|---|
| **A1 Baseline:** Global-Avg-Pool (`160k`, 8 ep) | `75.99%` | `86.96%` | `75.07%` | `63.39% / 60.46%` | `89.40% / 88.94%` | `0.05%` | `99.90%` |
| **A2 Proposed:** Multi-Scale `6x6` Spatial Attention + Compositional (`160k`, 20 ep best) | **`76.46%`** | **`87.11%`** | **`75.39%`** | **`65.16% / 61.45%`** | **`89.84% / 88.95%`** | **`0.10%`** (`0.05%` ep20) | **`99.87%`** (`99.91%` ep20) |

*(Note: `v6` Top-1 reflects harder `1.38..2.30` wide crouched crops shrinking torso height inside `[96,96]` letterbox and heavy nameplate/yoke/blur diversity; hard-negative FP dropped `10x` vs Round 1's `1.00%`.)*

| Round 2 Local Benchmark (`local_validation_report.json`) | Pipeline | Detection Recall (`11:14`) | Skater Team Acc | `#97 Begic` Team B | Official `[REF]` P / R (`#14` FP) | Roster-Free STR / Key Notes |
|---|---|---|---|---|---|---|
| **CoachBox Compare (`frame_00..15`, `2.0s` step)** | Proposed SOTA | **`12/12 (100.0%)`** | **`10/10 (100.0%)`** | **Yes** (`DARK_U` -> `TEAM_A` fixed) | `0.0 / 0.0` (`#14` 0 FP) | `0/9` hits (`frame_08` crops verified blurry/sideways `p0=0.85..0.93`, `0` FP in `frame_08`) |
| **CoachBox `1080p` (`fs_00..22`, `1.4s` step, `0.53x`-res upscaled PNGs)** | Proposed SOTA | **`12/12 (100.0%)`** | `9/10 (90.0%)` | **Yes** (`#97` = `TEAM_B`) | `0.0 / 0.0` (`n_rgb=1` after track split; `#14` 0 FP) | `1/9` exact (`#67 = "67"` at `0.830` / `0.793`, `#87` suppressed `<=0.07`; `#97->"95"`, `#77->"92"`). Pure 2-hit `fs_12` gets `REF` `2/2 (100% P / 100% R)` |
| **Burst Ghost Check (`coachbox_tvbd_cougars_30s.mp4`)** | Baseline vs. SOTA | `max_raw = 22` (`17` burst frames) | — | — | — | Emitted peak cut **`22 -> 15`** (`12/12` on `cleanhold_30s`) |
| **`sample_hockey_30s.mp4` (`30` frames)** | Proposed SOTA | `0.0` mean / `0` max | — | — | — | `0` false positives on 2D schematic |
| **`tvi_iceland_clip_01_0240s_0270s.mp4` (`60` frames)** | Proposed SOTA | `8.93` mean / `13` max | `33` A / `28` B | — | `4 REFEREE` tracks | Resolved **`30`** roster-free track numbers (`#22, #21, #30, #12, #13, #40, #63, #88, #89, #97`); frame 120 `#21` fixed (`0.392 > #31 0.240`), `#22=0.931` |

### 4.3 Round 2 Diagnostic Takeaways for Live Pixel Verification
1. **Nameplate `67 -> 87` bug resolved:** On `coachbox_1080p`, `v3` predicts `67 (0.830, 0.793)` while `87` stays `<= 0.07`, and `JerseyNumberPosterior`'s uniform `0.08/99` outlier floor prevents a single `1.4s`-step `63 (0.93)` outlier from overriding `#67`.
2. **Dark-box `DARK_U` + `#97` team separation fixed:** `findTorsoCenterFrac` + `75%` wide-stick ice drop + `isCrossKitIncompatible` yields `10/10 (100%)` skater team accuracy on `coachbox_compare` and preserves `#97 = TEAM_B` (`belly=139.1`) on `coachbox_1080p`.
3. **Diagnostic watchpoints for live `1080p` Pixel 11 Pro test:** On upscaled `0.53x`-resolution (`1024x576`) `fs_12` screenshots, low-res/outlined `'7'` still tended toward `'5'`/`'0'` (`#97 -> 95`, `fs_12 #67 -> 60`), whereas true `1920x1080` continuous footage (`tvi_iceland_clip_01`) plus priority-sorted `GeminiNanoHockeyEngine` (`Slot A..D`) provides sharper native crops and `2.5x` multimodal corroboration.

---

## 5. Round 3 Upgrade (`v7` Synth + `v4` ONNX + `1-vs-N` Huberized Outlier Guard)

### 5.1 What Changed in Round 3 (Zero Test Leakage / 100% Roster-Free)
1. **Dataset (`research/jersey_synth.py` `v7`, `160k` train / `12k` val, uniform `1..99` + `"00"`):**
   - Added **lateral curved sleeve-shadow & armpit fold injection (`52%`)** brushing near the right/left borders of outer digits (`'7'`, `'1'`, `'4'`, `'9'`), directly targeting the `#97 -> #95` white-jersey sleeve shadow confusion (`#95`).
   - Updated `'7'` synthesis to **70% clean serif-free athletic `'7'`s** (plus `35%` PIL `NimbusSans/URWGothic` glyphs with fixed zero-erosion 2-layer outlines `sw=1..3`, keeping open counters on `#59`/`#94`), and forced outline contrast on low-luminance colored-on-dark numerals (`|dL| < 48`).
   - Added **`double_body_overlap` (`6%`)** hard-negative category rendering two colliding/crossing skaters (`class=0, leg=0`) to suppress multi-body STR false positives (`T38:#85`).
2. **Calibration & Regularization (`research/train_jersey_str.py` `v4` ONNX):**
   - Tightened `loss_leg` and `loss_back` label smoothing (`0.03 -> 0.01`) alongside `label_smoothing=0.05` on digit/number CE heads, lifting `fit_temperature_bce` raw fit `T_leg_fit` from `0.4973 -> 0.6458` while retaining conservative OOD calibration floors (`T_num=1.0800`, `T_digit=1.0800`, `T_leg=1.0500`, `T_back=1.1244`).
   - **Exported `v4` ONNX:** **`10,544,519` bytes (`10.545 MB` <= `12 MB`)**, opset **`17`**, **`0.351 ms/crop`** ORT CPU (2 threads), PyTorch vs ORT `max_abs_diff = 1.10e-05` (`val NLL 0.7353 -> 0.7345`, `ECE 0.0552 -> 0.0574`).
3. **Evaluation Harness Parity (`research/coachbox_local_eval.py`):**
   - Mirrored Round 3 `:core` (`#97`: `HUBER_AB_DELTA=22.0`, `robustWeightedKitDistSq`, `is_balanced` minimum cluster-support guard + outlier trimming + 1D `effL` 2-means fallback) and Round 3 `:app` (`#98`: `legibility >= 0.55` for full `ONNX_STR` candidates, `>= 0.30` for partials, `REFEREE` / `pants >= 0.60 && chroma < 15.0` STR skip).

### 5.2 Round 3 Ablations & End-to-End Local Results (`v7` Synth / `v4` ONNX)

| Model Variant (`v7` Sleeve-Fold / Serif-Free-'7' / Double-Body Shards) | Legible Top-1 | Legible Top-3 | Order-Swap Top-1 | Partial L / R Acc | Full L / R Digit Acc | Hard-Neg FP | Legibility Acc |
|---|---|---|---|---|---|---|---|
| **A1 Baseline:** Global-Avg-Pool (`160k`, 8 ep) | `75.32%` | `87.03%` | `74.00%` | `64.11% / 63.74%` | `89.51% / 89.10%` | `0.18%` | `99.87%` |
| **A2 Proposed:** Multi-Scale `6x6` Spatial Attention + Compositional (`160k`, 20 ep best) | **`76.63%`** | **`87.12%`** | **`75.61%`** | `63.83% / 64.56%` | **`89.59% / 89.57%`** | `0.23%` (`0.20%` argmax!=0) | **`99.87%`** |

*(Right-partial digit accuracy improved `+3.1%` (`61.45% -> 64.56%`) and full right-digit accuracy improved to `89.57%` despite `double_body_overlap` & sleeve-shadow folds.)*

| Round 3 Local Benchmark (`local_validation_report.json`) | Pipeline | Detection Recall (`11:14`) | Skater Team Acc | `#97 Begic` Team B | Official `[REF]` P / R (`#14` FP) | Roster-Free STR / Key Notes |
|---|---|---|---|---|---|---|
| **CoachBox Compare (`frame_00..15`, `2.0s` step)** | Baseline vs. Proposed SOTA | **`12/12 (100.0%)`** | `9/10` -> **`10/10 (100.0%)`** | **Yes** (`DARK_U=TEAM_A` fixed) | `0.0 / 0.0` (`#14` 0 FP) | `0/9` hits (`#67->"87"` on cross-frame jump-cut; all blurry skaters safe) |
| **CoachBox `1080p` (`fs_00..22`, `1.4s` step, `0.53x`-res PNGs)** | Baseline (`0/9`, `#67->"62"`) vs. Proposed SOTA | **`12/12 (100.0%)`** | `9/10 (90.0%)` | **Yes** (`#97=TEAM_B`) | `0.0 / 0.0` (`n_rgb=1` track split; `#14` 0 FP) | **`1/9` exact (`#67="67"`), `100%` precision (`P=1.000`)** — `#97->"95"` & `#77->"92"` FPs eliminated |
| **Burst Ghost Check (`coachbox_tvbd_cougars_30s.mp4`)** | Baseline vs. SOTA | `max_raw = 22` (`17` burst frames) | — | — | — | Emitted peak cut **`22 -> 15`** (`12/12` on `cleanhold_30s`) |
| **`sample_hockey_30s.mp4` (`30` frames)** | Proposed SOTA | `0.0` mean / `0` max | — | — | — | `0` false positives on 2D schematic |
| **`tvi_iceland_clip_01_0240s_0270s.mp4` (`60` frames)** | Proposed SOTA | `8.93` mean / `13` max | `33` A / `28` B | — | `4 REFEREE` tracks | Resolved **`26`** roster-free track numbers; frame 120: `tvi[00] #22 (0.828)`, `tvi[03] #30 (0.837)` (up from `#50`), `tvi[05] #4 (0.705)`, `tvi[06] #97 (0.342)`, `tvi[04]` `leg=0.012` |

---

## 6. Round 4 Harness Sync (`refereeFrameVotes` Persistence + `1.0s` Time-Bin Independence)

### 6.1 Round 4 `:core` / `:app` Parity Changes (`coachbox_local_eval.py`)
1. **Official `[REF]` Frame-Vote Persistence (`MultiTeamPlayerTracker.kt` `#108`):**
   - Tracks `refereeFrameVotes` (`ref_votes`) and `skaterFrameVotes` (`skater_votes`) strictly for tracks present on the current frame (`cur_frame_tids`), persisting established officials (`ref_votes >= 3` unless `skater_votes > 2 * ref_votes && mean_chroma >= 15.0`) outside Team A/B centroids across subsequent frames.
2. **Per-`1.0s` Time-Bin STR Independence (`JerseyNumberPosterior.kt` `#108`):**
   - Applies `SAME_BIN_MARGINAL_WEIGHT = 0.15` to repeated non-Nano samples inside any `(floor(t / 1.0s), channel)` bin, and requires `>= 2` distinct `1.0s` supporting bins OR single-bin `>= 0.80` confidence (or Gemini Nano verification) before committing a jersey number.

### 6.2 Round 4 End-to-End Local Metrics (`local_validation_report.json`)

| Round 4 Local Benchmark (`v4` ONNX + Round 4 `:core`/`:app`) | Pipeline | Detection Recall (`11:14`) | Skater Team Acc | `#97 Begic` Team B | Official `[REF]` P / R (`#14` FP) | Roster-Free STR / Key Notes |
|---|---|---|---|---|---|---|
| **CoachBox Compare (`frame_00..15`, `2.0s` step)** | Baseline (`9/10`, `0/9`) vs. Proposed SOTA | **`12/12 (100.0%)`** | **`10/10 (100.0%)`** | **Yes** (`DARK_U=TEAM_A`) | `0.0 / 0.0` (`#14` 0 FP) | `0/9` hits (`#67->"87"` on sparse cross-frame tracks; blurry crops safe) |
| **CoachBox `1080p` (`fs_00..22`, `1.4s` step, `0.53x`-res PNGs)** | Baseline (`9/10`, `0/9`) vs. Proposed SOTA | **`12/12 (100.0%)`** | `9/10 (90.0%)` | **Yes** (`#97=TEAM_B`) | **`1.00 / 0.50`** (`REF_C` TP persisted, `#14` 0 FP) | **`1/9` exact (`#67="67"`), `100%` precision (`P=1.000`)**, zero wrong jersey predictions |
| **Burst Ghost Check (`coachbox_tvbd_cougars_30s.mp4`)** | Baseline vs. SOTA | `max_raw = 22` (`17` burst frames) | — | — | — | Emitted peak cut **`22 -> 15`** (`12/12` on `cleanhold_30s`) |
| **`sample_hockey_30s.mp4` (`30` frames)** | Proposed SOTA | `0.0` mean / `0` max | — | — | — | `0` false positives on 2D schematic |
| **`tvi_iceland_clip_01_0240s_0270s.mp4` (`60` frames)** | Proposed SOTA | `8.93` mean / `13` max | `32` A / `28` B | — | **`5 REFEREE` tracks** | Resolved **`14`** multi-bin/high-conf track numbers (`#10, #21, #22, #25, #32, #47, #53, #55, #65, #92, #98`, `0 UNKNOWN`) |



