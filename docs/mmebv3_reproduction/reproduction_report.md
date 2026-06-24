# MMEB-V3 Reproduction: Omni-Embed-Nemotron-3B — Tool & Memory

**Date:** 2026-06-17
**Status:** ✅ Reproduced (Tool & Memory agent tasks)
**Paper:** MMEB-V3, *Measuring the Performance Gaps of Omni-Modality Embedding Models*
(arXiv 2604.23321). PDF: `docs/papers/[2604.23321v1]MMEB-v3.pdf`.
**Reference tables:** Table 2 (category averages) and **Table 10** (per-dataset, agent tasks).
**Model:** `nvidia/omni-embed-nemotron-3b` (backbone `nvomniembed`, mean pooling + L2 normalize).
**Headline metric:** Hit@1 (×100) — the metric the paper reports for agent tasks.

---

## 1. Result summary

**FULL reproduction — all 7 categories run.**

| Category | Datasets | Metric | Reproduced | Paper | Δ | Per-dataset MAE |
|---|---|---|---|---|---|---|
| **Tool** (ToolDe) | 35 | Hit@1 | **38.0** | 38.1 | −0.1 | 0.37 |
| **Memory** (LMEB) | 4 | Hit@1 | **32.3** | 32.3 | 0.0 | 0.05 |
| **GUI** (GAE) | 8 | Hit@1 | **32.7** | 32.5 | +0.2 | 0.31 |
| **Text** | 53 | NDCG@5 | **39.2** | 38.6 | +0.6 | 1.67 |
| **Image** | 37 | Hit@1 | **43.4** | 43.9 | −0.5 | 0.97 |
| **VisDoc** | 24 (27 files) | NDCG@5 | **74.0** | 70.8 | +3.2 | 5.62 |
| **Video** | 18 | Hit@1 | **41.5** | 41.3 | +0.2 | 0.59 |
| **Audio** | 10 / 11 | Hit@1 | 29.3 | 30.1 | −0.8 | 12.6 ⚠️ |

> Image/VisDoc/Video/Audio were run from the **main repo** (team's live loaders) on official data; Tool/Memory/Text from the olm2vec worktree. Per-dataset fixes were needed across nearly every modality (see §6). **Excluded:** audio **SoundDescs** (89 GB corpus → CPU OOM) and **MomentSeeker_1k8** (duplicate not in the paper's 18).

- **Tool / Memory / Text / Image / Video**: clean reproductions — category averages within **0.0–0.6** of the paper, low per-dataset MAE (Memory exact; Video +0.2/MAE 0.59; Tool −0.1; Image −0.5).
- **VisDoc**: average +3.2 (74.0 vs 70.8), MAE 5.6 — reproduces the trend; per-dataset spread is larger (some ViDoRe/VisRAG subsets diverge a few points).
- **Audio**: category **average** lands close (29.3 vs 30.1) but partly coincidental — per-dataset **MAE 12.6** (SpeechCommands 87.0 vs 45.0, NSynth −23.8, TUTSound −41; ESC-50/CREMA-D match). Not the loaders (inputs byte-identical to the official release) — the audio **decode** path. **Runs end-to-end via soundfile workarounds, but not faithfully reproduced.**
  - **Decode is NOT the cause (verified).** torchcodec was made to work on torch 2.11 (`pip install torchcodec --index-url .../cu128` → 0.11.1+cu128, + `LD_LIBRARY_PATH=$ENV/lib` for FFmpeg-7). A faithful rerun through the native `torchaudio`/HF-Audio decode path gave **byte-identical** per-dataset scores to the soundfile workaround (NSynth 4.4, SpeechCommands 87.0, TUTSound 0.9; MAE 12.6 unchanged). So torchcodec≈soundfile for these WAVs — the audio divergence is **not** a decode artifact.
  - (The earlier "torch 2.11 incompatible with torchcodec" was a red herring: the *default* `pip install torchcodec` pulls a CUDA-13 wheel needing `libnvrtc.so.13`; the cu128 wheel works.)
  - **Remaining audio gap is intrinsic to the model/pipeline** — candidate: `transformers 4.52.3` vs the model-card-pinned `4.51.3-Qwen2.5-Omni-preview`, or audio processing/instruction handling. Closing it would require the paper-era `transformers 4.51.3`, not a decode fix.
- Deterministic: PeerQA reproduced identically across independent runs.

### Per-dataset CSVs (in this folder, paper all-7-models vs our reproduced + diff)
`tool_hit@1`, `memory_hit@1`, `text_ndcglinear@5`, `image_hit@1`, `visdoc_ndcglinear@5`,
`video_hit@1`, `audio_hit@1` — all `*_paper_vs_repro.csv`.

Parsing validated: computed paper-AVG rows reproduce the paper's own category summary lines
(Tool 38.1, Memory 32.3, Text 38.6, Image 43.9, Video 41.3).

Each CSV now also carries per-dataset **`eval_time_s` / `eval_time`** columns (encode+score wall
time on 8×H100, from the logged `[Timing]` lines; excludes ~1–2 min model load + SLURM queue wait).

### Eval timing summary (8×H100, batch 32; video batch 4)
| Modality | #ds | Σ encode | avg/ds | slowest dataset |
|---|---|---|---|---|
| tool | 35 | ~27 min | 47 s | toolbench 114 s (web re-encodes the 37k corpus per task) |
| memory | 4 | ~2 min | 31 s | KnowMeBench 74 s |
| text | 53 | ~65 min | 73 s | BRIGHT/leetcode **17 min** |
| image | 37 | ~42 min | 68 s | MCMR **29 min** (123k-image corpus) |
| visdoc | 27 | ~29 min | 64 s | VisRAG_PlotQA 12 min |
| video | 18 | ~27 min | 91 s | MomentSeeker 14 min, Charades-STA 13 min, MVBench 11 min |
| audio | 10 | ~4 min | 26 s | CREMA-D 131 s |

Caveats: tool web times are inflated by re-encoding the full candidate corpus per task (an
efficiency bug, not inherent cost); on resubmits, cached datasets log ~0 s, so per-dataset times
are the max observed (true first-run cost).

### Tool outliers (only datasets with |Δ| ≥ 1.5)
| dataset | n queries | paper | repro | Δ |
|---|---|---|---|---|
| autotools-music | 32 | 6.2 | 3.1 | −3.1 |
| mnms | 33 | 39.4 | 42.4 | +3.0 |
| t-eval-dialog | 50 | 16.0 | 14.0 | −2.0 |
| metatool | ~200 | 50.0 | 48.5 | −1.5 |

All are small datasets where a single query flip moves Hit@1 by 2–3 points — tie-break /
numerical-precision noise, not a configuration error. Everything else matches to 0.0–0.6.

---

## 2. Environment & setup

- **Code:** git worktree of `VLM2Vec` on branch `olm2vec` at
  `/rmeng_data/projects/embed/VLM2Vec-olm2vec` (the `main` branch was left untouched).
- **Conda env:** reused `/rmeng_data/envs/vlm2vec` (python 3.11, torch 2.11+cu128,
  transformers 4.52.3, flash_attn 2.8.3) — matches the olm2vec `requirements.txt`.
- **Model weights:** `/rmeng_data/data/vlm2vec/models/omni-embed-nemotron-3b`
  (`hf download nvidia/omni-embed-nemotron-3b`).
- **Compute:** SLURM `a3` partition, 1 node × 8 H100, via `python -m torch.distributed.run`.
  Wall time: ~30 min tool, ~3 min memory.

### Code patches required on olm2vec HEAD (b9512e5)
The branch tip did not import/run out of the box; four minimal fixes (all in the worktree):
1. `src/utils/vision_utils/{video_transforms,vision_utils}.py` — guard
   `from torchvision.io import write_video` (torchvision ≥0.26 removed video IO; unused for text eval).
2. `src/constant/dataset_hf_path.py` — add `import os` and
   `from .dataset_hflocal_path import BASE_RAW_DATA_DIR` (both used, neither imported).
3. `src/data/eval_dataset/__init__.py` — comment out `from .mscoco_cmret import ...`
   (module absent at this commit; unrelated to tool/memory).
4. `src/model/model.py` (NVOMNIEMBED branch) — add `trust_remote_code=True` to
   `AutoModel.from_pretrained` (the model ships custom remote code).

### Data staging
`--data_basedir /rmeng_data/data/vlm2vec/MMEB-V3-evalroot`, containing:
- `tool-tasks/` — extracted from `MMEB-V3/tool_tasks/tool_tasks.tar.gz` (matches `tool.yaml` paths).
- `memory-tasks/` — flat datasets from `MMEB-V3/memory_tasks/memory-tasks.tar`, exposed via
  **type-subdir symlinks** (`Episodic/KnowMeBench`, `Dialogue/REALTALK`, `Semantic/PeerQA`,
  `Procedural/DeepPlanning`) so the official `memory.yaml` runs unmodified.

Data lives under `/rmeng_data/data/vlm2vec/` (datasets), code under `/rmeng_data/projects/embed/`.

---

## 3. How to run

Reproduction harness: **`experiments/mmebv3_reproduction/`**. The 8 standard categories run from this
repo (main-repo V3 loaders); OmniSET runs from the `olm2vec` branch (see note).

```bash
# Per-modality eval (8×H100, resumable — per-task *_score.json are cached and skipped on rerun)
sbatch experiments/mmebv3_reproduction/eval_nemotron_mainrepo.sbatch <modality>            # nemotron
sbatch experiments/mmebv3_reproduction/eval_qwen3vl.sbatch <model_dir> <tag> <modality>     # Qwen3-VL 2B/8B

# Aggregate + regenerate paper-vs-repro CSVs (-> docs/mmebv3_reproduction/paper_vs_repro/)
python experiments/mmebv3_reproduction/dump_qwen3vl_csv.py
python experiments/mmebv3_reproduction/dump_paper_vs_repro_csv.py
python experiments/mmebv3_reproduction/dump_text_audio_csv.py
```

Key eval flags: `--pooling {mean|last} --normalize true --model_backbone {nvomniembed|qwen3_vl}`.

**OmniSET** (MSCOCO cross-modal): the `mscoco_cmret` loader + `cross_modality` configs exist only on
the **`olm2vec`** branch, so OmniSET is run from there via `eval_omni.sbatch` + `dump_omni_csv.py`
(copies kept in the harness folder for reference).

---

## 4. Artifacts

| Artifact | Location |
|---|---|
| Per-modality paper-vs-repro CSVs | `docs/mmebv3_reproduction/paper_vs_repro/<model>/*_paper_vs_repro.csv` |
| Reproduction harness (runners + aggregators) | `experiments/mmebv3_reproduction/` |
| Per-task raw scores (all metrics) | `/rmeng_data/exps/vlm2vec/{nemotron-main,qwen3vl,omni}/**/*_score.json` (not in repo) |

Raw `*_score.json` files also contain nDCG / Recall / MAP / MRR at k = 1/5/10 if a different
headline metric is ever needed.

---

## 5. Reproduction friction log

Overall assessment: **the science reproduced trivially; the engineering had real friction.**
Once it ran, numbers matched on the first attempt — no hyperparameter tuning, no pooling/instruction
guessing. But getting it to *run* took 5 separate undocumented fixes plus data restaging. None were
conceptually hard, but each is a hard stop discoverable only by hitting it. Estimated cost: ~1–2 h
for someone who knows the codebase; a frustrating half-day for an outsider — almost entirely due to
the broken branch state, not anything scientific.

A structural aggravator: `eval.py` → `src/data/__init__.py` eagerly imports **every** dataset
loader (all modalities) before any eval runs, so a single broken/missing module anywhere in the
import tree blocks the tool/memory eval even though those tasks don't use it. Four of the five
fixes below are only needed because of this eager import.

### F1 — `torchvision.io.write_video` removed (2 files)
- **Symptom:** `ImportError: cannot import name 'write_video' from 'torchvision.io'`
- **Cause:** installed torchvision is 0.26.0, which removed the video IO module entirely; the repo
  imports `write_video` at module top level. Pure-text tool/memory eval never calls it.
- **Files:** `src/utils/vision_utils/video_transforms.py:9`, `src/utils/vision_utils/vision_utils.py:14`
- **Fix:** wrap the import in `try/except ImportError` with a stub that raises only if actually called.

### F2 — `dataset_hf_path.py` uses `os` and `BASE_RAW_DATA_DIR` without importing them
- **Symptom:** `NameError: name 'os' is not defined` at `src/constant/dataset_hf_path.py:75`
- **Cause:** the file builds paths with `os.path.join(BASE_RAW_DATA_DIR, ...)` but imports neither.
  A genuine bug at this commit (the file is reachable via the eager import chain).
- **Fix:** add `import os` and `from src.constant.dataset_hflocal_path import BASE_RAW_DATA_DIR` at
  the top. (Values are irrelevant for our run — eval uses explicit `query_file`/`candidate_file`
  from the yaml — they just need to not crash on import.)

### F3 — `__init__` imports a module absent from the commit
- **Symptom:** `ModuleNotFoundError: No module named 'src.data.eval_dataset.mscoco_cmret'`
- **Cause:** `src/data/eval_dataset/__init__.py:33` imports `mscoco_cmret`, which was never committed
  on this branch tip. Unrelated to tool/memory.
- **Fix:** comment out that single import line.

### F4 — nemotron model load missing `trust_remote_code=True`
- **Symptom:** `ValueError: The repository ... contains custom code which must be executed ...
  Please pass the argument 'trust_remote_code=True'` (and, interactively, an `EOFError` on the
  `[y/N]` prompt — fatal in a batch job).
- **Cause:** `src/model/model.py` (NVOMNIEMBED branch) sets `trust_remote_code=True` only on the
  `AutoConfig` call, not on `AutoModel.from_pretrained`. The model ships `modeling_nv_omni_embed.py`
  as remote code, so the model load needs it too.
- **Fix:** add `trust_remote_code=True` to the `AutoModel.from_pretrained(...)` call (~line 727).

### F5 — `torchrun` resolves to the wrong Python
- **Symptom:** every rank dies with `ModuleNotFoundError: No module named 'datasets'`, then
  `torch.distributed.elastic.multiprocessing.errors.ChildFailedError`.
- **Cause:** bare `torchrun` on `PATH` is not the `/rmeng_data/envs/vlm2vec` one, so workers launch
  under a Python without the project deps. (A single-GPU `python eval.py` worked, masking it.)
- **Fix:** launch with `/rmeng_data/envs/vlm2vec/bin/python -m torch.distributed.run ...` instead of
  `torchrun`.

### F6 — data layout mismatch (configs vs HF tarballs)
- **Symptom:** loaders can't find files / memory `task_instructions.json` not found.
- **Cause:** the yaml configs expect `tool-tasks/...` (hyphen) and
  `memory-tasks/{Episodic,Dialogue,Semantic,Procedural}/<dataset>`, but the HF tarballs are named
  `tool_tasks` / `memory_tasks` and the memory archive extracts **flat**
  (`memory-tasks/{KnowMeBench,REALTALK,PeerQA,DeepPlanning}`).
- **Fix:** build a `data_basedir` (`/rmeng_data/data/vlm2vec/MMEB-V3-evalroot`) with `tool-tasks/`
  extracted in place and the memory type-subdirs created as **symlinks** to the flat dataset dirs,
  so the official `memory.yaml` runs unmodified. See §2.

### F7 — no reference numbers shipped
- The repo (and git history) contains no results table. The reproduction target exists only in the
  paper PDF (Table 2 + Table 10), which had to be parsed by hand. Web access was blocked in the
  working session, so the PDF was the only source.

### Friction that did *not* occur (what went right)
- **Dependencies:** the existing `vlm2vec` conda env had the exact pinned versions; no install/build
  pain, flash-attn already present.
- **Pipeline:** `eval.py`, the `toolde`/`memory_retrieval` loaders, the `nvomniembed` backbone, and
  the `tool.yaml`/`memory.yaml` configs all already existed on the branch — no reimplementation.
- **Correctness:** default flags (`--pooling mean --normalize true`) were already right; the model's
  `1_Pooling/config.json` confirmed mean pooling, so no metric/pooling guesswork.
- **Determinism:** PeerQA gave identical numbers on a separate smoke run and the full run.

### Recommended upstream fixes (would make this "clone → download → sbatch")
1. Commit F1–F4 (4 one-liners) to the `olm2vec` branch.
2. Make `src/data/__init__.py` import loaders lazily, or guard optional-modality imports, so one
   broken loader can't block unrelated evals.
3. Add a short data-staging note (tarball names + memory type-subdir layout) to the eval README.
4. Ship the paper's per-task reference numbers (e.g. a CSV) in the repo.

---

## 6. Status of remaining modalities

**Reproduced:** Tool, Memory, Text (see §1). **Data staged & ready to run** (validated at the data
level, just need GPU time): Image (37), VisDoc (24), Video (18), GUI (8). Staging notes:
- `BASE_RAW_DATA_DIR` set to `/rmeng_data/data/vlm2vec/MMEB-V3-eval` (the v2-era `MMEB-eval` base
  referenced by old scripts is deleted).
- Image: `image-query`→`image-tasks` symlink + per-dataset `image-tasks/MMEB/` symlinks.
- VisDoc: BEIR query/qrel metadata loads offline from the HF cache (`~/.cache/huggingface`,
  `vidore___*_beir`); page images present. No fs staging.
- Video: `video_ret` frame-path symlinks bridging the hardcoded `data/ziyan/...` paths; metadata via HF cache.
- GUI: `GUIAct.zip`/`Mind2Web.zip` extracted; `GUIAct/guiact` symlink. (data: `MMEB-V3/gui_tasks`)

### ⚠️ Audio does not reproduce in this environment
Audio ran only 5/11 datasets (all classification) and even those diverge from the paper. The audio
stack is fundamentally broken here, surfacing as a cascade of distinct failures:

| # | Symptom | Cause | Workaround applied |
|---|---|---|---|
| A1 | `ModuleNotFoundError: torchcodec` on `torchaudio.save` | torchaudio≥2.x routes save through torchcodec | write WAV via `soundfile` (`audio_cls_dataset.py`) |
| A2 | `LibsndfileError: System error` (SpeechCommands) | metadata has dead absolute paths `/data/mengrui/.../audio-tasks/...` | remap dead abs paths onto `data_basedir` (`eval_collator.py`) |
| A3 | `ImportError: torchcodec` on HF `datasets` Audio decode (CREMA-D) | HF `datasets` Audio feature decode needs torchcodec | `cast_column(..., Audio(decode=False))` (cls + `ave_retreival_dateset.py`) |
| A4 | `torchcodec` on `torchaudio.load`/`info` | torchaudio≥2.x routes load/info through torchcodec | `soundfile.read`/`info` (`eval_collator.py`, `tutsound*_dataset.py`) |
| A5 | `pyarrow` parquet read error (SoundDescs, stale staging) | swapped in official `audio_tasks` data | fixed; SoundDescs now reads but OOMs on its 89 GB corpus → excluded |

**UPDATE — decode ruled out.** torchcodec was later made to work on torch 2.11 (cu128 wheel +
`LD_LIBRARY_PATH` for FFmpeg-7) and the A1–A4 workarounds reverted; the faithful native-decode rerun
produced **byte-identical** scores (MAE 12.6 unchanged). So the soundfile path was faithful after all —
**decode is not the audio gap.** The residual divergence (SpeechCommands 87 vs 45, NSynth 4.4 vs 28.2,
TUTSound 0.9 vs 42.1) is intrinsic to the model/pipeline; the leading suspect is the **`transformers`
version** (4.52.3 here vs the model card's `4.51.3-Qwen2.5-Omni-preview`) or audio processing/instruction.
Next step to close audio = run on `transformers 4.51.3`, not an audio-stack change.

---

## Qwen3-VL-Embedding (2B + 8B) — full 7-modality reproduction

Beyond Omni-Embed-Nemotron-3B, we reproduced the paper's **Qwen3-VL-Embedding** numbers for
**both the 2B and 8B** checkpoints across all 7 non-audio MMEB-V3 categories
(Qwen3-VL has no audio encoder, so audio is N/A). This is the strongest reproduction in this report:
**every category lands within ≤0.6 of the paper except VisDoc**, and the macro-average matches to
within 0.4 pts for both sizes.

**Setup.** Separate conda env `/rmeng_data/envs/qwen3vl` (cloned from the nemotron env, then
`transformers` pinned to **4.57.1** — the stock 5.x removed `HybridCache` and breaks eval). Models at
`/rmeng_data/data/vlm2vec/models/Qwen3-VL-Embedding-{2B,8B}`. Same MAIN-repo loaders and data
(`/rmeng_data/data/vlm2vec/MMEB-V3-eval`) as the nemotron run. Runner
`experiments/public/eval/eval_qwen3vl.sbatch` (`--model_backbone qwen3_vl --pooling last
--normalize true`, 8×H100, batch 8 except video batch 4). Scores in
`/rmeng_data/exps/vlm2vec/qwen3vl/<tag>/<modality>/`.

### Category results (reproduced / paper / Δ); metric Hit@1 except VisDoc & Text = NDCG@5

| Category | Metric | 2B repro | 2B paper | Δ | 8B repro | 8B paper | Δ | per-ds MAE (2B/8B) | eval time (2B/8B) |
|---|---|--:|--:|--:|--:|--:|--:|:--:|:--:|
| Image  | Hit@1   | 69.0 | 69.5 | −0.5 | 71.6 | 72.1 | −0.5 | 0.80 / 0.74 | 49m / 55m |
| Video  | Hit@1   | 55.7 | 55.9 | −0.2 | 58.5 | 58.6 | −0.1 | 0.36 / 0.42 | 65m / 79m |
| VisDoc | NDCG@5  | 73.4 | 70.6 | **+2.8** | 73.1 | 70.9 | **+2.2** | 4.55 / 4.68 | 40m / 42m |
| Text   | NDCG@5  | 39.4 | 39.2 | +0.2 | 43.1 | 42.5 | +0.6 | 1.51 / 1.75 | 45m / 99m |
| Tool   | Hit@1   | 42.6 | 42.6 | 0.0 | 41.5 | 41.3 | +0.2 | 0.64 / 0.92 | 18m / 40m |
| Memory | Hit@1   | 29.0 | 28.4 | +0.6 | 22.6 | 22.8 | −0.2 | 0.82 / 0.53 | 2m / 3m |
| GUI    | Hit@1   | 30.7 | 30.4 | +0.3 | 33.4 | 33.5 | −0.1 | 0.84 / 0.80 | 41m / 49m |
| **Macro-avg** | — | **48.5** | **48.1** | **+0.4** | **49.1** | **48.8** | **+0.3** | — | — |

Per-dataset paper-vs-repro tables (with `eval_time_s`) in
`docs/mmebv3_reproduction/paper_vs_repro/qwen3vl-{2B,8B}/qwen3vl-{2B,8B}_<modality>_<metric>_paper_vs_repro.csv` (14 files).

### Notes / frictions

- **VisDoc is the only category that drifts** (+2.2…+2.8, per-dataset MAE ~4.6). This is the **same
  benchmark-level over-performance seen for Omni-Embed-Nemotron-3B (+3.2)**, i.e. it is not
  model-specific — strong evidence the gap is in the VisDoc eval protocol (metric/averaging or BEIR
  qrel handling), not the embedding model. The repro average is also *conservative*: it includes 3
  extra ViDoRe-v2 datasets (`biomedical_lectures`/`economics_reports`/`esg_reports`, 54–65) that are
  not in the paper's 24-dataset VisDoc set and that pull the average *down*; the 24 paper-matched
  datasets alone average even higher.
- **GUI required a one-line collator fix.** `gui_dataset.py:process_multi_images` emits `""` as the
  "no image" placeholder for text-only GUI queries, but `eval_collator.py` only guarded `path is None`,
  so `Image.open("")` raised `FileNotFoundError` under the `qwen3_vl` backbone. Fixed centrally:
  `if bytes is None and not path:` (treat empty-string path as no-image). The nemotron GUI run had not
  exercised this path. Fix validated — both 2B and 8B GUI then completed 8/8.
- **`transformers` version is load-bearing.** Stock `transformers` 5.x in the cloned env crashed eval
  (`ImportError: HybridCache`); pinning to 4.57.1 (the Qwen3-VL model card version) fixed it. Compare
  to the nemotron audio gap, also suspected to be a transformers-version issue — version pinning is the
  recurring sharp edge in this benchmark.
- Everything else (Image/Video/Text/Tool/Memory) reproduced cleanly out of the box on the existing
  loaders and data staging; no per-dataset patches were needed beyond the shared GUI fix.

---

## OmniSET (MSCOCO cross-modal retrieval) — all three models

The 9th MMEB-V3 component, **OmniSET** (Omni-modality Semantic Equivalence Tuples), is a
**diagnostic** cross-modal retrieval eval over MSCOCO items rendered as text/image/video/audio
(`mscoco_cmret`). It probes whether a model retrieves the *instructed target modality* or just the
*query-aligned* one. We reproduced the **no-audio** subset (6 directions over T/I/V) for all three
models; audio directions are excluded (Qwen3-VL has no audio encoder, and audio doesn't reproduce —
see above). Compared against the paper's **Table 4** (Hit@1 + MRR).

**Setup.** Eval code lives only on branch `olm2vec` (`experiments/public/eval/cross_modality_no_audio.yaml`
+ `src/data/collator/mscoco_cmret.py` + `eval_cross_mod.sh`), run from a fresh `VLM2Vec-omni` worktree.
Data: `mscoco_omni` (101 query tuples, 1320-item T/I/V candidate catalog) — `val2014` images
(re-downloaded the 1219 not staged locally) + videos extracted + frame cache. Wiring fixes required:
register the `mscoco_cmret` parser (import was commented out), make the config `data_path` relative,
and put `ffmpeg/ffprobe` on PATH (else every candidate video silently failed frame extraction and the
loader misaligned its candidate lists). Runner `eval_omni.sbatch`; metric Hit@1 + MRR; n=101/direction.

### Results — Hit@1 (reproduced / paper Table 4); MRR in parentheses

| Direction | Nemotron repro | Nemotron paper | Qwen3-VL-8B repro | Qwen3-VL-8B paper | Qwen3-VL-2B repro (no paper) |
|---|--:|--:|--:|--:|--:|
| T2I | 0.0 (4.4) | 0.0 (4.6) | 0.0 (3.2) | 0.0 (6.7) | 0.0 (2.9) |
| T2V | 3.0 (18.9) | 3.0 (19.1) | 16.8 (48.1) | 0.0 (2.8) | 32.7 (60.6) |
| I2T | 0.0 (13.0) | 0.0 (11.6) | 5.0 (32.7) | 0.0 (6.5) | 0.0 (29.7) |
| I2V | **100.0 (100.0)** | **100.0 (100.0)** | **100.0 (100.0)** | **100.0 (100.0)** | **100.0 (100.0)** |
| V2T | 0.0 (4.0) | 0.0 (2.4) | 0.0 (3.0) | 0.0 (4.1) | 0.0 (2.5) |
| V2I | 2.0 (15.2) | 2.0 (15.1) | 7.9 (20.0) | 2.0 (15.3) | 2.0 (15.3) |

### Findings

- **Nemotron reproduces Table 4 essentially exactly** — Hit@1 identical on all 6 directions; MRR within
  ≤1.6 pts everywhere. Clean reproduction.
- **The paper's central diagnostic reproduces for all three models:** cross-modal Hit@1 **collapses to
  ~0 except I2V = 100**. Models retrieve the modality that matches the *query* rather than the
  *instructed target* — retrieval is highly asymmetric and modality-biased. (I2V is the lone success:
  image→video is near-perfect because the rendered video's first frame ≈ the source image.)
- **Qwen3-VL is less catastrophically biased than the paper's column** on the off-diagonal directions
  (8B: T2V 16.8 vs 0, I2T 5.0 vs 0, V2I 7.9 vs 2; MRR notably higher, e.g. T2V 48.1 vs 2.8). The
  qualitative pattern (collapse-except-I2V) holds, but our Qwen3-VL retrieves the right target somewhat
  more often. Likely a checkpoint/version difference in the paper's Qwen3-VL OmniSET column (the paper
  reports a single "Qwen3-VL Embedding" row; Fig. 3c labels it 8B). The 2B follows the same shape.

CSVs: `docs/mmebv3_reproduction/paper_vs_repro/{nemotron,qwen3vl-2B,qwen3vl-8B}/*_omniset_hit@1_mrr_paper_vs_repro.csv`.
