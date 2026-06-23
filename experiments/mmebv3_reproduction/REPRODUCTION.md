# MMEB-V3 reproduction kit

Reproduces the MMEB-V3 paper eval on this (PR #205) codebase, verified for
**Omni-Embed-Nemotron-3B** and **Qwen3-VL-Embedding-2B/8B**. One universal runner
(`run_eval.sbatch`) + aggregators, all env-var driven (no machine-specific paths).

> The audio env-compat fixes this kit depends on are committed separately on this branch
> ("Audio eval: torch-2.11 / torchcodec env compatibility fixes"). They are required on
> any **torch 2.11** env (torchaudio there has no `.info` and routes save/load through
> torchcodec, which can't use in-memory BytesIO).

## 1. Environment
- Python 3.11, torch 2.11+cu128, transformers **4.52.x** for nemotron / **4.57.1** for Qwen3-VL.
- `pip install soundfile` (used for all audio decode/encode — see fixes above).
- FFmpeg-7 must be importable: the conda env's `bin/` (ffmpeg, ffprobe) goes on `PATH` and
  its `lib/` on `LD_LIBRARY_PATH`. The runner does this automatically from `ENV_PYTHON`.
- torchcodec is optional; the kit deliberately avoids it for audio.

## 2. Data
Set `DATA_BASEDIR` to the MMEB-V3 root (same value used for `MMEB_V3_DATA_DIR`). Configs use
**relative** roots under it (e.g. `audio-tasks/nsynth-1k`, `image-tasks/...`). Use the dataset's
HF download + the repo's decompression scripts; the audio set needs these `audio-tasks/` dirs:
`nsynth-1k esc50 urbansound8k speechcommand-1k creamD sounddescs-1k clotho AVE speechcoco-1k tutsound`.

## 3. Run (universal runner)
`run_eval.sbatch` is fully env-var driven. Examples:

```bash
EVAL=experiments/mmebv3_reproduction/run_eval.sbatch
DATA=/path/to/MMEB-V3            # == DATA_BASEDIR == MMEB_V3_DATA_DIR
NEMO=/path/to/omni-embed-nemotron-3b
PYNEMO=/path/to/envs/nemotron/bin/python     # transformers 4.52.x
PYQWEN=/path/to/envs/qwen3vl/bin/python      # transformers 4.57.1

# --- Audio (Nemotron). NOTE: AUDIO_MAX_SECONDS=30 is REQUIRED (else 30s clips truncate to 10.24s);
#     NPROC=1 avoids a CPU-OOM in SoundDescs corpus materialization. ---
MODEL_PATH=$NEMO MODEL_BACKBONE=nvomniembed POOLING=mean ENV_PYTHON=$PYNEMO \
  DATA_BASEDIR=$DATA DATASET_CONFIG=experiments/public/eval/audio.yaml \
  OUTPUT_PATH=$PWD/out/nemotron/audio AUDIO_MAX_SECONDS=30 NPROC=1 \
  sbatch --exclude=<bad_nodes> $EVAL

# --- Any image/video/text/... category (Nemotron, 8 GPU) ---
MODEL_PATH=$NEMO MODEL_BACKBONE=nvomniembed POOLING=mean ENV_PYTHON=$PYNEMO \
  DATA_BASEDIR=$DATA DATASET_CONFIG=experiments/public/eval/image.yaml \
  OUTPUT_PATH=$PWD/out/nemotron/image NPROC=8 sbatch --exclude=<bad_nodes> $EVAL

# --- Qwen3-VL (2B/8B): backbone qwen3_vl, pooling last, no audio ---
MODEL_PATH=/path/Qwen3-VL-Embedding-8B MODEL_BACKBONE=qwen3_vl POOLING=last ENV_PYTHON=$PYQWEN \
  DATA_BASEDIR=$DATA DATASET_CONFIG=experiments/public/eval/image.yaml \
  OUTPUT_PATH=$PWD/out/q3vl8B/image sbatch --exclude=<bad_nodes> $EVAL

# --- OmniSET (cross-modal). no-audio variant for models without audio (Qwen3-VL) ---
MODEL_PATH=$NEMO MODEL_BACKBONE=nvomniembed POOLING=mean ENV_PYTHON=$PYNEMO \
  DATA_BASEDIR=$DATA/mscoco_omni DATASET_CONFIG=experiments/public/eval/cross_modality_no_audio.yaml \
  OUTPUT_PATH=$PWD/out/nemotron/omni NPROC=1 sbatch --exclude=<bad_nodes> $EVAL
```
`eval.py` caches `{dataset}_score.json` and skips completed datasets — re-submit to resume.

## 4. SLURM gotchas (this cluster)
- Exclude FabricManager-down / squatted nodes (the preflight aborts fast if you land on one; resubmit).
- Audio: use `NPROC=1` (8 ranks each materialize the SoundDescs corpus → >512G CPU OOM).

## 5. Expected results — Nemotron audio (Hit@1), vs the authors' reference
| dataset | this kit | reference | dataset | this kit | reference |
|---|--:|--:|---|--:|--:|
| NSynth | 0.225 | 0.227 | AVE | 0.142 | 0.142 |
| UrbanSound8K | 0.541 | 0.541 | SpeechCOCO | 0.319 | 0.323 |
| ESC-50 | 0.767 | 0.765 | TUTSound | 0.490 | 0.537 |
| SpeechCommands | 0.870 | 0.870 | TUTSound(hard) | 0.041 | 0.044 |
| CREMA-D | 0.273 | 0.269 | Clotho | 0.154 | 0.123 |
| | | | SoundDescs | 0.248 | 0.170 |

8/11 match within 0.02; Clotho/TUTSound within ~0.05; **SoundDescs is the one outlier** (+0.08,
likely a candidate-subset / crop-length setting difference — kept consistent at `AUDIO_MAX_SECONDS=30`).

Other categories (Nemotron + Qwen3-VL 2B/8B) reproduced the paper within ≤0.6 per category (VisDoc
is a consistent +2–3 benchmark-level offset); see the per-model CSVs + report in the
`rmeng/mmebv3-repro` branch under `docs/mmebv3_reproduction/`.

## 6. Aggregate vs paper
`dump_*.py` build paper-vs-reproduced CSVs from `OUTPUT_PATH`. Override I/O via env:
`SCORES_ROOT=<output root> OUT_DIR=<csv dir> python experiments/mmebv3_reproduction/dump_text_audio_csv.py`
