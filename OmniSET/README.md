# OmniSET Visualization Pipeline

This directory provides a lightweight visualization / retrieval demo pipeline:

1. Compute and save embeddings (including direction-specific query embeddings)
2. Compute t-SNE coordinates and cosine shifts
3. Plot sampled 7-cluster t-SNE figures and the heatmap

> **Note**: this folder is the visualization / lightweight retrieval demo only.
> The full OmniSET cross-modal retrieval evaluation is run via
> `experiments/public/eval/eval_omniset.sh` at the project root (relative to
> this folder: `../experiments/public/eval/eval_omniset.sh`). See that script
> for the complete evaluation workflow.

## Directory Structure

- `01_compute_embeddings.py`: Step 1
- `02_compute_tsne_and_shift.py`: Step 2
- `03_plot_tsne_and_heatmap.py`: Step 3
- `04_lightweight_retrieval.py`: Lightweight retrieval (top-k from embeddings NPZ)
- `run_plot.sh`: One-command plotting runner (auto env activation + auto Step1/Step2/Step3)
- `run_retrieval_top_10.sh`: One-command retrieval runner (default top-10)
- `omniset_pipeline/`: Reusable modules (data loading, encoding, analysis, plotting)
- `notebooks/omniset_pipeline.ipynb`: End-to-end notebook workflow

## One-Command Run

```bash
cd OmniSET
bash run_plot.sh
```

If you want to force recomputing embeddings:

```bash
FORCE_REEMBED=1 bash run_plot.sh
```

If you want to pin one GPU and use an explicit dataset path:

```bash
CUDA_VISIBLE_DEVICES=0 \
DATASET_ROOT=/path/to/MMEB-V3/omniset \
META_FILE=omniset.jsonl \
bash run_plot.sh
```

## Lightweight Retrieval

Default top-10 retrieval:

```bash
bash run_retrieval_top_10.sh
```

Use all modalities as candidate pool:

```bash
CANDIDATE_POOL=all bash run_retrieval_top_10.sh
```

Manual script call:

```bash
python 04_lightweight_retrieval.py
```

Custom query + instruction (text query mode):

```bash
python 04_lightweight_retrieval.py \
  --source-modality t \
  --target-modality i \
  --query-text "a cat looking at a bowl" \
  --instruction "Retrieve the matching image with the same semantics." \
  --topk 10
```

Save retrieval output JSON:

```bash
python 04_lightweight_retrieval.py \
  --save-json ./outputs/retrieval_top10.json
```

## Quick Start (Manual 3 Steps)

Run the following commands inside `OmniSET`:

```bash
python 01_compute_embeddings.py \
  --dataset-root "/path/to/MMEB-V3/omniset" \
  --meta-file "omniset.jsonl" \
  --catalog-file "catalog.jsonl" \
  --model-path "$HOME/.cache/huggingface/omni-embed-nemotron-3b" \
  --output-dir ./outputs/step1_embeddings \
  --max-samples 0

python 02_compute_tsne_and_shift.py \
  --embedding-npz ./outputs/step1_embeddings/embeddings_and_queries.npz \
  --output-dir ./outputs/step2_analysis \
  --tsne-perplexity 30 \
  --tsne-metric cosine

python 03_plot_tsne_and_heatmap.py \
  --cluster-points-csv ./outputs/step2_analysis/tsne_cluster_points.csv \
  --delta-matrix-csv ./outputs/step2_analysis/delta_cosine_matrix.csv \
  --output-dir ./outputs/step3_figures \
  --cluster-sample-ratio 0.25
```

## Sampling and Plotting Logic

- 7-cluster plots are grouped by `plot_source_modality_name + point_role + target_modality`.
- Sampling uses deterministic head sampling per group: keep `max(1, int(n * ratio))`.
- Each source-modality figure contains:
  - `Query` (circle, source color)
  - `Instruction + Query` (downward triangle, target color)
  - `Target` (upward triangle, target color)
- The heatmap uses `mean_delta_cosine_similarity_instr_minus_raw`:
  - Positive values mean the instruction-conditioned query is closer to the target than the raw query.

## Main Outputs

Step 1:
- `embeddings_and_queries.npz`
- `embedding_summary.json`

Step 2:
- `tsne_reference_points.csv`
- `pair_shift_metrics.csv`
- `tsne_cluster_points.csv`
- `delta_cosine_matrix.csv`
- `delta_cosine_matrix.json`
- `analysis_summary.json`

Step 3:
- `sampled_tsne_cluster_points.csv`
- `tsne_clusters/tsne_clusters_source_*.png`
- `delta_cosine_heatmap.png`
- `delta_cosine_heatmap.pdf`
