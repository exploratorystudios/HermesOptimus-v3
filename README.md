# Hermes Optimus: TI-84 Neural Network Word Classifier

## Status
This repository now contains two major production-relevant model lines:

- `V2`: 12-word classifier using direct 12-class outputs with hybrid input encoding.
- `V3`: 24-word classifier using the same hybrid input encoding but 12-bit binary output decoding.

Current recommended working model: **V3 24-word (`train_24word.py`, `nn_weights_24word.npz`)**.

Observed on February 18, 2026 (from repository weight files):

- V2 (`weights/nn_weights_30_50_12.npz`): `96.86%` (`185/191`)
- V3 24-word (`nn_weights_24word.npz`): `92.29%` (`347/376`)

## Core Idea
All modern variants use the same 30-dimensional input encoding:

- Dimensions `0-3`: sorted positional values (normalized A-Z values)
- Dimensions `4-29`: binary letter presence flags for `A-Z`

This gives:

- scramble resilience from order-invariant letter-set signals
- typo/adjacency resilience from continuous positional values

## Version Architecture

### V2 (12-word direct classification)
- Script: `train.py`
- Architecture: `30-50-12`
- Outputs: 12 logits mapped directly to 12 words
- Categories:
  - `BACK, DARK, EACH, FROM, JUST, BEEN, GOOD, MUCH, SOME, TIME, LIKE, ONLY`

### V3 (24-word binary codebook classification)
- Script: `train_24word.py`
- Architecture: `30-50-12`
- Outputs: 12 bits decoded to nearest 12-bit pattern (Hamming distance)
- Effective dictionary size: 24 words
- Categories:
  - `BACK, DARK, EACH, FROM, JUST, BEEN, GOOD, MUCH, SOME, TIME, LIKE, ONLY`
  - `WORK, WAVE, ZERO, ZONE, VAIN, VAST, QUIZ, HELP, FIND, PLUS, YAWN, STOP`

### Experimental / non-production variants
- `train_24word_positional.py`: 24-word binary output with full raw positional input for slots 0-3.

## Repository Components

### TI artifacts
- `HERMESV2.txt`, `HERMESV2.8xp`: V2 TI-BASIC program artifacts
- `HERMESV3.txt`, `HERMESV3.8xp`: V3 TI-BASIC program artifacts
- `[I].8xm`, `[J].8xm`, `L4.8xl`, `L5.8xl`: matrix/list payload files

### Training / testing scripts
- `train.py`: V2 trainer/test harness
- `train_24word.py`: V3 24-word trainer/test harness
- `train_24word_positional.py`: V3 positional variant
- `test_lut_activation.py`: LUT validation for V2 sigmoid approximation flow

### Weight conversion
- `weights_to_csv.py`: converts verbose TI assignment dumps to CSV files for matrix/list pipelines

### Visualization
- `visualize_network_compact.py`: legacy V2-focused visualization (12 direct outputs)
- `visualize_network_24word_binary.py`: V3 visualization (24-word + 12-bit decoding aware)

## Quick Start

### 1) Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

### 2) Run V3 24-word core test from existing weights
```bash
venv/bin/python train_24word.py --load_weights nn_weights_24word.npz --test
```

### 3) Run V2 core test from existing weights
```bash
venv/bin/python train.py --load_weights weights/nn_weights_30_50_12.npz --hidden-size 50 --test --test-modes original,scrambled,substitution,drop,adjacency
```

### 4) Generate V3 visualizations (new)
```bash
venv/bin/python visualize_network_24word_binary.py nn_weights_24word.npz --words JUST WORK
```

This produces:
- `encoding_comparison_24word.png`
- `encoding_heatmap_24word.png`
- `network_architecture_24word.png`
- `prediction_comparison_24word.png`
- `binary_codebook_24word.png`
- `weight_distribution_24word.png`
- `weight_heatmaps_24word.png`

## Training Commands

### Train V3 24-word model
```bash
venv/bin/python train_24word.py --epochs 500000 --lr 0.02 --lr-decay 0.3 --hidden-size 50 --test
```

Outputs:
- `nn_weights_24word.npz`
- `nn_weights_24word_verbose.txt`

### Train V2 baseline model
```bash
venv/bin/python train.py --epochs 500000 --lr 0.02 --lr-decay 0.3 --hidden-size 50 --test
```

Outputs (default names):
- `nn_weights_30_50_12.npz`
- `nn_weights_30_50_12_verbose.txt`

## TI Deployment Workflow (applies to both V2 and V3)
1. Train or load + export verbose weights.
2. Convert verbose weights to CSV with `weights_to_csv.py`.
3. Convert CSV into TI files (`[I].8xm`, `[J].8xm`, `L4.8xl`, `L5.8xl`) via SourceCoder/TokenIDE.
4. Transfer TI program + data files via TI Connect CE.

## Why V2 and V3 differ
- V2 uses direct 12-way classification, so each output neuron corresponds to one word.
- V3 reuses 12 outputs as bit channels, then decodes nearest codebook pattern to reach 24 words.
- V3 doubles vocabulary without increasing output width, but decoding introduces code-collision sensitivity and can reduce robustness versus direct-class outputs.

## Known Limitations
- Test generation is randomized each run; exact accuracy percentages vary slightly run-to-run.
- Visualization files are static PNG exports; no interactive explorer is included.

## Suggested Primary Files
- Model training: `train_24word.py`
- Model weights: `nn_weights_24word.npz`
- V3 visualization: `visualize_network_24word_binary.py`
- Website update guide: `WEBSITE_UPDATE_v0.md`
