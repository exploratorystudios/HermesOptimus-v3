# Hermes Optimus V3: TI-84 Neural Network Word Classifier

## Overview
Hermes Optimus V3 is a TI-84 Plus Silver Edition neural network classifier for 4-letter word correction/classification.

It is a feedforward backpropagated model with sigmoid activation, deployed on calculator hardware with a compact `30-50-12` architecture and 24-word dictionary decoding.

## V3 Core Design

### Architecture
- Input: `30`
- Hidden: `50`
- Output: `12`

### Input encoding (30 features)
- Features `0-3`: sorted positional letter values (normalized A-Z)
- Features `4-29`: binary A-Z letter presence flags

### Output decoding (12-bit -> 24 words)
V3 does not use direct one-neuron-per-word output. Instead:
1. Each output neuron represents one bit channel.
2. Output values are thresholded at `0.5` to produce a 12-bit vector.
3. The model searches all 24 dictionary codewords and picks the nearest by Hamming distance.

## V3 Dictionary (24 words)
`BACK, DARK, EACH, FROM, JUST, BEEN, GOOD, MUCH, SOME, TIME, LIKE, ONLY, WORK, WAVE, ZERO, ZONE, VAIN, VAST, QUIZ, HELP, FIND, PLUS, YAWN, STOP`

## Repository Focus (V3)

### Training and testing
- `train_24word.py`: main V3 trainer/test harness
- `nn_weights_24word.npz`: V3 trained weights
- `nn_weights_24word_verbose.txt`: V3 verbose TI-export weights

### Visualization
- `visualize_network_24word_binary.py`: V3 static visualization suite (7 PNG outputs)

### Export and TI deployment
- `weights_to_csv.py`: converts verbose weight dump to CSV matrix/list payloads
- `HERMESV3.txt`, `HERMESV3.8xp`: TI-BASIC program artifacts
- `[I].8xm`, `[J].8xm`, `L4.8xl`, `L5.8xl`: TI matrix/list payload files

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

## Run V3 Tests (PC)

### Test existing trained model
```bash
venv/bin/python train_24word.py --load_weights nn_weights_24word.npz --test
```

### Train and then test
```bash
venv/bin/python train_24word.py --epochs 500000 --lr 0.02 --lr-decay 0.3 --hidden-size 50 --test
```

Outputs:
- `nn_weights_24word.npz`
- `nn_weights_24word_verbose.txt`

## V3 Visualization

### Static plots
```bash
venv/bin/python visualize_network_24word_binary.py nn_weights_24word.npz --words JUST WORK
```

Generates:
- `encoding_comparison_24word.png`
- `encoding_heatmap_24word.png`
- `network_architecture_24word.png`
- `prediction_comparison_24word.png`
- `binary_codebook_24word.png`
- `weight_distribution_24word.png`
- `weight_heatmaps_24word.png`

## TI Deployment Workflow (V3)
1. Train or load V3 model and export verbose weights.
2. Convert verbose file to CSV using `weights_to_csv.py`.
3. Convert CSVs into TI files (`[I].8xm`, `[J].8xm`, `L4.8xl`, `L5.8xl`) via SourceCoder/TokenIDE.
4. Transfer `HERMESV3.8xp` and matrix/list payloads with TI Connect CE.

## Performance Reporting Note
V3 is typically reported with two views:
- Stress-suite accuracy: full adversarial perturbation mix.
- Realistic typo/transposition accuracy: filtered practical error patterns.

Keep both labels explicit when sharing results.

## Essential Folder
If you want a minimal portable V3 bundle, use files in:
- `essential/`

This folder contains the V3 trainer, V3 visualizers, export tool, V3 weights, TI artifacts, and generated V3 PNGs.
