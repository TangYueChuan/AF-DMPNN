# AF-DMPNN: Adaptive Fragment-enhanced Directed Message Passing Neural Network for Molecular Property Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/downloads/)

AF-DMPNN is an enhanced directed message passing neural network (D-MPNN) for molecular property prediction. It couples a **chemically defined fragment layer** (derived from BRICS decomposition) to the atom-level encoder, so that fragments *participate in message passing* rather than being appended as fixed descriptors. The model learns task-relevant substructure importance and provides fragment-level explanations alongside each prediction.

On seven MoleculeNet benchmarks, AF-DMPNN improves over the D-MPNN backbone and seven further graph baselines, with **RMSE reductions of 6.0–14.2%** on the three regression datasets. Fragment-masking and error analyses show that the gains are concentrated in molecules bearing several interacting functional groups — precisely the cases where atom-level models are weakest.

## News / Status

- The manuscript is under preparation for submission to a peer-reviewed journal. The full experimental protocol, ablation study, and fragment-level interpretability analyses are described in the manuscript and its Supporting Information.

## Model Overview

AF-DMPNN is built on an **atom–fragment dual-layer graph** and proceeds through four coupled modules:

1. **Dual-scale parallel directed message passing** — the molecule and its BRICS fragments are encoded in parallel with directed (bond-level) message passing; fragment context is injected into atom messages at every iteration.
2. **Gated residual fusion** — a gating layer with residual connection modulates how fragment context enters each directed-edge message, suppressing redundant fragment noise.
3. **Fragment-importance-weighted cross-scale attention** — a learned per-fragment importance score is embedded (as a log term) into the attention logits between atoms and fragments, biasing each atom toward task-relevant substructures; gated residual fusion then produces enhanced atom representations.
4. **Multiscale aggregation** — features are aggregated over atoms (sum & max pooling), fragments (importance-weighted pooling), and the whole molecule, and the concatenated embedding is fed into an MLP prediction head.

For formal definitions and equations, see the Methods section of the manuscript.

## Repository Structure

```
AF-DMPNN/
├── chemprop/                  # Modified Chemprop framework (MIT)
│   ├── args.py                # Training/prediction arguments (adds --message_model, --fragment_type)
│   ├── data/                  # Dataset loading & splitting
│   ├── features/              # Featurization + BRICS fragmentation (get_frags)
│   ├── models/
│   │   ├── af_dmpnn.py        # AF-DMPNN (full model: dual-scale + cross-scale attention)
│   │   ├── af_DMPNN_noAtt.py  # Ablation: Hier-D-MPNN-BRICSwoAtt (no cross-scale attention)
│   │   ├── Frag_DMPNN_Gate.py # Ablation: D-MPNN-BRICSGate (gated fusion only)
│   │   ├── Frag_DMPNN_Linear.py # Ablation: D-MPNN-BRICSConcat (linear concatenation only)
│   │   ├── model.py           # MoleculeModel wrapper (encoder selection)
│   │   └── mpn.py             # Original D-MPNN encoder (baseline)
│   └── train/                 # Training, cross-validation, evaluation, prediction
├── train.py                   # Training entry point
├── predict.py                 # Prediction entry point
├── interpret.py               # Interpretation entry point
├── requirements.txt
└── LICENSE.txt                # MIT
```

## Installation

The code is based on Chemprop and tested with Python 3.8 + PyTorch (CPU) + RDKit.

```bash
conda create -n afdmpnn python=3.8
conda activate afdmpnn
conda install -c conda-forge rdkit
pip install -r requirements.txt
# install PyTorch matching your CUDA version, e.g.:
# pip install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

## Datasets

Seven MoleculeNet benchmarks are used in the manuscript (MoleculeNet):

| Dataset | Molecules | Avg. Atoms | Avg. Fragments | Tasks | Task Type |
|---------|----------:|-----------:|---------------:|------:|-----------|
| ClinTox | 1,484 | 26 | 7 | 2 | Classification |
| SIDER | 1,427 | 31 | 5.5 | 27 | Classification |
| Tox21 | 7,831 | 19 | 5 | 12 | Classification |
| ToxCast | 8,597 | 32 | 14 | 617 | Classification |
| ESOL | 1,128 | 13 | 4 | 1 | Regression |
| FreeSolv | 643 | 9 | 2 | 1 | Regression |
| Lipophilicity | 4,200 | 27 | 8 | 1 | Regression |

Each CSV contains a `smiles` column followed by one or more target columns. The original datasets are from [MoleculeNet](https://moleculenet.org/) (Wu et al., Chem. Sci. 2018).

## Usage

### Training

Train AF-DMPNN on a regression task with 10-fold cross-validation:

```bash
python train.py \
  --data_path data/Lipophilicity.csv \
  --dataset_type regression \
  --metric rmse \
  --message_model af_dmpnn \
  --fragment_type brics \
  --num_folds 10 \
  --seed 0 \
  --split_type random \
  --hidden_size 300 \
  --depth 3 \
  --dropout 0.1 \
  --batch_size 50 \
  --epochs 80 \
  --init_lr 1e-4 \
  --max_lr 1e-3 \
  --final_lr 1e-4 \
  --warmup_epochs 2.0 \
  --save_dir results/Lipophilicity
```

Train AF-DMPNN on a classification task:

```bash
python train.py \
  --data_path data/clintox.csv \
  --dataset_type classification \
  --metric auc \
  --message_model af_dmpnn \
  --fragment_type brics \
  --num_folds 10 \
  --seed 0 \
  --split_type random \
  --hidden_size 300 \
  --depth 3 \
  --dropout 0.1 \
  --batch_size 50 \
  --epochs 80 \
  --init_lr 1e-4 \
  --max_lr 1e-3 \
  --final_lr 1e-4 \
  --warmup_epochs 2.0 \
  --save_dir results/ClinTox
```

### Baseline D-MPNN

Set `--message_model dmpnn` (the default) and keep every other argument identical:

```bash
python train.py \
  --data_path data/Lipophilicity.csv \
  --dataset_type regression \
  --metric rmse \
  --message_model dmpnn \
  --fragment_type brics \
  ... # same hyperparameters as above
```

### Fragmentation rules

The fragment layer supports several decomposition rules via `--fragment_type`:
`brics` (default), `murcko`, `recap`, `combo` (murcko + brics), `BaseFeatures`, and `custom`.

### Prediction

```bash
python predict.py \
  --test_path data/test.csv \
  --checkpoint_dir results/Lipophilicity \
  --preds_path predictions.csv
```

## Reproducing Manuscript Results

10-fold cross-validation, random split (8:1:1), single model per fold (no ensembling). Results reported in the manuscript:

**Classification (ROC-AUC)**

| Dataset | D-MPNN | AF-DMPNN |
|---|---:|---:|
| ClinTox | 0.886 | **0.919** |
| SIDER | 0.637 | **0.643** |
| Tox21 | 0.848 | **0.852** |
| ToxCast | 0.743 | **0.762** |

**Regression (RMSE, lower is better)**

| Dataset | D-MPNN | AF-DMPNN | Improvement |
|---|---:|---:|---:|
| Lipophilicity | 0.591 | **0.549** | −7.1% |
| ESOL | 0.647 | **0.608** | −6.0% |
| FreeSolv | 1.092 | **0.937** | −14.2% |


## Citation

If you use AF-DMPNN in your research, please cite:

```bibtex
@article{tang2026afdmpnn,
  author  = {Tang, Yuechuan and Wu, Qingxin and Yang, Jufen and Deng, Qiang and Liu, Jian},
  title   = {Adaptive Fragment-enhanced D-MPNN for Molecular Property Prediction},
  journal = {},
  year    = {2026},
  note    = {in preparation}
}
```

## Acknowledgments

This work is built upon [Chemprop](https://github.com/chemprop/chemprop) (Yang et al., *J. Chem. Inf. Model.* 2019; D-MPNN) and uses [RDKit](https://www.rdkit.org/) for cheminformatics operations. The BRICS fragmentation scheme follows Degen et al., *ChemMedChem* 2008.

## License

This project is released under the [MIT License](LICENSE.txt). The original Chemprop code remains under its MIT license (Copyright (c) 2020 Wengong Jin, Kyle Swanson, Kevin Yang, Regina Barzilay, Tommi Jaakkola).

