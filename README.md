# Meta-FAN

This repository contains a **method-focused, runnable implementation of Meta-FAN** for few-shot building energy forecasting, based on the paper:

> **Meta-FAN: Building Energy Forecasting via Few-Shot Feature-Wise Adaptation**  
> Zhehao Zhang, Jiacheng Cui, Gengjie Huang, Ji Bian, Chunyang Zuo, and Guoming Tang.  
> *Energy* (accepted, October 2026). https://doi.org/10.1016/j.energy.2026.142828

The public code is intentionally limited to the **core Meta-FAN architecture and its training/adaptation pipeline**. It is a compact technical implementation of the proposed method, **not a release of the complete experimental environment or the artifacts used to produce the paper's reported results**.

## Implementation scope

The repository retains the essential components needed to run the Meta-FAN workflow:

1. **Source-domain pretraining:** train an LSTM forecasting backbone on source domains.
2. **Transferability descriptors:** estimate feature stability and source-target distribution shift.
3. **Meta-FAN training:** use episodic meta-learning to learn descriptor-conditioned initialization of feature-wise adaptation parameters.
4. **Few-shot target adaptation:** adapt FAN on target support samples, fine-tune the forecasting backbone, and evaluate predictions.

The implementation covers these four stages end to end. It does **not** include the paper's baseline implementations, ablation experiments, plotting or diagnostic utilities, confidential datasets, experiment results, pretrained weights, cached arrays, CSV exports, figures, logs, or manuscript sources.

This distinction is intentional: the repository makes the **method's technical structure and execution flow** available without representing itself as a complete reproduction package for the publication.

## Repository layout

```text
.
├── run_metafan.py        # four-stage Meta-FAN entry point
├── model/
│   ├── lstm.py           # forecasting backbone
│   ├── meta_fan.py       # FAN and MetaGenerator
│   └── maml.py           # episodic meta-training
└── utils/
    ├── data.py           # external tensor interface and synthetic smoke data
    ├── descriptors.py    # feature stability and domain shift
    ├── loss.py
    ├── seed.py
    └── training.py       # pretraining, adaptation, and evaluation
```

The training code does not write output files. Metrics are printed to the terminal and returned in memory.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch wheels depend on the local CUDA version. If necessary, install a compatible PyTorch build before installing the remaining dependencies.

## End-to-end smoke test

To exercise the complete four-stage pipeline using synthetic data generated in memory:

```bash
python run_metafan.py --quick
```

This run covers source pretraining, SHAP-based feature stability estimation, episodic MAML training, target FAN adaptation, backbone fine-tuning, and evaluation.

**The synthetic run is a functional check of the implementation, not a reproduction of the paper's benchmarks or reported numerical results.**

## Running with authorized data

The entry point accepts an external `.npz` file containing pre-windowed arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `X` | `(N, T, F)` | input windows |
| `y` | `(N,)` | energy-consumption target |
| `timestamps` | `(N,)` | datetime strings or NumPy datetimes |
| `domains` | `(N,)` | source/target domain labels |
| `feature_names` | `(F,)` | optional feature names |

At least three domains are required: two source domains and one target domain. Samples within each domain are sorted chronologically. The earliest target samples form the few-shot support set; later samples are used for testing.

```bash
python run_metafan.py \
  --data /path/to/authorized_windows.npz \
  --target-domain target_site \
  --support-samples 288
```

When the feature `energy_last` is present, it bypasses FAN alignment. The bypass list can be changed using `--skip-features`.

Run `python run_metafan.py --help` for the available model and training options.

## Data availability and reproducibility

The confidential research dataset is **not publicly distributed**. This repository neither grants access to that dataset nor includes the data preparation, full experiment configurations, and other artifacts required to reproduce the paper's numerical findings. Users must supply their own authorized data in the documented tensor format.

The available code is intended to support **inspection and execution of the Meta-FAN method itself**. Experimental comparisons and conclusions should be attributed to the paper, rather than inferred from the synthetic smoke test.

## Citation and license

If you refer to Meta-FAN in academic work, please cite the paper listed above. Final publication details, including a DOI if assigned, can be added when available.

No software license has been specified in this repository yet; licensing information will be added after the applicable copyright and release terms are confirmed.
