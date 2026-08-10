# Meta-FAN

A complete, artifact-free implementation of **Meta-FAN** for few-shot building
energy forecasting.

The repository includes the full runnable method:

1. source-domain LSTM pretraining;
2. feature stability and source-target shift descriptors;
3. episodic Meta-FAN training;
4. target FAN adaptation, backbone fine-tuning, and evaluation.

It intentionally excludes paper baselines, ablation studies, visualization and
diagnostic scripts, confidential data, experiment results, model weights,
cached arrays, CSV outputs, figures, logs, and manuscript sources.

## Repository layout

```text
.
├── run_metafan.py        # complete four-stage entry point
├── model/
│   ├── lstm.py           # forecasting backbone
│   ├── meta_fan.py       # FAN and MetaGenerator
│   └── maml.py           # episodic meta-training
└── utils/
    ├── data.py           # external tensor interface and synthetic smoke data
    ├── descriptors.py    # feature stability and domain shift
    ├── loss.py
    ├── seed.py
    └── training.py       # pretraining, target adaptation, and evaluation
```

No repository code writes files during training. Metrics are printed to the
terminal and returned in memory.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\Activate.ps1     # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch wheels depend on the local CUDA version. Install the appropriate
PyTorch build first if the default wheel is unsuitable.

## End-to-end smoke run

The following command executes all four stages on data generated in memory:

```bash
python run_metafan.py --quick
```

It performs source pretraining, SHAP stability estimation, MAML, target
adaptation, fine-tuning, and evaluation. The synthetic mode is a functional
smoke test only and is not a paper benchmark.

## Running with authorized data

The public entry point accepts an external `.npz` file containing already
windowed arrays:

| Array | Shape | Meaning |
|---|---:|---|
| `X` | `(N, T, F)` | input windows |
| `y` | `(N,)` | energy-consumption target |
| `timestamps` | `(N,)` | datetime strings or NumPy datetimes |
| `domains` | `(N,)` | source/target domain labels |
| `feature_names` | `(F,)` | optional feature names |

At least three domains are required: two source domains and one target domain.
Samples within each domain are sorted chronologically. The first target
samples form the few-shot support set and later samples form the test set.

```bash
python run_metafan.py \
  --data /path/to/authorized_windows.npz \
  --target-domain target_site \
  --support-samples 288
```

`energy_last` bypasses FAN alignment when that feature name is present. Change
the bypass list with `--skip-features`.

Run `python run_metafan.py --help` for all model and training options.

## Data and reproduction scope

The confidential dataset used in the associated research is not distributed.
This repository grants no access or rights to it and does not claim to
reproduce the paper's numerical results without the original data. Users must
prepare authorized data in the documented tensor interface.

License and citation metadata will be added after the copyright holder and
final author information are confirmed.
