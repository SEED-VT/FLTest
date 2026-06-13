# Installation

FLTest is developed against an **isolated conda environment (Python 3.11)** so it never
touches your system Python. CPU is the default and is deterministic; a GPU (Apple `mps` or
`cuda`) can be selected per-config for speed.

## Conda (recommended)

```bash
conda env create -f environment.yml      # creates env "fltest" (Python 3.11)
conda activate fltest
pip install -e ".[dev]"                   # core (reference + Flower) + test tooling
```

The core install pulls PyTorch (CPU/MPS wheels on macOS, CPU/CUDA on Linux), Flower, and
`flwr-datasets`.

## Optional backends & docs

```bash
pip install -e ".[nvflare]"   # NVFlare backend (requires Python <= 3.11)
pip install -e ".[docs]"      # this documentation site (mkdocs-material)
```

!!! note "Why Python 3.11"
    NVFlare does not yet support Python 3.12+. The core (reference + Flower) works on
    3.10–3.12, but the env is pinned to 3.11 so the NVFlare extra installs cleanly.

## Verify

```bash
fltest list
# Frameworks: ['flare', 'flower', 'flwr', 'nvflare', 'reference']
# Attacks:    ['backdoor', 'dlg', 'gaussian', 'label_flip', 'sign_flip']
# Defenses:   ['gradient_noise', 'krum', 'median', 'norm_clip', 'trimmed_mean']
# Metrics:    ['accuracy', 'loss', 'per_client']

pytest tests/ -q          # 13 passing
```

## Docker (deliverable)

A CPU/Linux `Dockerfile` is provided:

```bash
docker build -t fltest .
docker run --rm -it fltest fltest diff examples/configs/differential.yaml
```

## Devices

Set `device:` in a config to `cpu` (default, deterministic), `mps` (Apple Silicon), or
`cuda`. Differential and metamorphic tests pin `cpu` by default because GPU kernels are not
bit-reproducible.
