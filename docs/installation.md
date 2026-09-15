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
pip install -e ".[hf]"        # Hugging Face models via `model_name: hf:<id>`
pip install -e ".[docs]"      # this documentation site (mkdocs-material)
```

!!! warning "Intel Macs cap PyTorch at 2.2.2"
    PyTorch ships no macOS x86_64 wheel past 2.2.2, so an Intel Mac cannot install a newer
    one. transformers 5 requires torch 2.5 or later and disables PyTorch below it, printing
    `Disabling PyTorch because PyTorch >= 2.5 is required but found 2.2.2` and leaving only
    tokenizers usable. The `[hf]` extra therefore pins transformers below 5, which works on
    both architectures. Apple Silicon is unaffected and reaches current torch releases.

    torch 2.2.2 also needs `numpy<2`. A fresh environment that installs a current numpy
    beside it fails on import with a message about NumPy 1.x versus 2.x, which reads as
    unrelated to the version you actually pinned.

!!! note "Hugging Face versions"
    Dataset ids are namespaced (`uoft-cs/cifar10`), which is required from
    huggingface-hub 1.16 onward and works on earlier versions too. If a Hub call fails in
    another way, check `huggingface-hub`, `datasets`, and `transformers` against each
    other before suspecting FLTest.

!!! note "Why Python 3.11"
    NVFlare does not yet support Python 3.12+. The core (reference + Flower) works on
    3.10–3.12, but the env is pinned to 3.11 so the NVFlare extra installs cleanly.

## Verify

```bash
fltest list
# Frameworks: ['flare', 'flower', 'flwr', 'nvflare', 'reference']
# Attacks:    ['backdoor', 'dlg', 'gaussian', 'label_flip', 'sign_flip']
# Defenses:   ['gradient_noise', 'krum', 'median', 'mpc_aggregation', 'norm_clip',
#              'secure_aggregation', 'trimmed_mean']
# Metrics:    ['accuracy', 'loss', 'per_client']

pytest tests/ -q          # 57 passing
```

`fltest list` and `fltest pitfalls` return immediately, because neither needs to load a
model or a dataset. The first `fltest run`, `fltest diff`, or `fltest metamorphic` in a
new environment imports PyTorch, Flower, and the dataset stack, which can take a minute
before any output appears. Those commands print a notice while they load, and subsequent
runs start in a few seconds.

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
