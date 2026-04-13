# mace-model

`mace-model` is a repository that contains only the MACE model code derived
from the upstream Torch and JAX projects:

- [`ACEsuit/mace`](https://github.com/ACEsuit/mace)
- [`ACEsuit/mace-jax`](https://github.com/ACEsuit/mace-jax)

Its purpose is to keep both the Torch and JAX MACE implementations in one
repository, minimize code overlap between them, and deliberately exclude
training, data handling, and training workflow code.

Companion repository:

- [`equitrain`](https://github.com/bamescience/equitrain): preprocessing,
  training, fine-tuning, and run management

It provides:

- a PyTorch backend through `mace_model.torch`
- a JAX backend through `mace_model.jax`
- a shared core so both backends implement the same model semantics
- a small CLI for constructing initialized models from a config file

It installs as a single Python package namespace, `mace_model`. The backend
implementations and shared code live below that namespace instead of being
installed as separate top-level packages.

It does not provide the training stack. Data handling, preprocessing, training
loops, checkpoint management, and run management belong in `equitrain`.

The intended workflow is:

1. use `mace-model` to initialize, convert, or download a MACE model artifact
2. use [`equitrain`](https://github.com/bamescience/equitrain) to preprocess
   data, train, fine-tune, and manage training runs

## Purpose

The role of this repository is to define the model once and expose it through
two backends.

The goal is not to hide all backend differences. The goal is to:

- share the model logic
- keep Torch-specific code local to the Torch backend
- keep JAX-specific code local to the JAX backend
- validate parity between both implementations

That keeps the architecture work in one place instead of maintaining two model
codebases that drift over time.

In this workspace, `mace-model` is the model-definition repository and
[`equitrain`](https://github.com/bamescience/equitrain) is the training
repository.

## Install

Install the repository in editable mode when developing locally:

```bash
pip install -e .
```

This installs the `mace-model` runtime dependencies together with the CLI
entry points.

JAX functionality additionally requires the native `cuequivariance-jax` runtime
to be available on the machine. If those native ops are missing, Torch-only
functionality still works, but JAX model execution and the JAX-dependent tests
will be unavailable.

Programmatic imports now follow the same single-package structure:

```python
from mace_model import torch, jax
from mace_model.torch import ScaleShiftMACE
from mace_model.jax.tools import build_model
```

If you want the full training workflow, install
[`equitrain`](https://github.com/bamescience/equitrain) alongside this
repository. The repository boundary is intentional:

- `mace-model`: model definitions, backend adapters, model serialization,
  conversion, and foundation-model export
- `equitrain`: dataset preprocessing, training loops, fine-tuning, checkpoints,
  and experiment execution

## Initialize A Model From Config

The main entry point for constructing an initialized model is:

```bash
mace-model-init --config examples/initial-model.toml
```

The CLI is intentionally small. The supported arguments are:

- `--config`: path to a `.toml` or `.json` config file
- `--backend`: optional override for `torch` or `jax`
- `--output`: optional output path override
- `--print-example-config`: print a starter config and exit

Useful commands:

```bash
mace-model-init --print-example-config
mace-model-init --config examples/initial-model.toml
mace-model-init --config examples/initial-model.toml --backend jax
mace-model-init --config examples/initial-model.toml --output models/init-model
```

If you prefer not to rely on the installed script, the same entry point can be
called with:

```bash
python -m mace_model.cli --config examples/initial-model.toml
```

## Convert A Torch Model To JAX

Use `mace-model-convert` when you already have a Torch model and want a local
JAX bundle:

```bash
mace-model-convert path/to/model.pt --output models/model-jax
```

Accepted inputs:

- a Torch checkpoint containing a pickled model
- a Torch checkpoint containing `{"model": ...}`
- a local Torch bundle directory containing `config.json` and `state_dict.pt`

Main arguments:

- positional `torch_model`: input Torch checkpoint or local Torch bundle directory
- `--output`: output path for the JAX bundle
- `--head`: optional head selection for multi-head Torch models
- `--device`: device used if an intermediate legacy Torch conversion is required

Examples:

```bash
mace-model-convert foundation.pt --output models/foundation-jax
mace-model-convert models/local-torch-model --output models/local-torch-model-jax
```

The output is a standard local JAX bundle:

- `config.json`
- `params.msgpack`

## Download A Foundation Model

`mace-model` also provides a second CLI for downloading an upstream MACE
foundation model and exporting it in the local backend format:

```bash
mace-model-foundation --backend jax --source mp --model medium-mpa-0
```

Supported sources are:

- `mp`
- `off`
- `anicc`
- `omol`

Main arguments:

- `--backend`: `torch` or `jax`
- `--source`: foundation source
- `--model`: optional variant or model path, depending on the source
- `--head`: optional head to select from a multi-head foundation model
- `--output`: optional output path
- `--device`: upstream Torch download/conversion device, default `cpu`
- `--dtype`: optional upstream loader dtype override

Examples:

```bash
mace-model-foundation --backend torch --source mp --model medium-mpa-0
mace-model-foundation --backend jax --source off --model small --output models/off-small-jax
mace-model-foundation --backend torch --source mp --model medium-mpa-0 --head Default
```

Behavior:

- for `jax`, the downloader writes a local JAX bundle (`config.json` +
  `params.msgpack`)
- for `torch`, the downloader converts the upstream Torch foundation model to
  the local `mace_model.torch` implementation and writes `config.json` +
  `state_dict.pt`

Internally, foundation loading uses the repo-local legacy checkpoint loader and
conversion path. It does not import the upstream `mace` package at runtime.

## Config Structure

The config is split into a small number of clear sections.

Top-level keys:

- `backend`: `torch` or `jax`
- `model_class`: currently `MACE` or `ScaleShiftMACE`
- `seed`: initialization seed
- `output`: optional default output path

Shared model options live under `[model]`.

Backend-specific overrides live under:

- `[torch.model]`
- `[jax.model]`

This means the common architecture can be defined once, while backend-only
details stay local to the backend that needs them.

A typical layout looks like this:

```toml
backend = "torch"
model_class = "ScaleShiftMACE"
seed = 0

[model]
r_max = 4.5
num_bessel = 4
num_interactions = 2
hidden_irreps = "16x0e + 16x1o"
MLP_irreps = "8x0e"
atomic_numbers = [11, 17]
atomic_energies = [-1.25, -2.0]

[torch.model]
use_edge_irreps_first = false

[jax.model]
replace_symmetric_contraction = false
attn_num_heads = 4
```

The repository includes a fuller example in `examples/initial-model.toml`.

## What The CLI Writes

The output format depends on the selected backend.

For `torch`:

- directory output writes `config.json` and `state_dict.pt`
- file output ending in `.pt`, `.pth`, or `.ckpt` writes a single payload

For `jax`:

- directory output writes `config.json` and `params.msgpack`
- file output ending in `.json` or `.msgpack` writes the paired bundle files

The saved `config.json` is a normalized snapshot of the resolved model config,
including backend-specific overrides.

## Programmatic Usage

If you want to build models directly in Python, use the backend entry points.

Torch:

```python
from mace_model.torch import (
    MACE,
    ScaleShiftMACE,
    compile_model,
    export_model,
    graph_to_inference_args,
    make_inference_wrapper,
)
```

JAX:

```python
from mace_model.jax import MACE, ScaleShiftMACE
from mace_model.jax.tools import build_model, load_model_bundle, prepare_template_data
```

Use Torch when you want a standard `torch.nn.Module`.

Use JAX when you want an `nnx` model, JAX transforms, or model bundles that fit
into a JAX workflow.

## Compile And Export Torch Models

For Torch inference, `mace-model` exposes a tensor-only wrapper plus helper
functions for `torch.compile` and `torch.export`.

The wrapper expects the standard graph tensors:

- `positions`
- `node_attrs`
- `edge_index`
- `shifts`
- `unit_shifts`
- `cell`
- `batch`
- `ptr`
- optional `head`

Example:

```python
import torch
from mace_model.torch import compile_model, export_model, graph_to_inference_args

compiled = compile_model(model, output_keys=("energy", "node_energy"))
args = graph_to_inference_args(graph)
energy, node_energy = compiled(*args)

exported = export_model(
    model,
    graph,
    output_keys=("energy", "node_energy"),
    strict=False,
)
exported_energy, exported_node_energy = exported.module()(*args)
```

The compile/export helpers are inference-oriented:

- they wrap the model with a stable tensor-only signature
- they request only the observables implied by `output_keys`
- they currently target the standard non-LAMMPS graph path

## Relationship to Equitrain

The intended split is:

- [`mace-model`](https://github.com/bamescience/mace-model): model definitions,
  backend adapters, model serialization helpers, conversion, and foundation
  export
- [`equitrain`](https://github.com/bamescience/equitrain): data pipelines,
  training loops, checkpoints, and run management

## Development

Run the test suite with:

```bash
python -m pytest -q tests
```

The real foundation-model parity test is opt-in and only runs when
`MACE_MODEL_RUN_FOUNDATION_PARITY=1` is set.
