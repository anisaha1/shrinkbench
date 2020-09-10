# ULP Pruning for Backdoor Attacks

This is the implementation for pruning based model sanitization against backdoor attacks. It is based on the open source PyTorch library [ShrinkBench](https://github.com/JJGO/shrinkbench).

# Installation

First, install the dependencies, this repo depends on
 - `PyTorch`
 - `Torchvision`
 - `NumPy`
 - `Pandas`
 - `Matplotlib`

To install the module itself you just need to clone the repo and  add the parent path it to your `PYTHONPATH`. For example:

```bash
git clone git@github.com:JJGO/shrinkbench.git shrinkbench

# Bash
echo "export PYTHONPATH=\"$PWD:\$PYTHONPATH\"" >> ~/.bashrc

# ZSH
echo "export PYTHONPATH=\"$PWD:\$PYTHONPATH\"" >> ~/.zshrc
```

# Experiments

See [here](jupyter/experiment_tutorial.ipynb) for a notebook showing how to run pruning experiments.

## Modules

The modules are organized as follows:

| submodule | Description |
| ---- | ---- |
| `analysis/` | Aggregated survey results over 80 pruning papers |
| `datasets/` | Standardized dataloaders for supported datasets |
| `experiment/` | Main experiment class with the data loading, pruning, finetuning & evaluation |
| `metrics/` | Utils for measuring accuracy, model size, flops & memory footprint |
| `models/` | Custom architectures not included in `torchvision` |
| `plot/` | Utils for plotting across the logged dimensions |
| `pruning/` | General pruning and masking API.  |
| `scripts/` | Executable scripts for running experiments (see `experiment/`) |
| `strategies/` | Baselines pruning methods, mainly magnitude pruning based |

