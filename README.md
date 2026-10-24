# FeReMM: Federated Reinforcement-based Model Merging

Code for **FeReMM**, a federated learning framework where a learned policy decides which client models get merged into the global model each round. It is compared against **FedAvg**, **FedProx**, and a **centralized** baseline on five MedMNIST datasets.

> Accepted at **IEEE BIBM 2026**. Citation below.

## How FeReMM works

Each federated round:

1. Every client trains its local copy of the model for a few epochs on its own data partition.
2. The client models and the global model are checked on a shared validation set. FeReMM collects per-layer gradient and activation statistics (cosine similarity, KL divergence, gradient magnitude) plus per-client loss and metrics.
3. A small policy network (`feremm_policy.py`) turns these signals into a state and picks which clients to merge. Its four branches cover gradients, activations, metrics, and loss.
4. The chosen clients are merged into the global model, weighted by data size (`feremm_utils.merge_clients`). Each of the 20 candidate merges is scored on the validation set by its improvement E, the policy is updated with a per-client reward, and the candidate with the highest E becomes the new global model.

Training runs for 20 global rounds. Clients train for 5 local epochs per round (25 for ResNet-50 on OCTMNIST).

## Repository layout

| Path | Purpose |
|------|---------|
| `run_centralized.py` | Centralized baseline (100 epochs) |
| `run_fedavg.py` | FedAvg baseline |
| `run_fedprox.py` | FedProx baseline |
| `run_feremm.py` | FeReMM (proposed method) |
| `feremm.py`, `feremm_policy.py`, `feremm_utils.py` | FeReMM round loop, policy network and reward, state and merge helpers |
| `federated.py` | Shared FedAvg/FedProx round loop |
| `save_results.py` | Saves each run's result tensors when `--save_data` is set |
| `train/` | Training and evaluation engines (`engine_class`, `engine_proxclass`, `engine_feremmclass`) |
| `load_model.py`, `load_resnet.py`, `model_layers.py` | ResNet-18/50 construction, optimizers, layer names used for policy features |
| `dataloaders/` | MedMNIST loading, IID client partitioning, dataloaders |
| `parser_check.py` | Command-line argument parsing and validation |
| `preprocessing.py` | Seeding (seed 42) and device selection |
| `logs/logger.py` | Training, validation, test, merge, policy, and reward logs |
| `visualization.py` | Plots (PR curves, parameter distances, logit trajectories, class distributions) |

## Requirements

- Python 3.10
- A CUDA GPU is strongly recommended. The code falls back to CPU, but that is very slow.

```bash
conda create -n feremm python=3.10 -y
```

```bash
conda activate feremm
```

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

## Data

The paper uses five [MedMNIST v2](https://medmnist.com/) datasets:

| `--dataset` | MedMNIST dataset | Image size | Classes |
|-------------|------------------|-----------:|--------:|
| `retina_mnist` | RetinaMNIST | 224 | 5 |
| `breast_mnist` | BreastMNIST | 224 | 1 (binary, BCE loss) |
| `derma_mnist` | DermaMNIST | 224 | 7 |
| `tissue_mnist` | TissueMNIST | 28 | 8 |
| `oct_mnist` | OCTMNIST | 64 | 4 |

**The datasets are not included in this repository.** The first time you use a dataset, it downloads automatically from the official MedMNIST source into `.medmnist/` at the repository root. That folder is git-ignored. Later runs reuse the downloaded files.

MedMNIST is distributed under its own terms, and each sub-dataset has its own license (mostly CC BY 4.0 or CC BY-NC 4.0). See [medmnist.com](https://medmnist.com/). If you use the data, cite MedMNIST v2 (Yang et al., *Scientific Data*, 2023) and the original sources.

## Running experiments

Run every script from the repository root. Each one takes `--model_name` (`resnet18` or `resnet50`) and `--dataset` (see the table above).

Every script also accepts two optional flags:

- `--save_data`: save the run's result tensors (`.pt` files) to `evaluation/saved_data/`.
- `--save_plots`: save the run's per-experiment plots to `evaluation/plots/`.

Without them, a run only writes its logs. Output folders are created automatically.

**Centralized baseline**

```bash
python run_centralized.py --model_name resnet18 --dataset oct_mnist
```

**Federated runs** also take `--num_clients` (at least 3). The paper uses 5, 8, and 10 clients.

**Poisoned clients (`--bad_clients N`, default 0)** turn the last `N` clients into attackers that run a *scaling attack*. Every 4th round (rounds 4, 8, 12, 16, 20), each attacker's update relative to the global model is multiplied by `num_clients` before aggregation (`feremm_utils.scaling_attack`). FedAvg and FedProx merge these updates as usual. FeReMM records how often its policy chooses to merge an attacker and saves this as `bad_merge_plot.png`. `N` must be between 0 and `num_clients - 1`. When `N` > 0, output files get `N` added after the client count, e.g. `52fedavg_pr.pt` for 5 clients and 2 attackers.

***Traditional Experiment Example Commands***
```bash
python run_fedavg.py --model_name resnet18 --dataset oct_mnist --num_clients 5
```

```bash
python run_fedprox.py --model_name resnet18 --dataset oct_mnist --num_clients 5
```

```bash
python run_feremm.py --model_name resnet18 --dataset oct_mnist --num_clients 5
```
***Poisoned Client Experiment Example Commands***
```bash
python run_fedavg.py --model_name resnet18 --dataset oct_mnist --num_clients 6 --bad_clients 2
```

```bash
python run_fedprox.py --model_name resnet18 --dataset oct_mnist --num_clients 6 --bad_clients 2
```

```bash
python run_feremm.py --model_name resnet18 --dataset oct_mnist --num_clients 6 --bad_clients 2
```
### Outputs

- **Logs:** `logs/<framework>/<model>/<dataset>/<num_clients>{training,validate,testing,merging,policy,reward}.log`
- **Saved tensors** (with `--save_data`): `evaluation/saved_data/<model>/<dataset>/`, including precision/recall, parameter distances, and logit trajectories
- **Plots** (with `--save_plots`): `evaluation/plots/<framework>/<model>/<dataset>/`

The paper's multi-panel figures are built from the saved tensors and plots, so run the experiments with `--save_data --save_plots` first. `python visualization.py` then rebuilds the parameter-distance figure; the logit-trajectory figure uses `plot_logit_trajectories` and `create_collage` in `visualization.py`.

## Reproducibility

`seed_everything(42)` fixes the Python, NumPy, and PyTorch seeds and enables deterministic cuDNN. Client partitions are IID random splits of the training set made with that seed. The optimizer is AdamW with lr 1e-5 and weight decay 1e-4 for the models, and lr 1e-4 for the policy.

## Citation

```bibtex
@inproceedings{shaheen2026feremm,
  title = {FeReMM: Federated Reinforcement-based Model
Merging for Medical Image Analysis},
  author = {Shaheen, Jordan and Bai, Jun},
  booktitle = {2026 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  year = {2026}
}
```

## License

This project is released under the [MIT License](LICENSE).
