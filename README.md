# flsim — a modular federated learning simulator

`flsim` simulates federated learning over a wireless network. It trains real PyTorch models on real
datasets, and at the same time it models what the training would cost on physical devices: how long
each client computes, how long its uploads and downloads take over a fading wireless channel, and how
much energy it spends. Every run reports accuracy **and** simulated time, energy and traffic, so
learning methods can be compared on the axes that matter for deployment.

Four distributed-learning paradigms are built in, each demonstrated by one example:

| Paradigm | What changes | Orchestrator | Algorithm | Example |
|---|---|---|---|---|
| **Synchronous FL** | server waits for every selected client, then averages | `Simulator` | `FedAvg` | `examples/fedavg_experiment.py` |
| **Asynchronous FL** | server updates the model on every single arrival | `AsyncSimulator` | `FedAsync` | `examples/fl_vs_fedasync_experiment.py` |
| **Over-the-air aggregation** | clients transmit simultaneously; the channel computes the sum | `Simulator` + uplink-physics hook | `FedOTA` | `examples/ota_experiment.py` |
| **Split learning** | the model is cut in two; clients train only the first part | `SplitSimulator` | SL, SplitFed V1, SplitFed V2 | `examples/splitfed_experiment.py` |

All four sit on **one physical model** (same channel, same compute and energy formulas, same device
population), which is what makes their time and energy numbers directly comparable.

The guiding design rule: **to build a new method, override only what changes.** An algorithm is a
small class with a few hooks; the orchestrators, the physics, the logging and the plotting are shared.

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Architecture](#architecture)
4. [The physical model](#the-physical-model)
5. [The four paradigms](#the-four-paradigms)
   - [Synchronous FL — FedAvg](#synchronous-fl--fedavg)
   - [Asynchronous FL — FedAsync](#asynchronous-fl--fedasync)
   - [Over-the-air aggregation — FedOTA](#over-the-air-aggregation--fedota)
   - [Split learning — SL / SplitFed V1 / V2](#split-learning--sl--splitfed-v1--v2)
6. [The examples](#the-examples)
7. [Writing your own experiments and algorithms](#writing-your-own-experiments-and-algorithms)
8. [Configuration](#configuration)
9. [Outputs and metrics](#outputs-and-metrics)
10. [Tests](#tests)

---

## Installation

Python ≥ 3.10. From the repository root:

```bash
pip install -e .            # the framework (flsim) and its dependencies
pip install -e ".[dev]"     # + pytest, to run the tests
```

or `pip install -r requirements.txt` and run the scripts from the repository root.

Datasets (MNIST, CIFAR-10, CIFAR-100) are downloaded automatically to `~/.flsim/data/` on first use.

---

## Quick start

Run every command from the repository root.

```bash
# Synchronous FedAvg on MNIST (30 clients, non-IID shards)
python examples/fedavg_experiment.py
python examples/fedavg_experiment.py --rounds 50 --clients_per_round 5 --lr 0.005

# Synchronous FL vs asynchronous FedAsync — ResNet-34 / CIFAR-10, IID and non-IID
python examples/fl_vs_fedasync_experiment.py --smoke     # small laptop-sized version
python examples/fl_vs_fedasync_experiment.py             # full run (GPU recommended)

# Over-the-air aggregation (FedOTA, three MSE budgets) vs digital FedAvg
python examples/ota_experiment.py

# Normal / FL / FedAsync / SL / SplitFed V1 / SplitFed V2 on accuracy, time, energy and traffic
python examples/splitfed_experiment.py

# Re-draw the plots of any saved run
python plot_results.py outputs/fedAVG/fedavg/fedavg.csv
```

Each example writes its CSV files and figures under `outputs/<experiment>/` (see
[Outputs and metrics](#outputs-and-metrics)). A GPU is used automatically when one is available.

---

## Architecture

`flsim` is layered. Each layer depends only on the layers below it, and every extensible component is
defined by an abstract interface, so a new algorithm, channel or partitioner plugs in without touching
the rest.

```
  examples/  ── your experiment scripts: choose algorithms, settings, comparison plots
      │
  experiments/  Experiment, AsyncExperiment, SplitExperiment
      │         load a YAML config + overrides → build all components → run → CSV, plots, RunResult
      │
  core/      Simulator (sync, + OTA hook)   AsyncSimulator   SplitSimulator
      │      Server, Client, SplitClient, Evaluator, loggers
      │
  algorithms/  FedAvg, FedAsync, FedOTA (+ FedProx)          ── the learning rules
  system/      time, energy, over-the-air and split-cost physics
  channel/     path loss, fading, Shannon rate
  data/        datasets + IID / shard / Dirichlet partitioners
  models/      MNIST/CIFAR CNNs, AlexNet, VGG, ResNet (all splittable)
  profiles/    per-client devices: distance, CPU frequency, transmit power
  allocators/  bandwidth / power / CPU-frequency allocation
      │
  interfaces/  abstract base classes — the contracts every component implements
```

### Project layout

```
flsim/
├── interfaces/                  # contracts (abstract base classes, no logic)
│   ├── algorithm.py               FederatedAlgorithm       sync: select_clients, configure_client, aggregate
│   ├── async_algorithm.py         AsyncFederatedAlgorithm  async: select_clients, mixing_weight, aggregate_async
│   ├── splittable.py              Splittable               split learning: ordered_layers()
│   ├── channel_model.py           ChannelModel             channel_gain, achievable_rate_bps
│   ├── time_model.py              TimeModel                computation / upload / download time
│   ├── allocator.py               ResourceAllocator        bandwidth, power, CPU frequency
│   └── partitioner.py             DataPartitioner          partition, describe
│
├── algorithms/
│   ├── fedavg.py                  FedAvg    sample-weighted averaging (sync)
│   ├── fedasync.py                FedAsync  staleness-aware mixing on each arrival (async)
│   ├── fedota.py                  FedOTA    over-the-air aggregation with MSE-budget device selection
│   └── fedprox.py                 FedProx   FedAvg + proximal term (minimal example of configure_client)
│
├── core/
│   ├── simulator.py               synchronous round loop (+ optional uplink-physics hook used by FedOTA)
│   ├── async_simulator.py         discrete-event asynchronous simulator (sliding window, staleness)
│   ├── split_simulator.py         split-learning orchestrator (SL / SFLV1 / SFLV2)
│   ├── server.py, client.py       global model + algorithm; local PyTorch training
│   ├── split_client.py            forward/backward relay across the cut layer
│   ├── training_utils.py          local work per round (epochs or H mini-batch steps)
│   ├── evaluator.py               test-set evaluation
│   └── logger.py, async_logger.py CSV logging
│
├── system/
│   ├── cellular_time.py           computation time C·D/(f·q), Shannon-rate upload/download time
│   ├── energy.py                  computation energy κ·C·D·f², transmit energy p·t
│   ├── ota.py                     OTAChannel — over-the-air aggregation physics
│   ├── split_cost.py              SplitCostModel — split-learning latency / energy / traffic
│   ├── split_model.py             split_model(model, cut_layer) → (client part, server part)
│   └── flops.py                   forward-MAC counter (per-sample workload measured from the model)
│
├── channel/                     # path_loss.py (3GPP UMa + shadowing), exp_fading.py, fdma.py
├── data/                        # iid.py, shard.py, dirichlet.py + loaders/ (mnist, cifar10, cifar100, ham10000)
├── models/                      # mnist_cnn, cifar_cnn, alexnet, vgg16/19, resnet18/34 + factory.py
├── profiles/                    # ClientSystemProfile + heterogeneous device generation
├── allocators/                  # EqualSplitAllocator (FDMA equal bandwidth, max power)
├── experiments/
│   ├── base.py                    Experiment + RunResult (sync; shared plotting for all paradigms)
│   ├── async_base.py              AsyncExperiment
│   ├── split_base.py              SplitExperiment
│   ├── compare_algorithms.py      AlgorithmComparison (several algorithms, one config)
│   ├── parameter_sweep.py         ParameterSweep (one setting over several values)
│   ├── reporting.py               unified CSV + standard plots for every run
│   └── wiring.py                  config loading and component factories
├── configs/                     # base.yaml (every option, documented), mnist_fedavg.yaml
└── tests/                       # unit tests (pytest)

examples/                        # the four experiments above
plot_results.py                  # re-plot any saved CSV
```

### How a run is put together

1. **Config.** A YAML file (`flsim/configs/base.yaml` documents every option) plus a dictionary of
   overrides in dot notation, e.g. `{"learning.global_rounds": 50, "data.partition": "iid"}`.
2. **Wiring.** `flsim/experiments/wiring.py` turns the config into components: the dataset and its
   partition across clients, the model, the client device profiles, the channel model, the time and
   energy models, the resource allocator and the algorithm. Any component can be replaced by passing
   `components={"algorithm": FedAsync(...)}` (or `"allocator"`, `"channel_model"`, …).
3. **Simulation.** The orchestrator for the paradigm runs the training loop. PyTorch does the real
   learning; the physical models compute the simulated time and energy of every step. Simulated time is
   analytic — it never depends on how fast your own machine is.
4. **Results.** Every run writes its CSV, a unified CSV with the same columns for every paradigm, and a
   standard set of plots, and returns a `RunResult` for comparison plots in your script.

---

## The physical model

All four paradigms share these formulas, which is why their costs are comparable.

**Devices.** Each client has a distance to the base station, a CPU frequency `f` and a transmit power
`p` (`flsim/profiles`), fixed or heterogeneous (e.g. `f ~ U[f_min, f_max]`, distance `~ U[d_min, d_max]`).

**Computation.** Local training on `D` samples (or `H` mini-batches of size `b`, `D = H·b`) takes

```
t_comp = C · D / (f · q)            E_comp = κ · C · D · f² / q
```

where `C` is the per-sample workload. By default `C` is **measured from the model** (its multiply-
accumulate count, `system.cycles_per_sample_mode: model_macs`), so any model gets a consistent time
scale without hand-tuning; `q` is FLOPs per cycle and `κ` the switched capacitance.

**Channel.** The channel gain `g` comes from the configured model: `path_loss` (3GPP urban macro path
loss with frozen log-normal shadowing) or `exp_fading` (`g = h0 · ρ · d^-α`, Rayleigh power fading
`ρ ~ Exp(1)` redrawn every round). An optional SNR floor (`wireless.min_snr_db`) bounds deep fades.

**Communication.** Digital uploads use FDMA: each client gets a share `B_k` of the bandwidth and
transmits at the Shannon rate

```
R_k = B_k · log2(1 + g_k · p_k / (N0 · B_k))        t_up = bits / R_k        E_tx = p_k · t_up
```

with the upload size either fixed or the real model size (`wireless.upload_size_mode: model`). The
downlink is modeled the same way (optionally with a separate base-station power, or ignored).

**Per paradigm,** only the way these pieces combine changes:

| Paradigm | Time of one step | Uplink physics |
|---|---|---|
| Sync FL | round = slowest selected client (compute + upload + download) | FDMA |
| Async FL | each update arrives at its own time; no waiting for others | FDMA, bandwidth shared by the clients in flight |
| Over-the-air | round = slowest compute + one simultaneous transmission | analog superposition, time independent of the number of clients |
| Split learning | device and server compute + activation / gradient / model transfers | FDMA, faster edge server for the server part |

---

## The four paradigms

### Synchronous FL — FedAvg

Each round the server selects `clients_per_round` clients, sends them the global model, each trains
locally, and the server replaces the model with the sample-weighted average of the returned models:

```
w ← Σ_k (n_k / n) · w_k
```

The round lasts as long as the slowest selected client (the straggler). `FedAvg`
(`flsim/algorithms/fedavg.py`) implements the `FederatedAlgorithm` interface:

| Hook | Default | Purpose |
|---|---|---|
| `select_clients(all_clients, num_to_select, rng, **kwargs)` | uniform random | which clients train this round |
| `configure_client(client, global_model, round_idx)` | no-op | per-round client setup |
| `aggregate(global_model, client_updates)` | **required** | the new global weights |

`select_clients` receives the system context (`channel_model`, `noise_psd_w_per_hz`,
`bw_per_client_hz`, `round_idx`), so channel- or rate-aware selection needs no extra plumbing.

### Asynchronous FL — FedAsync

Based on Xie, Koyejo & Gupta, *"Asynchronous Federated Optimization"* (2019). The server never waits:
every arriving update is mixed into the global model immediately, and a replacement client is
dispatched with the newest model.

| | Synchronous (FedAvg) | Asynchronous (FedAsync) |
|---|---|---|
| Server waits for | all selected clients | nobody — each arrival triggers an update |
| Update rule | weighted average | `x_t = (1 − α_t) · x_{t−1} + α_t · x_new` |
| Step duration | slowest client | the arrival time of one update |
| Staleness | 0 | `t − τ`: how many model versions old the update is |

**Simulation.** `AsyncSimulator` is a discrete-event simulator. `window_size` clients are in flight at
once (each trains on a copy of the model it received — staleness is real, not imitated). The earliest
arrival is popped from a priority queue, mixed in with weight `α_t`, the simulated clock jumps to its
arrival time, and one replacement is dispatched on the updated model. The uplink bandwidth is shared by
the clients in flight.

**Staleness functions.** How much a stale update is trusted, `α_t = α · s(t − τ)`:

```python
from flsim.algorithms.fedasync import FedAsync

FedAsync(alpha=0.5)                                          # constant:   s(k) = 1
FedAsync(alpha=0.5, staleness_func="polynomial", a=0.5)      # polynomial: s(k) = (k + 1)^(-a)
FedAsync(alpha=0.5, staleness_func="hinge", a=10.0, b=4.0)   # hinge:      s(k) = 1 if k ≤ b else 1/(a(k − b) + 1)
FedAsync(alpha=0.5, rho=0.01)                                # + local proximal term (ρ/2)·‖x − x_τ‖²
```

`window_size` (config `async_fl.window_size`) is the concurrency level, not a cohort: all clients stay
eligible, and a larger window means more overlap, more staleness and more throughput. One async "global
epoch" is **one** client update, so compare async and sync runs on simulated time and cumulative
energy, not on rounds.

Interface `AsyncFederatedAlgorithm` (`flsim/interfaces/async_algorithm.py`): `aggregate_async(...)`
(required), `mixing_weight(base_alpha, staleness)`, `select_clients(...)`, `configure_client(...)`.

### Over-the-air aggregation — FedOTA

Based on Yang, Jiang, Shi & Ding, *"Federated Learning via Over-the-Air Computation"*
(arXiv:1812.11750). Instead of separate digital uploads, the selected clients transmit **at the same
time** over the whole band, and the superposition of their analog signals at the receiver *is* the
weighted sum. Upload time no longer grows with the number of clients; the price is aggregation noise.

| | Digital FedAvg (FDMA) | FedOTA |
|---|---|---|
| Uplink | each client its own bandwidth slice | all clients simultaneously, full band |
| Aggregation | exact weighted average | weighted average + Gaussian noise of the achieved MSE |
| Upload time | `bits / R_k`, grows with the number of clients | `symbols / B`, independent of the number of clients |
| Transmit energy | `p · t_up` | `symbols · ‖b_i‖²` (transmit-scalar energy) |
| Device selection | any | every device whose MSE ≤ `gamma` (maximum participation within an error budget) |

The physics live in `flsim.system.ota.OTAChannel` (single-antenna receiver):

```
η   = min_i  P0 · g_i / φ_i²                    power-normalizing factor
b_i = sqrt(η) · φ_i / sqrt(g_i)                  zero-forcing transmit scalar
MSE = σ² / η                                     achieved aggregation error
ẑ   = Σ φ_i w_i / Σ φ_i  +  N(0, MSE / (Σ φ_i)²) noisy weighted average
```

with `φ_i = n_i` (samples), `g_i` the channel gain, `P0` the peak power and `σ²` the receiver noise.

```python
from flsim.algorithms.fedota import FedOTA
from flsim.channel.conversions import dbm_to_watts

FedOTA(p0_w=dbm_to_watts(10.0), noise_power_w=1.03e-20, gamma_db=5.0)   # looser gamma: more devices, more noise
```

`FedOTA` is an ordinary synchronous algorithm with three overrides: `select_clients` (MSE threshold),
`aggregate` (noisy over-the-air sum) and `recompute_uplink_physics`, a hook the synchronous `Simulator`
calls when an algorithm defines it, so the CSV's upload time and energy columns follow the OTA
physics instead of FDMA. Per-round extras are kept on the object: `mse_history`, `energy_history`,
`excluded_history`.

### Split learning — SL / SplitFed V1 / V2

Based on Thapa et al., *"SplitFed: When Federated Learning Meets Split Learning"* (AAAI-22,
arXiv:2004.12088). The model is cut at a layer: each client runs the first part, sends the cut-layer
activations ("smashed data") to the server, the server finishes the forward pass and the loss, and the
gradient travels back across the cut. Raw data never leaves the client, and the client holds only part
of the model.

**The cut.** Every model implements `Splittable.ordered_layers()`, a flat list of its layers;
`learning.cut_layer` is the split index (client: `layers[:cut]`, server: the rest). Residual blocks are
atomic, so ResNets are cut between blocks.

**Three variants, two independent choices** — how the client parts and the server parts are combined
across clients (`sequential`: one shared model, clients one after another; `parallel_fedavg`: one copy
per client, averaged at the end of the round):

| Variant | `client_mode` | `server_mode` |
|---|---|---|
| **SL** (vanilla split learning) | `sequential` | `sequential` |
| **SFLV1** (SplitFed V1) | `parallel_fedavg` | `parallel_fedavg` |
| **SFLV2** (SplitFed V2) | `parallel_fedavg` | `sequential` |

A new split scheme is just another point on these two axes — no new orchestration code.

**Exactness.** The forward/backward relay is exact: training through the cut gives bitwise the same
weights as training the unsplit model.

**Costs.** `SplitCostModel` (`flsim/system/split_cost.py`) charges device computation, activation
upload, server computation (on a faster edge server, `split.server_cpu_frequency_hz`), gradient
download and the model exchange, on the same FDMA and energy formulas as the other paradigms. The
device/server compute split at the cut is measured from the model. Latency combines per variant: SL
sums over clients, SFLV1 takes the slowest, SFLV2 follows staged synchronization barriers. Split runs
log `round_latency_s`, `traffic_bytes` and energy alongside accuracy.

```python
from flsim.experiments.split_base import SplitExperiment

class MySplit(SplitExperiment):
    def run(self):
        sl    = self.run_single_split("sl",    client_mode="sequential",      server_mode="sequential")
        sflv1 = self.run_single_split("sflv1", client_mode="parallel_fedavg", server_mode="parallel_fedavg")
        sflv2 = self.run_single_split("sflv2", client_mode="parallel_fedavg", server_mode="sequential",
                                      cut_layer=4)
        self.plot_comparison({"SL": sl, "SFLV1": sflv1, "SFLV2": sflv2})

MySplit(base_config="flsim/configs/mnist_fedavg.yaml", output_dir="outputs/my_split/").run()
```

---

## The examples

### `examples/fedavg_experiment.py` — synchronous FedAvg

FedAvg on MNIST with the configuration in `flsim/configs/mnist_fedavg.yaml`: 30 clients with a
non-IID shard partition (2 shards each), 10 clients per round, an exponential-fading channel and
heterogeneous CPU frequencies. Command-line options `--rounds`, `--clients_per_round`,
`--local_epochs`, `--lr`, `--seed`. Output: `outputs/fedAVG/`.

### `examples/fl_vs_fedasync_experiment.py` — synchronous vs asynchronous FL

FedAvg against FedAsync (polynomial staleness) on ResNet-34 / CIFAR-10 at 64×64, under heterogeneous
devices and channels, at two data-heterogeneity levels (IID and Dirichlet α = 0.1) — four runs. Both
methods do the same local work per update (`H` mini-batch steps), so they are compared on simulated
time and cumulative energy. Produces accuracy-vs-round and accuracy-vs-time comparisons, staleness,
energy and time bars, and a time-to-accuracy table. A full run is a GPU job; `--smoke` shrinks it to 4
clients and 4 rounds. Output: `outputs/fl_vs_fedasync_experiment/`.

### `examples/ota_experiment.py` — over-the-air aggregation

Digital FedAvg (all devices, exact averaging, FDMA) against FedOTA at three aggregation-MSE budgets
(γ = −3, 5 and 10 dB) on MNIST with 30 devices: a tighter budget admits fewer devices with less noise,
a looser one more devices with more noise. Plots accuracy, cumulative energy, the achieved MSE and the
number of participating devices. The noise power and γ values are calibrated to this configuration's
channel gains (the method is in the script's header). Output: `outputs/ota_experiment/`.

### `examples/splitfed_experiment.py` — split learning across paradigms

Six paradigms on MNIST with 5 clients, all on the same cost model: Normal (centralized) training, FL
(FedAvg), FedAsync, SL, SplitFed V1 and SplitFed V2 (cut after the convolutional features). Compares
test accuracy against global epoch and simulated time, cumulative energy, per-round traffic and
latency, and best accuracy. FedAsync runs `clients × rounds` updates so every method does the same
total training work. Output: `outputs/splitfed_experiment/`.

---

## Writing your own experiments and algorithms

### A custom synchronous algorithm

Subclass `FederatedAlgorithm` (only `aggregate` is required) or an existing algorithm, and override
only what differs:

```python
from collections import OrderedDict
import torch
from flsim.algorithms.fedavg import FedAvg
from flsim.interfaces.algorithm import FederatedAlgorithm

class FedMedian(FederatedAlgorithm):                 # new aggregation rule
    def aggregate(self, global_model, client_updates) -> OrderedDict:
        return OrderedDict((k, torch.stack([u.state_dict[k].float() for u in client_updates]).median(0).values)
                           for k in client_updates[0].state_dict)

class NearestClients(FedAvg):                        # new selection rule, FedAvg aggregation
    def select_clients(self, all_clients, num_to_select, rng, **kwargs):
        return sorted(all_clients, key=lambda c: c.profile.distance_m)[:num_to_select]
```

`flsim/algorithms/fedprox.py` shows the third hook, `configure_client`, in five lines (it sets the
client's proximal coefficient).

### A custom asynchronous algorithm

Subclass `FedAsync` or `AsyncFederatedAlgorithm`:

```python
import math
from flsim.algorithms.fedasync import FedAsync

class ExponentialStaleness(FedAsync):               # only the staleness decay changes
    def __init__(self, alpha=0.5, lam=0.05):
        super().__init__(alpha=alpha)
        self.lam = lam

    def mixing_weight(self, base_alpha, staleness):
        return base_alpha * math.exp(-self.lam * staleness)
```

### An experiment

```python
from flsim.experiments.async_base import AsyncExperiment
from flsim.algorithms.fedavg import FedAvg
from flsim.algorithms.fedasync import FedAsync

class MyExperiment(AsyncExperiment):                # AsyncExperiment can run sync and async runs
    def run(self):
        fl = self.run_single("fedavg", label="FedAvg",
                             config_overrides={"learning.global_rounds": 100},
                             components={"algorithm": FedAvg()})
        fa = self.run_single_async("fedasync", label="FedAsync",
                                   config_overrides={"learning.global_rounds": 1000,
                                                     "async_fl.window_size": 10},
                                   components={"algorithm": FedAsync(alpha=0.5, staleness_func="polynomial")})
        self.plot_comparison({"FedAvg": fl, "FedAsync": fa}, plot_configs=[
            {"metric": "test_accuracy", "x": "simulated_time_s", "ylabel": "Test accuracy"},
            {"metric": "cumulative_energy_j", "x": "simulated_time_s", "ylabel": "Cumulative energy (J)"},
        ])

MyExperiment(base_config="flsim/configs/mnist_fedavg.yaml", output_dir="outputs/my_experiment/").run()
```

`Experiment.run_single`, `AsyncExperiment.run_single_async` and `SplitExperiment.run_single_split` all
return a `RunResult` (`final_accuracy`, `best_accuracy`, `total_energy_j`, `total_simulated_time_s`,
`avg_staleness`, `.df`), so runs from different paradigms go into the same `plot_comparison` /
`plot_bar` calls. For larger studies, `AlgorithmComparison` runs several algorithms on one
configuration and `ParameterSweep` varies one setting (e.g. `wireless.total_bandwidth_hz`).

### Other components

Every component has an interface in `flsim/interfaces/`; implement it and pass the instance through
`components=` (or register a name in `flsim/experiments/wiring.py` to select it from YAML):

- **Model** — any `nn.Module` with a `num_classes` argument; add `ordered_layers()` to make it
  splittable, and register it in `flsim/models/factory.py`.
- **Dataset** — a loader returning `(train_set, test_set)`; add it to `wiring._load_dataset()`.
- **Partitioner** (`DataPartitioner`), **channel** (`ChannelModel`), **time model** (`TimeModel`),
  **resource allocator** (`ResourceAllocator`).

---

## Configuration

`flsim/configs/base.yaml` lists and explains every option; `flsim/configs/mnist_fedavg.yaml` is the
compact configuration the MNIST examples start from. Any value can be overridden per run with dot
notation:

```python
config_overrides = {"learning.global_rounds": 50, "data.partition": "dirichlet", "data.dirichlet_alpha": 0.1}
```

| Section | Key options |
|---|---|
| `data` | `dataset` (mnist, cifar10, cifar100, ham10000), `model_name`, `image_size`, `num_clients`, `partition` (iid, shard, dirichlet), `dirichlet_alpha` |
| `learning` | `global_rounds`, `clients_per_round`, `local_epochs` or `local_iters` (H mini-batch steps), `batch_size`, `learning_rate`, `cut_layer`, `stop_by_time_s` |
| `system` | `cpu_freq_mode` (fixed, discrete_ghz, uniform_ghz) and range, `cycles_per_sample_mode` (model_macs, manual), `flops_per_cycle`, `switched_capacitance`, `energy_scope` (total, device) |
| `wireless` | `channel_model` (path_loss, exp_fading), `deployment_shape` and distances, `total_bandwidth_hz`, transmit power, `noise_psd_dbm_per_hz`, `min_snr_db`, `upload_size_mode`, downlink options |
| `async_fl` | `alpha`, `window_size`, `rho` (optional proximal weight) |
| `ota` | `p0_w`, `noise_power_w`, `gamma_db` |
| `split` | `server_cpu_frequency_hz`, `q_device`, `q_server`, `server_frequency_shared` |
| `evaluation`, `plots` | evaluation interval; which per-run plots are drawn |

**Local work.** `learning.local_iters: H` gives every client exactly `H` mini-batch steps per round in
every paradigm (`null` uses `local_epochs` full passes instead). Use the same setting for all methods
in a comparison: training and cost then cover identical work.

---

## Outputs and metrics

Every run writes, under `outputs/<experiment>/<run_name>/`:

```
<run_name>.csv           the orchestrator's own log (one row per round, or per update for async)
<run_name>_unified.csv   the same run in a schema identical for every paradigm
<run_name>_*.png         standard plots: accuracy vs round and vs time, losses, energy, traffic, latency
```

The unified columns are `round, train_loss, test_accuracy, test_loss, simulated_time_s,
round_latency_s, avg_waiting_time_s, traffic_bytes, cumulative_traffic_bytes, total_energy_j,
cumulative_energy_j` (columns a paradigm does not have are empty and their plots are skipped).
Paradigm-specific columns stay in the raw CSV: `staleness` and `alpha_used` for async runs, per-client
compute/upload times and rates for synchronous runs.

| Metric | Meaning |
|---|---|
| `test_accuracy`, `test_loss` | global model on the test set, at evaluation rounds |
| `simulated_time_s` | cumulative simulated wall-clock time (sync: sum of round durations; async: arrival time of each update) |
| `round_duration_s` / `round_latency_s` | duration of one round (sync: slowest selected client) |
| `total_energy_j`, `cumulative_energy_j` | energy of the step and running total — use the cumulative value to compare paradigms |
| `traffic_bytes` | bytes moved in the step (split learning) |
| `staleness`, `alpha_used` | async only: model versions behind, and the mixing weight used |

`python plot_results.py <run>.csv [--out <folder>]` redraws the per-round plots (accuracy, loss, timing
and energy breakdown, channel gain, rate) from a saved raw CSV. It draws every plot for synchronous runs
(FedAvg, FedOTA) and skips the plots whose columns a FedAsync or split-learning CSV does not have; those
runs get their full plot set when the experiment runs.

---

## Tests

```bash
pytest            # flsim/tests: channel, partitions, time/energy models, FedAvg
```
