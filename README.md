# Boosting Reinforcement Learning for Critical Node Detection with Multi-Topology Training Strategies

A Deep Q-Network (DQN) agent with Graph Attention Networks (GAT) that learns to identify **critical nodes** — the minimal set of nodes whose removal maximally disrupts a network's connectivity.

## How it works

The agent treats network dismantling as a sequential decision problem:

1. **State**: node-level centrality features (degree, eigenvector, PageRank, closeness index) + graph-level features (heterogeneity, resilience, entropy)
2. **Action**: select and remove one node per step
3. **Reward**: weighted combination of reduction in the Largest Connected Component (LCC) and the Molloy–Reed robustness criterion
4. **Model**: GAT (Graph Attention Network v2) encodes the graph; a DQN with n-step learning and experience replay selects the best node to remove

Training uses randomly generated synthetic graphs so the agent generalises to unseen networks at inference time.

---

## Installation

### 1. Create a Python environment

```bash
conda create -n graph-dismantle python=3.10
conda activate graph-dismantle
```

### 2. Install PyTorch

Choose the command matching your hardware from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

**CPU only:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 12.1:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch Geometric

PyTorch Geometric must match your PyTorch version. See the [official install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

```bash
pip install torch_geometric
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Training generates synthetic graphs on-the-fly, so no dataset download is required.

```bash
python main.py -i configs/ba_minmax.json
```

To set a specific random seed (for reproducibility):

```bash
python main.py -i configs/ba_minmax.json -s 42
```

### Available configs

| Config | Graph type | Objective |
|--------|-----------|-----------|
| `configs/ba_minmax.json` | Barabasi–Albert | Minimize LCC |
| `configs/er_minmax.json` | Erdős–Rényi | Minimize LCC |
| `configs/mixed_minmax.json` | BA + ER + SW + Power-law | Minimize LCC (best for generalization) |

### Training outputs

| Path | Description |
|------|-------------|
| `model/<name>/checkpoint_best.pt` | Best model checkpoint (lowest validation AUC) |
| `model/<name>/checkpoint_latest.pt` | Most recent checkpoint (for resuming) |
| `log/<name>_training.json` | JSONL training log (episode, loss, reward, AUC) |

**Resuming**: Training resumes automatically from `checkpoint_latest.pt` if it exists.

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_episodes` | 500 000 | Total training episodes. Reduce to ~10 000 for a quick smoke-test. |
| `alpha` | 0.5 | Reward weight: `alpha × ΔLCC + (1-alpha) × ΔMolloy-Reed` |
| `gnn_model` | `"GAT"` | GNN backbone: `"GAT"`, `"SAGE"`, or `"GCN"` |
| `epsilon_decay_duration` | 20 000 000 | Steps over which ε decays from 1.0 → 0.01 |
| `hidden_layers` | `[[5,3],[2,3],[2,2,1]]` | Sizes of [conv layers, global MLP, output MLP] |
| `objective_function` | `"largestConnectedComponent"` | Also: `"pairwiseConnectivity"`, `"numberConnectedComponent"` |

Edit any config in `configs/` to change these.

---

## Inference

Run a trained model on any graph provided as a space-separated edge list:

```bash
python infer.py \
  -i configs/ba_minmax.json \
  -m model/ba_lcc/checkpoint_best.pt \
  -g path/to/your_graph.txt
```

### Graph file format

Plain text, one edge per line, space-separated node IDs. Comments (`#`) and header lines are ignored:

```
# my_graph.txt
0 1
0 2
1 3
2 3
3 4
```

### Inference output

```
Graph loaded: 50 nodes, 120 edges
Model loaded from: model/ba_lcc/checkpoint_best.pt
Running inference...

==================================================
RESULTS
==================================================
  Graph:                    my_graph.txt
  Nodes / Edges:            50 / 120
  Nodes removed:            12 (24.0% of graph)
  AUC:                      0.3142  (lower = faster dismantling)
  Critical point (slope):   0.2400
  Critical point (curv.):   0.2200

  Node removal order (original IDs):
  [7, 3, 15, 2, ...]
==================================================
```

---

## Project structure

```
.
├── main.py                  # Training entry point
├── infer.py                 # Inference entry point
├── requirements.txt         # Python dependencies
├── configs/                 # Hyperparameter configs (edit these)
│   ├── ba_minmax.json          # Barabasi-Albert, minimize LCC
│   ├── er_minmax.json          # Erdos-Renyi, minimize LCC
│   └── mixed_minmax.json       # Mixed graph types, minimize LCC
└── utils/
    ├── params.py            # Config loader
    ├── validation.py        # Validation graph generation + AUC
    ├── getClass.py          # Dynamic class loader
    ├── environment/
    │   ├── game.py          # GraphGame and GraphState (RL env)
    │   ├── envhelper.py     # Graph generation, reset, Molloy-Reed
    │   ├── nodeCentrality.py  # Node feature extraction
    │   ├── globalFeature.py   # Graph-level feature extraction
    │   └── objectiveFunction.py  # LCC, pairwise connectivity, etc.
    ├── reinforcement_learning/
    │   ├── dqn_TS.py        # DQN agent with n-step learning
    │   ├── GraphNN.py       # GAT / SAGE / GCN backbone
    │   ├── replay_buffer.py # Experience replay buffer
    │   ├── rl_environment.py
    │   └── rl_agent.py
    └── evaluation/
        └── evaluationhelper.py  # EvaluateModel, EvaluateACTION
```

---

## Objective functions

| Name | Effect |
|------|--------|
| `largestConnectedComponent` | Agent tries to shrink the giant component as fast as possible |
| `pairwiseConnectivity` | Minimizes the number of reachable node pairs |
| `numberConnectedComponent` | Maximizes the number of disconnected components |

Set `objective_function` in your config to switch between them.
