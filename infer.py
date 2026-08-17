#!/usr/bin/env python3
"""
Inference script: run a trained DQN model on a graph to identify critical nodes.

Usage:
    python infer.py -i <config.json> -m <model.pt> -g <graph.txt>

Arguments:
    -i / --config   Path to the training config JSON (same one used for training)
    -m / --model    Path to the model checkpoint (.pt file, e.g. checkpoint_best.pt)
    -g / --graph    Path to the graph edge-list file (.txt, space-separated node pairs)

Example:
    python infer.py -i configs/ba_lcc.json -m model/ba_lcc/checkpoint_best.pt -g my_graph.txt

Output:
    - Number of nodes and edges
    - Order in which critical nodes are removed
    - Fraction of nodes removed to dismantle the network
    - AUC (lower is better — the agent dismantled faster)
    - Critical point estimates (the fraction where the network collapses)
"""

import sys
import getopt
import os

import torch
import numpy as np

from utils.environment.game import GraphGame
from utils.reinforcement_learning.rl_environment import Environment
from utils.reinforcement_learning.dqn_TS import DQN
from utils.params import Params
from utils.getClass import objective_function, get_class_from_file
from utils.environment.nodeCentrality import Node_Centrality
from utils.environment.globalFeature import Global_Feature
from utils.evaluation.evaluationhelper import EvaluateModel
from utils.validation import area_under_curve
from main import BenchMark, estimate_pc_max_slope, estimate_pc_max_curvature


def main(argv):
    config_path = ""
    model_path = ""
    graph_path = ""

    try:
        opts, _ = getopt.getopt(argv, "hi:m:g:", ["config=", "model=", "graph="])
    except getopt.GetoptError:
        print("Usage: python infer.py -i <config.json> -m <model.pt> -g <graph.txt>")
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(__doc__)
            sys.exit()
        elif opt in ("-i", "--config"):
            config_path = arg
        elif opt in ("-m", "--model"):
            model_path = arg
        elif opt in ("-g", "--graph"):
            graph_path = arg

    if not config_path or not model_path or not graph_path:
        print("Error: all three arguments are required.")
        print("Usage: python infer.py -i <config.json> -m <model.pt> -g <graph.txt>")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"Error: model checkpoint not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(graph_path):
        print(f"Error: graph file not found: {graph_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config and build feature extractors / objective
    bm = BenchMark(config_path)

    # Load graph
    print(f"Loading graph from: {graph_path}")
    G, node_map = bm.input_graph(graph_path)
    N = G.vcount()
    E = G.ecount()
    print(f"Graph loaded: {N} nodes, {E} edges")

    # Build agent and load checkpoint
    GNN_class = get_class_from_file(bm.params.GNN[0], bm.params.GNN[1])
    agent = DQN(
        gnn_model=bm.params.gnn_model,
        state_representation_size=bm.params.centrality_features,
        global_feature_size=bm.params.global_features,
        hidden_layers_sizes=bm.params.hidden_layers,
        GraphNN=GNN_class,
    )
    checkpoint = torch.load(model_path, map_location=device)
    agent._q_network.load_state_dict(checkpoint["_q_network"])
    agent._q_network.to(device)
    agent._q_network.eval()
    print(f"Model loaded from: {model_path}")

    # Run inference
    print("Running inference...")
    game = GraphGame
    env = Environment(game)
    with torch.no_grad():
        rewards, value, actions = EvaluateModel(
            env,
            bm.objectiveFunction,
            bm.nodeCentrality,
            bm.globalFeature,
            bm.alpha,
            agent,
            G,
        )

    # Compute metrics
    x = np.flip(np.arange(N)[N::-1] / N)
    auc = area_under_curve(bm.condMaxNum, N, x[: len(value)], value)
    fraction = len(actions) / N
    pc_slope = estimate_pc_max_slope(value, maxnum_condition=bm.condMaxNum)
    pc_curvature = estimate_pc_max_curvature(value, maxnum_condition=bm.condMaxNum)

    # Map internal indices back to original node IDs
    reverse_map = {v: k for k, v in node_map.items()}
    original_order = [reverse_map.get(a, a) for a in actions]

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"  Graph:                    {graph_path}")
    print(f"  Nodes / Edges:            {N} / {E}")
    print(f"  Nodes removed:            {len(actions)} ({fraction * 100:.1f}% of graph)")
    print(f"  AUC:                      {auc:.4f}  (lower = faster dismantling)")
    print(f"  Critical point (slope):   {pc_slope['pc']:.4f}")
    print(f"  Critical point (curv.):   {pc_curvature['pc']:.4f}")
    print(f"\n  Node removal order (original IDs):")
    print(f"  {original_order}")
    print("=" * 50)

    return {
        "auc": auc,
        "fraction_removed": fraction,
        "critical_point_slope": pc_slope["pc"],
        "critical_point_curvature": pc_curvature["pc"],
        "removal_order": original_order,
        "objective_values": list(value),
    }


if __name__ == "__main__":
    main(sys.argv[1:])
