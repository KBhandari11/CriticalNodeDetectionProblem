#!/usr/bin/env python3
"""
Training entry point for the Graph Dismantling DQN agent.

Usage:
    python main.py -i <config.json> [-s <seed>]

Arguments:
    -i / --ifile    Path to the hyperparameter config JSON (see configs/)
    -s / --seed     Random seed (default: 1324567)

Example:
    python main.py -i configs/ba_lcc.json -s 42
"""
import torch
import numpy as np
from pathlib import Path
import json
import os, sys, getopt
import random
import gc
from multiprocessing import Pool

from utils.environment.game import GraphGame
from utils.environment.nodeCentrality import Node_Centrality
from utils.environment.globalFeature import Global_Feature
from utils.reinforcement_learning.rl_environment import Environment
from utils.validation import *
from utils.evaluation import jsonEncoder
from utils.evaluation.evaluationhelper import *
from utils.environment.envhelper import gen_new_graphs
from utils.reinforcement_learning.dqn_TS import DQN
from utils.params import Params
from utils.getClass import objective_function, get_class_from_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def _generate_one_graph(args):
        graph_type, seed = args
        return gen_new_graphs(graph_type=graph_type, seed=seed)

def _load_meta(checkpoint_base):
    """
    Load training metadata for resuming:
    - checkpoint_base: base path (e.g., './checkpoints/run1')
    Returns a dict with fields:
      last_episode, best_auc, best_ep, last_epsilon
    """
    meta_path = f"{checkpoint_base}_meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            # fill defaults if missing
            meta.setdefault("last_episode", 0)
            meta.setdefault("best_auc", float("inf"))
            meta.setdefault("best_ep", None)
            meta.setdefault("last_epsilon", None)
            return meta
        except Exception as e:
            print(f"[Warning] Failed to read meta file {meta_path}: {e}")
    # Default if no meta file found
    return {"last_episode": 0, "best_auc": float("inf"), "best_ep": None, "last_epsilon": None}


def _read_last_log_entry(path):
    """
    Read and print all lines from a JSONL training log file,
    and extract the most recent episode and epsilon.
    Returns (last_episode, last_epsilon).
    """

    if not os.path.exists(path):
        print(f"[Info] No log file found at {path}")
        return 0, None

    try:
        with open(path, "r") as f:
            lines = f.readlines()

        print(f"\n[Debug] ===== Printing all log entries from {path} =====")
        for i, line in enumerate(lines):
            if (i+1)%100==0:
                print(f"Line {i+1}: {line.strip()}")
        print("[Debug] ===== End of log file =====\n")

        if not lines:
            print("[Info] Log file is empty.")
            return 0, None

        # Parse last line
        last_line = lines[-1].strip()
        if not last_line:
            print("[Warning] Last line is empty.")
            return 0, None

        rec = json.loads(last_line)
        last_ep = int(rec.get("episode", 0))
        last_eps = float(rec.get("epsilon")) if rec.get("epsilon") is not None else None
        print(f"[Debug] Last recorded episode: {last_ep}, epsilon: {last_eps}")
        return last_ep, last_eps

    except Exception as e:
        print(f"[Warning] Could not read or parse log file {path}: {e}")
        return 0, None

def _save_meta(checkpoint_base, meta):
    meta_path = f"{checkpoint_base}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f)

def _moving_average(x, w):
    w = int(max(1, w))
    if w == 1:
        return x
    kernel = np.ones(w) / w
    # pad to reduce edge shrink
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xpad, kernel, mode="valid")
def estimate_pc_max_curvature(y, p=None,  maxnum_condition=False, smooth_window=9):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if p is None:
        p = np.linspace(0.0, 1.0, n)
    else:
        p = np.asarray(p, dtype=float)

    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-12)
    ys = _moving_average(y_norm, smooth_window)

    d1 = np.gradient(ys, p)
    d2 = np.gradient(d1, p)

     # 4) pick critical point
    if not maxnum_condition:
        idx = int(np.argmax(np.abs(d2)))   # largest drop magnitude (or just abs)
    elif maxnum_condition:
        idx = int(np.argmax(d2))           # strongest rise

    return {"pc":float(p[idx]),"p": p, "y_smooth": ys, "d1": d1, "d2": d2, "idx": idx}

def estimate_pc_max_slope(y, p=None, maxnum_condition=False, smooth_window=9):
    """
    y: list/np.array of objective values along an increasing p
    p: same length as y; if None, assumes evenly spaced [0,1]
    obj: 'largestConnectedComponent' | 'pairwiseConnectivity' | 'numberConnectedComponent'
    smooth_window: odd-ish window size for moving average smoothing
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if p is None:
        p = np.linspace(0.0, 1.0, n)
    else:
        p = np.asarray(p, dtype=float)
        assert len(p) == n

    # 1) normalize (helps compare scales)
    y_norm = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-12)

    # 2) smooth
    ys = _moving_average(y_norm, smooth_window)

    # 3) derivative dy/dp (centered via numpy gradient)
    dydp = np.gradient(ys, p)

    # 4) pick critical point
    if not maxnum_condition:
        idx = int(np.argmax(np.abs(dydp)))   # largest drop magnitude (or just abs)
    elif maxnum_condition:
        idx = int(np.argmax(dydp))           # strongest rise

    pc = float(p[idx])
    return{"pc": pc,"p": p, "y_norm": y_norm, "y_smooth": ys, "dydp": dydp, "idx": idx}

class BenchMark():
    def __init__(self,path):
        #path: relative path to the parameter  
        self.params = Params(path)
        self.nodeCentrality = Node_Centrality(self.params.centrality_feature_name)
        self.globalFeature = Global_Feature(self.params.global_feature_name)
        self.objectiveFunction = objective_function("utils/environment/objectiveFunction.py",self.params.objective_function)
        self.GNN = get_class_from_file(self.params.GNN[0], self.params.GNN[1])
        self.condMaxNum = self.objectiveFunction.__name__ == "numberConnectedComponent"
        self.alpha = self.params.alpha 
    

    def Generate_Batch_Graph(self, size, seed=None):
        args = [(self.params.graph_type, seed + i) for i in range(size)]
        with Pool(processes=min(os.cpu_count(), 4)) as pool:
            batch_graphs = pool.map(_generate_one_graph, args)
        return np.array(batch_graphs, dtype=object)

    def validation(self,size,env, agent, evaluation, x):
        AUC = []
        with torch.no_grad():
            for i in range(size):
                g = evaluation[i].copy()
                eval_step = env.reset(g,self.objectiveFunction,self.nodeCentrality,self.globalFeature,self.alpha)
                while not eval_step.last():
                    eval_output, prob = agent.step(eval_step, is_evaluation=True)
                    eval_step = env.step(eval_output)
                gamma = env.get_state.reward1
                N = env.get_state.Graph.vcount()
                AUC.append(area_under_curve(self.condMaxNum,N,x[i][:len(gamma)],gamma))

                del g

        meanAUC = np.mean(AUC)
        return meanAUC

    def bestModel(self, allEval):
        start = int(self.params.save_every)
        end = int(self.params.num_train_episodes)+start
        interval = int(self.params.save_every)
        model = np.arange(start, end, interval) 
        x = np.argmin(allEval)
        best = model[x]
        self.best_dir  = self.params.checkpoint_dir+"_"+str(best)

    def input_graph(self, graph_path):
        src = Path(graph_path)
        dst = src.with_suffix(src.suffix + ".clean")

        with src.open("r") as fin, dst.open("w") as fout:
            for line in fin:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    fout.write(f"{parts[0]} {parts[1]}\n")

        GRAPH = Graph.Read_Ncol(str(dst), directed=False)
        os.remove(dst)

        nodes = [v.index for v in GRAPH.vs]
        map = {n:int(i) for i, n in enumerate(nodes)}
        GRAPH = reset(GRAPH)  
        Graph.simplify(GRAPH)
        return GRAPH, map

    def train(self,agent=None):
        game = GraphGame
        env = Environment(game)
        size_CV = self.params.validation_test_size
        evaluation, eval_x = get_Validation(size_CV)#get_Validation(4,file_path)
        CV_AUC = []
        log_path = self.params.training_log
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # ---- Resume state (from meta + log + latest checkpoint) ----
        meta = _load_meta(self.params.checkpoint_dir)
        last_logged_ep, last_log_epsilon = _read_last_log_entry(log_path)

        # Throttling: how often to save the "latest" checkpoint
        save_latest_every = int(
            getattr(self.params, "save_latest_every",
                    getattr(self.params, "eval_every", 500))
        )
        save_latest_every = max(1, save_latest_every)  # safety

        ep_start = 0
        best_auc = float("inf")
        best_ep = None
        
        # Determine resume epsilon
        resume_epsilon = None
        if meta.get("last_epsilon") not in [None, "None"]:
            try:
                resume_epsilon = float(meta["last_epsilon"])
            except Exception:
                resume_epsilon = None

        if (resume_epsilon is None or resume_epsilon <= 0.0) and last_log_epsilon is not None:
            resume_epsilon = float(last_log_epsilon)

        # Default if still None
        if resume_epsilon is None:
            resume_epsilon = float(self.params.epsilon_start)
        else:
            print(f"[Resume] Using saved epsilon_start={resume_epsilon:.6f}")

        # Override epsilon_start dynamically
        self.params.epsilon_start = resume_epsilon

        if agent == None:
            agent = DQN(
                    gnn_model = self.params.gnn_model,
                    state_representation_size=self.params.centrality_features,
                    global_feature_size = self.params.global_features,
                    hidden_layers_sizes=self.params.hidden_layers,
                    replay_buffer_capacity=int(self.params.replay_buffer_capacity),
                    learning_rate=self.params.learning_rate,
                    update_target_network_every=  self.params.update_target_network_every,
                    learn_every=self.params.learn_every,
                    discount_factor=self.params.discount_factor,
                    min_buffer_size_to_learn=self.params.min_buffer_size_to_learn,
                    power = self.params.epsilon_power,
                    nsteps=self.params.nstep,
                    epsilon_start=self.params.epsilon_start,
                    epsilon_end=self.params.epsilon_end,
                    epsilon_decay_duration=self.params.epsilon_decay_duration,
                    batch_size=self.params.batch_size,
                    GraphNN = self.GNN)
        #agents.append(random_agent.RandomAgent(player_id=1, num_actions=num_actions))
        graph_batch_size = self.params.graph_batch_size
        latest_ckpt_path = f"{self.params.checkpoint_dir}_latest.pt"
        best_ckpt_path   = f"{self.params.checkpoint_dir}_best.pt"

        if os.path.exists(latest_ckpt_path):
            try:
                ckpt = torch.load(latest_ckpt_path, map_location="cpu")
                agent._q_network.load_state_dict(ckpt["_q_network"])
                agent._target_q_network.load_state_dict(ckpt["target_q_network"])
                agent._optimizer.load_state_dict(ckpt["_optimizer"])
                # Epsilon: prefer meta, fall back to last log, otherwise keep agent default
                if meta.get("last_epsilon") is not None:
                    agent.epsilon = meta["last_epsilon"]
                    agent.epsilon_start = last_log_epsilon 
                elif last_log_epsilon is not None:
                    agent.epsilon = last_log_epsilon
                    agent.epsilon_start = last_log_epsilon 

                ep_start = int(max(meta.get("last_episode", 0), last_logged_ep))
                best_auc = float(meta.get("best_auc", float("inf")))
                best_ep = meta.get("best_ep", None)
                print(f"[Resume] Loaded latest checkpoint at episode {ep_start}, epsilon={agent.epsilon}, best_auc={best_auc}, best_ep={best_ep}")
            except Exception as e:
                print(f"[Resume] Failed to load latest checkpoint ({e}). Starting fresh.")
        else:
            print("[Resume] No latest checkpoint found. Starting fresh.")

        #pregenerate it if needed
        # ----- Pre-generate batch when resuming -----
        # If resuming mid-batch, regenerate the correct batch deterministically.
        if ep_start == 0:
            if (ep_start) % self.params.graph_suffle == 0:
                Batch_Graph = self.Generate_Batch_Graph(graph_batch_size, seed=ep_start)
        else:
            # Determine last shuffle boundary to regenerate the correct batch
            last_shuffle_seed = int((ep_start // self.params.graph_suffle) * self.params.graph_suffle)
            print(f"[Resume] Regenerating graph batch for shuffle seed {last_shuffle_seed}")
            Batch_Graph = self.Generate_Batch_Graph(graph_batch_size, seed=last_shuffle_seed)
        
        num_eps = int(self.params.num_train_episodes)

        for ep in range(ep_start, num_eps):
            if (ep) % self.params.graph_suffle == 0:
                Batch_Graph = self.Generate_Batch_Graph(graph_batch_size,seed=ep)
            g = Batch_Graph[int(ep%graph_batch_size)].copy()
            time_step = env.reset(g,self.objectiveFunction,self.nodeCentrality,self.globalFeature,self.alpha)
            while not time_step.last():
                action, prob = agent.step(time_step) 
                time_step = env.step(action)
            # Episode is over, step agent with final info state.
            agent.step(time_step)

            episode_log = {"episode": ep + 1, 
                            "loss": float(agent._last_loss_value) if agent._last_loss_value != None else None , 
                            "reward": float(env.get_state._returns), 
                            "epsilon": float(agent.epsilon),
                            "eval_auc": None
                        }

            # Periodic evaluation
            if (ep + 1) % self.params.eval_every == 0:
                _eps_cache = float(agent.epsilon) 
                meanAUC = self.validation(size_CV, env, agent, evaluation, eval_x)
                episode_log["eval_auc"] = float(meanAUC)
                print(f"Evaluation_Dataset: {meanAUC}", end=" ", flush=True)
                agent.epsilon = _eps_cache  
                # Save BEST immediately (not throttled)
                if meanAUC < best_auc:
                    best_auc = float(meanAUC)
                    best_ep = ep + 1
                    best_checkpoint = {
                        "_q_network": agent._q_network.state_dict(),
                        "target_q_network": agent._target_q_network.state_dict(),
                        "_optimizer": agent._optimizer.state_dict(),
                    }
                    torch.save(best_checkpoint, best_ckpt_path)
                    print(f"\n[Checkpoint] BEST @ ep {best_ep} | AUC {best_auc:.6f}")

            # Append to log
            with open(log_path, "a") as f:
                f.write(json.dumps(episode_log) + "\n")

                
            # ---------- Throttled saving of LATEST checkpoint ----------
            if ((ep + 1) % save_latest_every == 0) or ((ep + 1) == num_eps):
                latest_checkpoint = {
                    "_q_network": agent._q_network.state_dict(),
                    "target_q_network": agent._target_q_network.state_dict(),
                    "_optimizer": agent._optimizer.state_dict(),
                }
                torch.save(latest_checkpoint, latest_ckpt_path)

                # Update meta only on save to keep disk I/O low
                meta["last_episode"] = ep + 1
                meta["last_epsilon"] = float(agent.epsilon)
                meta["best_auc"] = best_auc
                meta["best_ep"] = best_ep
                _save_meta(self.params.checkpoint_dir, meta)

            # Periodic prints
            if (ep + 1) % 100 == 0:
                print(f"episode: {ep}, loss: {agent._last_loss_value}, cummulative_reward:{env.get_state._returns}, epsilon:{agent.epsilon}",flush=True)
                gc.collect()

            del g

        # Finalize: choose best_dir
        if best_ep is not None and os.path.exists(best_ckpt_path):
            print(f"\nBest model: ep {best_ep} | AUC {best_auc:.4f} | path: {best_ckpt_path}")
            self.best_dir = best_ckpt_path
        else:
            print("\nNo best improvement recorded; using latest for best_dir.")
            self.best_dir = latest_ckpt_path


    def evaluation(self, graphpath, bestModel=None, useSingleStep= None):
        """
        Given a path to an edgelist of the Graph and a model, it evaluates the removal of node based on the objective function. 
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        G , _ = self.input_graph(graphpath)
        if bestModel == None:
            bestModel = self.best_dir 
        model = torch.load(self.params.checkpoint_dir+"_"+bestModel)
        N = G.vcount()
        game = GraphGame
        env = Environment(game)
        attacker = DQN( gnn_model = self.params.gnn_model,
                        state_representation_size=self.params.centrality_features,
                        hidden_layers_sizes= self.params.hidden_layers,
                        global_feature_size = self.params.global_features)
        attacker._q_network.to(device)
        if hasattr(attacker, "_target_network") and attacker._target_network is not None:
            attacker._target_network.to(device)
        attacker._q_network.load_state_dict(model["_q_network"])
        attacker._q_network.eval()
        rewards, value, actions = EvaluateModel(env, self.objectiveFunction, self.nodeCentrality, self.globalFeature,self.alpha, attacker, G,useSingleStep)
        x =  np.flip(np.arange(N)[N::-1]/N)
        auc = area_under_curve(self.condMaxNum,N,x[:len(value)],value)
        fraction = len(actions)/N
        pc_slope = estimate_pc_max_slope(value,maxnum_condition=self.condMaxNum )
        pc_curve = estimate_pc_max_curvature(value,maxnum_condition=self.condMaxNum)
        return auc, fraction, pc_slope, pc_curve

    def evaluationReward(self, graphpath, bestModel=None, useSingleStep= None):
        """
        Given a path to an edgelist of the Graph and a model, it evaluates the removal of node based on the objective function. 
        """
        G , _ = self.input_graph(graphpath)
        if bestModel == None:
            bestModel = self.best_dir 
        model = torch.load(self.params.checkpoint_dir+"_"+bestModel)
        N = G.vcount()
        game = GraphGame
        env = Environment(game)
        attacker = DQN( gnn_model = self.params.gnn_model,
                        state_representation_size=self.params.centrality_features,
                        hidden_layers_sizes= self.params.hidden_layers,
                        global_feature_size = self.params.global_features)
        attacker._q_network.load_state_dict(model["_q_network"])
        attacker._optimizer.load_state_dict(model["_optimizer"])
        rewards, value, actions = EvaluateModel(env, self.objectiveFunction, self.nodeCentrality, self.globalFeature,self.alpha, attacker, G,useSingleStep)
        x =  np.flip(np.arange(N)[N::-1]/N)
        auc = area_under_curve(self.condMaxNum,N,x[:len(value)],value)
        fraction = len(actions)/N
        return auc, fraction,env
 
    def evaluationOthers(self, graphpath, actionspath, useIndex =False):
        """
        Given a path to an edgelist of a Graph, the list of action sequences, "actions" in order, and an objectiveFunction it evaluates the removal of node based on the objective function. 
        """
        import warnings
        G , _ = self.input_graph(graphpath)
        N = G.vcount()
        fname = actionspath+graphpath.split("/")[-1]
        if os.path.exists(fname):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actions = np.loadtxt(fname, dtype=int, unpack=True)
                #print(actions,actions.shape,type(actions))
                if  len(actions.shape)==0 :
                    actions= np.array([actions])
            if len(actions)!= 0 :
                rewards, value, actions = EvaluateACTION(actions,self.objectiveFunction,G, useIndex)
                x =  np.flip(np.arange(N)[N::-1]/N)
                value = np.array(value)/value[0]
                auc = area_under_curve(self.condMaxNum,N,x[:len(value)],value)
                fraction = len(actions)/N
                pc_slope = estimate_pc_max_slope(value,maxnum_condition=self.condMaxNum )
                pc_curve = estimate_pc_max_curvature(value,maxnum_condition=self.condMaxNum)
                return value,auc, fraction, pc_slope, pc_curve
            else:
                print("No actions")
                return None,None, None, None, None
        else:
            print("Action file doesn't exist!")
            return None,None, None, None, None

    def compareBenchmarkSynthetic(self,evaluation,bestmodel, useIndex = False):
        """
        Evaluate Graph on Synthetic Graph. 
        """
        syntheticGraph = {"Degree":"./Dataset/SyntheticGraph/Degree/",
                    "Homogeneity":"./Dataset/SyntheticGraph/Homogeneity/", 
                   }
        result = {"Degree": { "barabasi" : {"auc": [], "fraction": []},
                           "erdos" : {"auc": [], "fraction": []},
                           "small-world" : {"auc": [], "fraction": []}
                           },
                  "Homogeneity": { "barabasi" : {"auc": [], "fraction": []},
                           "erdos" : {"auc": [], "fraction": []},
                           "small-world" : {"auc": [], "fraction": []}
                           }
                 }
        for folder in syntheticGraph:
            graphlist = os.listdir(syntheticGraph[folder]) 
            for graphname in graphlist:
                name = graphname.split("_")[0]
                graphpath = syntheticGraph[folder] + graphname
                value = evaluation(graphpath,bestmodel,useIndex = useIndex)
                if value[0] != None:
                    result[folder][name]["auc"].append(value[0])
                    result[folder][name]["fraction"].append(value[1])
        self.result_Syn = result
        
    def compareBenchmarkSyntheticMotif(self,evaluation,bestmodel, useIndex = False):
        """
        Evaluate Graph on Synthetic Graph with Motifs attached. 
        """
        syntheticMotifGraph = {"BA":"./Dataset/Motifs_Attached/New_BA/",
                    "Tree":"./Dataset/Motifs_Attached/New_Tree/", 
                   }
        result = {"BA": { "cycle" : {"auc": [], "fraction": []},
                                  "clique" : {"auc": [], "fraction": []},
                                  "house" : {"auc": [], "fraction": []},
                                  "grid" : {"auc": [], "fraction": []},
                                  "star" : {"auc": [], "fraction": []},
                                  "fan" : {"auc": [], "fraction": []},
                                  "diamond" : {"auc": [], "fraction": []}
                                },
                 "Tree": { "cycle" : {"auc": [], "fraction": []},
                                  "clique" : {"auc": [], "fraction": []},
                                  "house" : {"auc": [], "fraction": []},
                                  "grid" : {"auc": [], "fraction": []},
                                  "star" : {"auc": [], "fraction": []},
                                  "fan" : {"auc": [], "fraction": []},
                                  "diamond" : {"auc": [], "fraction": []}
                                }
                 }
        for folder in syntheticMotifGraph:
            graphlist = os.listdir(syntheticMotifGraph[folder])
            #graphlist.remove(".ipynb_checkpoints")
            for graphname in graphlist:
                name = graphname.split("_")[3]
                graphpath = syntheticMotifGraph[folder] + graphname
                if os.path.exists(graphpath):
                    value = evaluation(graphpath,bestmodel,useIndex = useIndex)
                    if value[0] != None:
                        result[folder][name]["auc"].append(value[0])
                        result[folder][name]["fraction"].append(value[1])
                '''for graph in result[folder]:
                    for measure in result[folder][graph]:
                        plt.plot(result[folder][graph][measure])
                        plt.title(folder+" :"+measure)
                        plt.show()'''
        self.result_SynMotif = result

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(argv):
    seed=1324567
    seed_condition= False 

    hyperparameter_path = ''
    opts, args = getopt.getopt(argv,"hi:s:",["ifile=", "seed="])
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <HyperParameter JSON file path> -s <seed number>')
            print("Example:")
            print ('model.py -i ./utils/hyperparameters/BA/ba_params.json -s 1324567')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            hyperparameter_path = arg
        elif opt in ("-s", "--seed"):
            seed = int(arg)
            seed_condition = True
        else:
            print("Error in the argument. Use -h for help")
    set_random_seed(seed=seed)
    bm = BenchMark(hyperparameter_path)
    if seed_condition:
        bm.params.checkpoint_dir = bm.params.checkpoint_dir + f"_{seed}"
        bm.params.training_log = bm.params.training_log.replace(".json", f"_{seed}.json")
    bm.train()
 
if __name__ == "__main__":
   main(sys.argv[1:])