#!/usr/bin/env python3
"""DQN agents trained on Breakthrough by independent Q-learning."""
from utils.environment.game import GraphGame
from utils.environment.nodeCentrality import Node_Centrality
from utils.environment.globalFeature import Global_Feature
from utils.reinforcement_learning.rl_environment import Environment
from utils.validation import *
from utils.evaluation import jsonEncoder
from utils.evaluation.evaluationhelper import *
from utils.environment.envhelper import gen_new_graphs
from utils.reinforcement_learning.dqn_TS import DQN
from utils.hyperparameters.params import Params
from utils.getClass import objective_function,get_class_from_file
from tqdm import tqdm
import torch
import numpy as np
import wandb
import os, sys, getopt

class BenchMark():
    def __init__(self,path):
        #path: relative path to the parameter  
        self.params = Params(path)
        self.nodeCentrality = Node_Centrality(self.params.centrality_feature_name)
        self.globalFeature = Global_Feature(self.params.global_feature_name)
        self.objectiveFunction = objective_function("utils/environment/objectiveFunction.py",self.params.objective_function)
        self.GNN = get_class_from_file(self.params.GNN[0], self.params.GNN[1])
        os.environ['WANDB_NOTEBOOK_NAME'] = self.params.wandb_notebook_name
        wandb.login()
            
    def Generate_Batch_Graph(self, size,seed = None):
        Batch_Graph = [gen_new_graphs(graph_type=self.params.graph_type,seed=seed+i) for i in range(size)]
        return np.array(Batch_Graph,dtype=object)

    def validation(self,size,env, agent, evaluation, x):
        AUC = []
        if self.objectiveFunction.__name__ == "numberConnectedComponent":
            condMaxNum = True
        else:
            condMaxNum = False
        for i in range(size):
            eval_step = env.reset(evaluation[i].copy(),self.objectiveFunction,self.nodeCentrality,self.globalFeature)
            while not eval_step.last():
                eval_output, prob = agent.step(eval_step, is_evaluation=True)
                eval_step = env.step(eval_output)
            gamma = env.get_state.reward1
            N = env.get_state.Graph.vcount()
            AUC.append(area_under_curve(condMaxNum,N,x[i][:len(gamma)],gamma))
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

    def input_graph(graph_path):
        GRAPH = Graph.Read_Ncol(graph_path, directed=False)
        nodes = [v.index for v in GRAPH.vs]
        map = {n:int(i) for i, n in enumerate(nodes)}
        GRAPH = reset(GRAPH)  
        Graph.simplify(GRAPH)
        return GRAPH, map

    def train(self,agent=None):
        game = GraphGame
        env = Environment(game)
        # WandB – Initialize a new run
        wandb.init(mode =self.params.wandb_mode,entity="bhandk", project=self.params.wandb_project,name=self.params.wandb_name,config=self.params)
        wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release
        size_CV = self.params.validation_test_size
        evaluation, eval_x = get_Validation(size_CV,seed=0)#get_Validation(4,file_path)
        CV_AUC = []
        if agent == None:
            agent = DQN(
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
        wandb.watch(agent._q_network, log="all")
        graph_batch_size = self.params.graph_batch_size
        for ep in tqdm(range(int(self.params.num_train_episodes))):
            if (ep) % self.params.graph_suffle == 0:
                Batch_Graph = self.Generate_Batch_Graph(graph_batch_size,seed=ep)
            time_step = env.reset(Batch_Graph[int(ep%graph_batch_size)].copy(),self.objectiveFunction,self.nodeCentrality,self.globalFeature)
            while not time_step.last():
                action, prob = agent.step(time_step) 
                time_step = env.step(action)
            # Episode is over, step agent with final info state.
            agent.step(time_step)
            wandb.log({"loss": agent._last_loss_value,"cummulative_reward":env.get_state._returns,"epsilon":agent.epsilon})
            if (ep + 1) % self.params.eval_every == 0:
                meanAUC = self.validation(size_CV, env, agent, evaluation, eval_x)
                CV_AUC.append(meanAUC)
                wandb.log({"Evaluation_Dataset": meanAUC})          
            if (ep + 1) % self.params.save_every == 0:
                checkpoint = {'_q_network': agent._q_network.state_dict(),'target_q_network': agent._target_q_network.state_dict(),'_optimizer' :agent._optimizer.state_dict()}
                title = self.params.checkpoint_dir+"_"+str(ep+1)
                torch.save(checkpoint, title)
        self.bestModel(CV_AUC)
        wandb.config.update({'best_dir':self.best_dir})
        wandb.finish() 

    def evaluation(self, graphpath, bestModel):
        """
        Given a path to an edgelist of the Graph and a model, it evaluates the removal of node based on the objective function. 
        """
        if self.objectiveFunction.__name__ == "numberConnectedComponent":
            condMaxNum = True
        else:
            condMaxNum = False
        G , _ = input_graph(graphpath)
        model = torch.load(bestModel)
        N = G.vcount()
        game = GraphGame
        env = Environment(game)
        attacker = DQN(state_representation_size=self.params.centrality_features,
                            hidden_layers_sizes= self.params.hidden_layers,
                           global_feature_size = self.params.global_features)
        attacker._q_network.load_state_dict(model["_q_network"])
        attacker._optimizer.load_state_dict(model["_optimizer"])
        rewards, value, actions = EvaluateModel(env, self.objectiveFunction, self.nodeCentrality, self.globalFeature, attacker, G)
        x =  np.flip(np.arange(N)[N:0:-1]/N)
        auc = area_under_curve(condMaxNum,N,x[:len(value)],value)
        fraction = len(actions)/N
        return auc, fraction
    
    def evaluationOthers(self, graphpath, actionspath):
        """
        Given a path to an edgelist of a Graph, the list of action sequences, "actions" in order, and an objectiveFunction it evaluates the removal of node based on the objective function. 
        """
        import warnings
        if self.objectiveFunction.__name__ == "numberConnectedComponent":
            condMaxNum = True
        else:
            condMaxNum = False
            
        G , _ = self.input_graph(graphpath)
        N = G.vcount()
        fname = actionspath+graphpath.split("/")[-1]
        with warnings.catch_warnings():
            warnings.simplefilter("Ignore")
            actions = np.loadtxt(fname, dtype=int, unpack=True)
        rewards, value, actions = EvaluateACTION(actions,self.objectiveFunction,G)
        x =  np.flip(np.arange(N)[N:0:-1]/N)
        value = np.array(value)/value[0]
        auc = area_under_curve(condMaxNum,N,x[:len(value)],value)
        fraction = len(actions)/N
        return auc, fraction

    def compareBenchmarkSynthetic(self,evaluation,bestmodel):
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
                try:
                    graphpath = syntheticGraph[folder] + graphname
                    auc, fraction = evaluation(graphpath, bestmodel)
                    result[folder][name]["auc"].append(auc)
                    result[folder][name]["fraction"].append(fraction)
                except:
                    pass
        self.result_Syn = result
        
    def compareBenchmarkSyntheticMotif(self,evaluation,bestmodel):
        """
        Evaluate Graph on Synthetic Graph with Motifs attached. 
        """
        syntheticMotifGraph = {"BA":"./Dataset/Validation/Motifs_Attached/BA/",
                    "Tree":"./Dataset/Validation/Motifs_Attached/Tree/", 
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
                try:
                    graphpath = syntheticMotifGraph[folder] + graphname
                    auc, fraction = evaluation(graphpath,bestmodel)
                    result[folder][name]["auc"].append(auc)
                    result[folder][name]["fraction"].append(fraction)
                except:
                    pass
            '''for graph in result[folder]:
                for measure in result[folder][graph]:
                    plt.plot(result[folder][graph][measure])
                    plt.title(folder+" :"+measure)
                    plt.show()'''
        self.result_SynMotif = result


def main(argv):
    hyperparameter_path = ''
    opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <HyperParameter JSON file path>')
            print("Example:")
            print ('model.py -i ./utils/hyperparameters/BA/ba_params.json')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            hyperparameter_path = arg
        else:
            print("Error in the argument. Use -h for help")
    bm = BenchMark(hyperparameter_path)
    bm.train()
 
if __name__ == "__main__":
   main(sys.argv[1:])