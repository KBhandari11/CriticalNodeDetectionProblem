#!/usr/bin/env python3
"""DQN agents trained on Breakthrough by independent Q-learning."""
from utils.environment.game import GraphGame
from utils.environment.nodeCentrality import Node_Centrality
from utils.environment.globalFeature import Global_Feature
from utils.reinforcement_learning.rl_environment import Environment
from utils.validation import get_Validation,area_under_curve
from utils.environment.envhelper import gen_new_graphs
from utils.reinforcement_learning.dqn_TS import DQN
#from utils.reinforcement_learning.dqn import DQN
from utils.hyperparameters.params import Params

from tqdm import tqdm
import torch
import copy
import numpy as np
from datetime import datetime

#WandB Alert
from datetime import timedelta
from wandb import AlertLevel

import wandb
import os
os.environ['WANDB_NOTEBOOK_NAME'] = './Training_DiffSize_Different_Graph'
wandb.login()
# DQN model hyper-parameters
params = Params("./utils/hyperparameters/Mix/mixed_params_CN.json")

#epsilon_decay_duration= int(2e6) if num_train_episodes == int(1e5) else int(2e7)
# WandB – Initialize a new run
now = datetime.now()
#wandb.init(entity="bhandk", project="Attack_and_Defense_Different_Graph_Type",name=now.strftime("%d/%m/%Y %H:%M:%S"),config=params)
wandb.init(mode="online",entity="bhandk", project="Attack_and_Defense_Different_Graph_Type",name=params.wandb_name,config=params)
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

nodeCentrality = Node_Centrality(params.centrality_feature_name)
globalFeature = Global_Feature(params.global_feature_name)

def Generate_Batch_Graph(size,seed = None):
    graph_type = ['erdos_renyi', 'powerlaw','small-world', 'barabasi_albert']
    Batch_Graph = [gen_new_graphs(graph_type,seed=seed+i) for i in range(size)]
    return np.array(Batch_Graph,dtype=object)
evaluation, eval_x = get_Validation(params.validation_test_size,seed=0)#get_Validation(4,file_path)

def main(agent=None):
    game = GraphGame
    env = Environment(game)
    size_CV = params.validation_test_size
    CV_AUC = []
    if agent == None:
        agent = DQN(
                state_representation_size=params.centrality_features,
                global_feature_size = params.global_features,
                hidden_layers_sizes=params.hidden_layers,
                replay_buffer_capacity=int(params.replay_buffer_capacity),
                learning_rate=params.learning_rate,
                update_target_network_every=  params.update_target_network_every,
                learn_every=params.learn_every,
                discount_factor=params.discount_factor,
                min_buffer_size_to_learn=params.min_buffer_size_to_learn,
                power = params.epsilon_power,
                nsteps=params.nstep,
                epsilon_start=params.epsilon_start,
                epsilon_end=params.epsilon_end,
                epsilon_decay_duration=params.epsilon_decay_duration,
                batch_size=params.batch_size)
    #agents.append(random_agent.RandomA1gent(player_id=1, num_actions=num_actions))
    wandb.watch(agent._q_network, log="all")
    graph_batch_size = params.graph_batch_size
    for ep in tqdm(range(int(params.num_train_episodes))):
        if (ep) % params.graph_suffle == 0:
            Batch_Graph = Generate_Batch_Graph(params.graph_batch_size,seed=ep)
        time_step = env.reset(Batch_Graph[int(ep%graph_batch_size)].copy(),nodeCentrality,globalFeature)
        #time_step = env.reset(gen_new_graphs())
        while not time_step.last():
            action, prob = agent.step(time_step)
            time_step = env.step(action)
        # Episode is over, step all agents with final info state.
        agent.step(time_step)
        #if agents[0]._last_loss_value != None:
        wandb.log({"loss": agent._last_loss_value,"cummulative_reward":env.get_state._returns,"epsilon":agent.epsilon})
        if (ep + 1) % params.eval_every == 0:
            AUC = []
            for i in range(size_CV):
                eval_step = env.reset(evaluation[i].copy(),nodeCentrality,globalFeature)
                while not eval_step.last():
                    eval_output, prob = agent.step(eval_step, is_evaluation=True)
                    eval_step = env.step(eval_output)
                lcc = env.get_state.lcc
                AUC.append(area_under_curve(eval_x[i][:len(lcc)],lcc))
            #wandb.log({"Eval_Homogeneous": np.mean(AUC[0:int(size_CV/2)]),"Eval_Heterogeneous":np.mean(AUC[int(size_CV/2):int(size_CV)])  })
            meanAUC = np.mean(AUC)
            CV_AUC.append(meanAUC)
            wandb.log({"Evaluation_Dataset": meanAUC})          
        if (ep + 1) % params.save_every == 0:
            checkpoint = {'_q_network': agent._q_network.state_dict(),'target_q_network': agent._target_q_network.state_dict(),'_optimizer' :agent._optimizer.state_dict()}
            title = params.checkpoint_dir+"_"+str(ep+1)
            torch.save(checkpoint, title)
    wandb.finish()
main()