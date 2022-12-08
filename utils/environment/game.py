# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
""" Graph Attack and Defense implemented in Python.
This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.
"""



import numpy as np
import networkx as nx
from torch_geometric import utils
from  utils.environment.envhelper import*
#import pyspiel





class GraphGame():
  """A Python version of the Graph game."""

  def __init__(self,Graph, centralityFeature, globalFeature):
    self.Graph = Graph
    self.centralityFeature =  centralityFeature
    self.globalFeature = globalFeature  

  def new_initial_state(Graph,centralityFeature, globalFeature):
    """Returns a state corresponding to the start of a game."""
    return GraphState(Graph,centralityFeature, globalFeature)



class GraphState():
  """A python version of the Tic-Tac-Toe state."""
  def __init__(self, Graph,centralityFeature, globalFeature):
    self._is_terminal = False
    self.Graph = Graph
    self.centralityFeature = centralityFeature
    self.globalFeature = globalFeature
    self.num_nodes = self.Graph.vcount()
    self.info_state =  from_igraph(self.centralityFeature,self.Graph)
    self.global_feature = self.globalFeature.get_global_features(self.Graph)
    self._reward = 0
    self._returns = 0
    self.lcc = [get_lcc(self.Graph)]
    self.r = []
    self.alpha = 0.75#(1-nx.density(self.Graph))
    self.beta = [molloy_reed(self.Graph)]

  def _legal_actions(self):
    """Returns a list of legal actions, sorted in ascending order."""
    all_nodes = np.array(self.Graph.vs["active"])
    active_nodes = np.where(all_nodes == 1)[0]
    #print("allNodes",all_nodes)
    action_sequence = active_nodes 
    #print("actionSequences",action_sequence)
    #return np.delete(action_sequence,-1) #for supernode
    return action_sequence

  def apply_actions(self, attack_node):
    """Applies the specified action to the state."""
    self.Graph.vs[attack_node]["active"] = 0
    ebunch = self.Graph.incident(attack_node)
    self.Graph.delete_edges(ebunch)
    cond, l = network_dismantle(self.Graph, self.lcc[0])
    self.info_state =  from_igraph(self.centralityFeature,self.Graph)
    self.global_feature = self.globalFeature.get_global_features(self.Graph)
    beta = molloy_reed(self.Graph)
    if beta == 0:
      beta = self.beta[-1]
      cond = True
    reward_1 = (self.lcc[-1] - l)/self.lcc[-1]
    reward_2 = (self.beta[-1] - beta)/self.beta[-1]
    self._rewards = ((self.num_nodes-len(self.lcc))/self.num_nodes)* (self.alpha * reward_1 +(1-self.alpha)*reward_2)
    self._returns += self._rewards
    self.beta.append(beta)  
    self.lcc.append(l)
    self.r.append(self._rewards)
    self._is_terminal = cond
    #print(cond,self.info_state.edge_index.size())
    
  def _action_to_string(self, player, action):
    """Action -> string."""
    return "{}({})".format(0 if player == 0 else 1, action)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns
  def rewards(self):
    """Total reward for each player over the course of the game so far."""
    return self._rewards

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return board_to_string(self.Graph)

  def new_initial_state(self,Graph,centralityFeature, globalFeature):
      self.Graph = Graph
      self.info_state =  from_igraph(centralityFeature,self.Graph)
      self.global_feature = globalFeature.get_global_features(self.Graph)
      self.lcc = [get_lcc(self.Graph)]
      self.r = []
      self.alpha = (1-nx.density(self.Graph))
      self.beta = [molloy_reed(self.Graph)]