""" A Python environment to simulate node removal on graph for the Critical Node Detection Problem."""
import numpy as np
import networkx as nx
from torch_geometric import utils
from  utils.environment.envhelper import*


class GraphGame():
  """ The Graph game for node removal"""

  def __init__(self,Graph, objectiveFunction, centralityFeature, globalFeature):
    self.Graph = Graph
    self.objectiveFunction = objectiveFunction
    self.centralityFeature =  centralityFeature
    self.globalFeature = globalFeature  

  def new_initial_state(Graph, objectiveFunction, centralityFeature, globalFeature):
    """Returns a state corresponding to the start of a game."""
    return GraphState(Graph,objectiveFunction, centralityFeature, globalFeature)



class GraphState():
  """Graph State"""
  def __init__(self, Graph,objectiveFunction ,centralityFeature, globalFeature):
    self._is_terminal = False
    self.Graph = Graph
    self.objectiveFunction = objectiveFunction
    self.centralityFeature = centralityFeature
    self.globalFeature = globalFeature
    self.num_nodes = self.Graph.vcount()
    self.info_state =  from_igraph(self.centralityFeature,self.Graph)
    self.global_feature = self.globalFeature.get_global_features(self.Graph)
    self._reward = 0
    self._returns = 0
    self.reward1 = [self.objectiveFunction(self.Graph)]
    self.reward2 = [molloy_reed(self.Graph)]
    self.r = []
    self.alpha = 0.75#(1-nx.density(self.Graph))
    

  def _legal_actions(self):
    """Returns a list of legal actions, sorted according to Graph node index."""
    """Need to be careful with igraph Graph index and Graph name """
    all_nodes = np.array(self.Graph.vs["active"])
    active_nodes = np.where(all_nodes == 1)[0]
    action_sequence = active_nodes 
    return action_sequence

  def apply_actions(self, attack_node):
    """Applies the specified action to the state and updates the Centrality Feature and Global Feature respectively."""
    self.Graph.vs[attack_node]["active"] = 0
    ebunch = self.Graph.incident(attack_node)
    self.Graph.delete_edges(ebunch)
    cond, gamma = network_dismantle(self.Graph,self.objectiveFunction, self.reward1[0])
    self.info_state =  from_igraph(self.centralityFeature,self.Graph)
    self.global_feature = self.globalFeature.get_global_features(self.Graph)
    beta = molloy_reed(self.Graph)
    if beta == 0:
      beta = self.reward2[-1]
      cond = True
    norm_gamma = abs(self.reward1[-1] - gamma)/self.reward1[-1]
    norm_beta = abs(self.reward2[-1] - beta)/self.reward2[-1]
    self._rewards = ((self.num_nodes-len(self.reward1))/self.num_nodes)* (self.alpha * norm_gamma +(1-self.alpha)*norm_beta)
    self._returns += self._rewards
    self.reward2.append(beta)  
    self.reward1.append(gamma)
    self.r.append(self._rewards)
    self._is_terminal = cond
    #print(cond,self.info_state.edge_index.size())
    
  def _action_to_string(self, player, action):
    """Each action -> string."""
    return "{}({})".format(0 if player == 0 else 1, action)

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward over the course of the simulation so far."""
    return self._returns
  def rewards(self):
    """Total reward over the course of thesimulation so far."""
    return self._rewards

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return board_to_string(self.Graph)

  def new_initial_state(self,Graph,objectiveFunction,centralityFeature, globalFeature):
      self.Graph = Graph
      self.objectiveFunction = objectiveFunction
      self.info_state =  from_igraph(centralityFeature,self.Graph)
      self.global_feature = globalFeature.get_global_features(self.Graph)
      self.reward1 = [objectiveFunction(self.Graph)]
      self.reward2 = [molloy_reed(self.Graph)]
      self.r = []
      self.alpha = 0.75#(1-nx.density(self.Graph))
      