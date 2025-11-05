""" A Python environment to simulate node removal on graph for the Critical Node Detection Problem."""
import numpy as np
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
  def __init__(self, Graph, objectiveFunction, centralityFeature, globalFeature):
      # immutable graph reference
      self.Graph              = Graph
      self.obj_fun            = objectiveFunction
      self.centralityFeature  = centralityFeature
      self.globalFeature      = globalFeature
      self.num_nodes = self.Graph.vcount()
      
      # --------  fast incremental book-keeping  -------- #
      self.active_v  = np.ones(Graph.vcount(), dtype=bool)        # alive nodes mask
      self.deg       = np.array(Graph.degree())                   # static original degrees
      self.sum_deg   = self.deg.sum().astype(float)
      self.sum_deg2  = (self.deg**2).sum().astype(float)

      # cache incidence lists once (list of np arrays)
      self.inc_edges = [np.fromiter(Graph.incident(v), dtype=int)
                        for v in range(Graph.vcount())]

      # --------  expensive statistics (computed in full)  -------- #
      self._refresh_features()          # sets self.info_state, self.global_feature
      self._rewards = 0
      self._returns = 0
      self.reward1   = [self.obj_fun(Graph)]    # γ_0  (objective)
      self.reward2   = [self._molloy_from_sums()]  # β_0  (Molloy–Reed)
      self._returns  = 0.0
      self.r = []
      self._is_term  = False
      self.alpha = 0.75#(1-nx.density(self.Graph))
      print("NEWWWWW")
      print(self.Graph)
      print("-----")
    

  def _molloy_from_sums(self):
        """β = Σk_i² / Σk_i   (uses current degree sums)"""
        return self.sum_deg2 / self.sum_deg if self.sum_deg else 0.
  
  def _refresh_features(self):
    """Re-compute info_state & global_feature on the *current* live subgraph."""
    live_idx = np.nonzero(self.active_v)[0]
    subg = self.Graph.subgraph(live_idx)
    self.info_state     = from_igraph(self.centralityFeature, subg)
    self.global_feature = self.globalFeature.get_global_features(subg)

  def _legal_actions(self):
        """IDs of still-active nodes (NumPy gives ~2× speed-up vs. np.where+==)."""
        return np.flatnonzero(self.active_v)

  def apply_actions(self, v):
    """Remove node *v* and update all statistics incrementally."""
    if not self.active_v[v]:
        raise ValueError(f"Node {v} already removed.")

    # 1. mask node & incident edges
    self.active_v[v] = False
    dv = self.deg[v]

    # 2. update Molloy–Reed numerator / denominator in O(1)
    self.sum_deg  -= 2 * dv
    self.sum_deg2 -= dv * dv
    beta = self._molloy_from_sums() or self.reward2[-1]

    # 3. objective / stopping condition  (FIXED order)
    cond, gamma = network_dismantle(
        self.Graph,
        self.obj_fun,
        self.reward1[0],
        self.active_v
    )

    # 4. step reward
    norm_gamma = abs(self.reward1[-1] - gamma) / self.reward1[-1]
    norm_beta  = abs(self.reward2[-1] - beta)  / self.reward2[-1]
    step_reward = ((self.num_nodes - len(self.reward1)) / self.num_nodes) * (
                    self.alpha * norm_gamma + (1 - self.alpha) * norm_beta)

    # 5. book-keeping
    self._rewards = step_reward
    self._returns += step_reward
    self.reward1.append(gamma)
    self.reward2.append(beta)
    self._is_term = cond
    self.r.append(step_reward)

    # 6. recompute full features (your requirement)
    self._refresh_features()
    print("action:",v)
    print(self.Graph)
    print("+++"*100)

  
    
  def is_terminal(self):          return self._is_term
  def returns(self):              return self._returns
  def rewards(self):              return self._rewards  # or however you log them
  def _action_to_string(self, _, a): return f"0({a})"

  def new_initial_state(self,Graph,objectiveFunction,centralityFeature, globalFeature):
      self.Graph = Graph
      self.objectiveFunction = objectiveFunction
      self.info_state =  from_igraph(centralityFeature,self.Graph)
      self.global_feature = globalFeature.get_global_features(self.Graph)
      self.reward1 = [objectiveFunction(self.Graph)]
      self.reward2 = [molloy_reed(self.Graph)]
      self.r = []
      self.alpha = 0.75#(1-nx.density(self.Graph))
      
