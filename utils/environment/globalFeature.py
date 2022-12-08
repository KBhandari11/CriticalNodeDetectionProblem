import random 
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from igraph import Graph



class Global_Feature():
    def __init__(self, feature = None):
        if feature == None:
            self.feature = ["heterogeneity","density","resilience","gini","entropy","transitivity"]
        elif feature == []:
            self.feature = ["none"]
        else:
            self.feature = feature
    def get_global_features(self, G):
        globalfeature = {}
        self.M = G.ecount()
        self.N = G.vcount()
        self.degs = np.array(G.degree())
        self.degs.sort()
        self.k1 = self.degs.mean()
        self.k2 = np.mean(self.degs** 2)
        self.div = self.k2 - self.k1**2
        for f in self.feature:
            globalfeature[f] = getattr(self, 'case_' + f)(G)
        if len(self.feature) == 1:
            x = np.hstack(list(globalfeature.values()))
        else:
            x = np.array(list(globalfeature.values()))
        x = np.nan_to_num(x,nan = 0)
        x = torch.from_numpy(x.astype(np.float32))
        return x
    def case_heterogeneity(self, g):
        if self.k1 != 0 : 
            return self.div/self.k1
        else: 
            return 0 
    def case_density(self, g):
        return (2*self.M)/(self.N*(self.N-1))
    def case_resilience(self,g):
        if self.k1 != 0 : 
            return self.k2/self.k1
        else:
            return 0 
    def case_gini(self,g):
        if self.k1 != 0 : 
            return np.sum(self.degs * (self.degs + 1))/(self.M*self.N) - (self.N+1)/self.N
        else: 
            return 0 
    def case_entropy(self,g):
        if self.k1 != 0 : 
            return entropy(self.degs/self.M)/self.N
        else: 
            return 0 
    def case_transitivity(self,g):
        return g.transitivity_undirected()
    def case_none(self, g):
        return np.array([])