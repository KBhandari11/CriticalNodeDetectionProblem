import random 
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from igraph import Graph
import multiprocessing
from multiprocessing import Pool


class Node_Centrality():
    def __init__(self, feature = None):
        if feature == None:
            self.feature = ["degree","eigen","pagerank","ci"]
        else:
            self.feature = feature

    def get_centrality_features(self, G):
        centrality = {}    
        for f in self.feature:
            centrality[f] = getattr(self, 'case_' + f)(G)
        x = np.column_stack(list(centrality.values()))
        x = np.nan_to_num(x,nan = 0)
        x = torch.from_numpy(x.astype(np.float32))
        return x
        '''centrality = {}    
        centralityFunction = lambda x: getattr(self, 'case_' + x)(G)
        num_processes = multiprocessing.cpu_count()
        results = []
        with Pool(num_processes) as p:
            results.append(p.map(centralityFunction, self.feature))
        for i, f in enumerate(self.feature):
            centrality[f] = results[i][0]
        x = np.column_stack(list(centrality.values()))
        x = np.nan_to_num(x,nan = 0)
        x = torch.from_numpy(x.astype(np.float32))
        return x'''

    def case_degree(self, g):
        """Get degree value of each node for the Graph"""
        return np.array(g.degree()) / (g.vcount() - 1)

    def case_eigen(self, g):
        """Get eigencentrality value of each node for the Graph"""
        try:
            eigen_centrality = np.array(g.eigenvector_centrality())
        except:
            #ARPACKOptions.tol =  int(10e-2)
            #value = Graph.arpack_defaults.tol = int(10e-2)
            eigen_centrality = np.array(g.eigenvector_centrality())
        return eigen_centrality

    def case_pagerank(self, g):
        """Get personalized pagerank value of each node for the Graph"""
        return np.array(g.personalized_pagerank())

    def case_ci(self, g):
        """Get collective influence value of each node for the Graph"""
        return self.get_ci(g, 2)

    def case_core(self, g):
        """Get k core value of each node for the Graph"""
        G = g.to_networkx()
        return np.array(list(nx.core_number(G).values()))

    def case_active(self, g):
        """Get active value of each node for the Graph"""
        return np.array(g.vs()["active"])#np.array(g.nodes.data("active"))[:,1]
    
    '''def get_Ball(self,g,v,l,n):
        """Get Ball value of a node'v', within length 'l' with neighbour of 'n'"""
        if l == 1:
            return [v]
        else:
            for i in g.neighbors(v):
                if not(i in n):
                    a = self.get_Ball(g,i,l-1,n)
                    if a == None:
                        n = list(set().union([i],n))
                    else:
                        n = list(set().union(a,[i],n))
                #print('n',n)
            if v in n:
                return list(set(n)-set([v]))
            else:
                return n'''
    def get_Ball(self,g,v,n):
        for i in g.neighbors(v):
            if not(i in n):
                n.append(i)
            for j in g.neighbors(i):
                if not(j in n):
                    n.append(j)
        return n
            
            
    def get_ci(self,g, l):
        """Compute the Collective Influence of the Graph."""
        ci = []
        degs = np.array(g.degree())
        #G_nx = g.to_networkx()
        results = []
        for i in g.vs.indices:            
            #n = self.get_Ball(g,i,l,[i])
            #result = pool.apply_async(self.get_Ball, args= ((g,i,[i]),))
            result = self.get_Ball(g,i,[i])
            results.append((i,result))
            #np.array([path[-1] for path in g.get_shortest_paths(i) if path and len(path) <= l])
            #np.array([path[-1] for path in g.get_shortest_paths(i) if path and len(path) <= l])
            #print("path",n)
            #print("ball",get_Ball(g,i,l,[i]))
            #print("networkx",list(nx.single_source_shortest_path(G_nx,i,l))[0:])
        #pool.join()
        #pool.close()
        for result in results:
            i, n = result#result.get()
            j = np.sum(degs[n] - 1)
            ci.append((g.degree(i) - 1) * j)
        ci = np.array(ci)
        if np.std(ci) != 0:
            ci = (ci - np.mean(ci)) / np.std(ci)
        else:
            ci = (ci - np.mean(ci))
        return ci

    '''
    def get_ci(g, l):
        ci = []
        degs = np.array(g.degree())
        for i in g.vs.indices:
            n = [path[-1] for path in g.get_all_shortest_paths(i) if path and len(path) <= l]
            print(n)
            if i in n:
                n = n.remove(i)
            print(n)
            n = np.array(n)
            j = np.sum(degs[n] - 1)
            ci.append((g.degree(i) - 1) * j)
        return ci
    '''