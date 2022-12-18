import random 
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from igraph import Graph


       
def gen_graph(cur_n, g_type,seed=None):
    random.seed(seed)
    if g_type == 'erdos_renyi':
        g = Graph.Erdos_Renyi(n=cur_n, p=random.uniform(0.10,0.15), directed=False)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n=cur_n, m=random.randint(2,4), p=random.uniform(0.01,0.05),seed = seed)
        g = Graph.from_networkx(g)
    elif g_type == 'small-world':
        g = Graph.Watts_Strogatz(dim =1 ,size=cur_n, nei=random.randint(2,5), p=random.uniform(0.1,0.2))
    elif g_type == 'barabasi_albert':
        g = Graph.Barabasi(n=cur_n, m=random.randint(1,3),directed=False)
    elif g_type == 'geometric':
        g = Graph.GSG(n=cur_n, radius=random.uniform(0.1,0.4),directed=False)
    g.vs['name'] = range(cur_n)
    return g
def add_super_node(graph):
    x = len(graph)
    ebunch = [(i,x) for i in range(x)]
    graph.add_node(x)
    graph.add_edges_from(ebunch)
    return graph
    
def gen_new_graphs(graph_type,seed = None):
    random.seed(seed)
    np.random.seed(seed)
    a = np.random.choice(graph_type) if len(graph_type) !=1 else graph_type[0]
    number_nodes = random.randint(30,50)
    graph = gen_graph(number_nodes, a,seed)
    #graph =add_super_node(graph)
    active = 1
    graph.vs["active"] = active
    return graph    
  

def reset(graph):
    active = 1
    graph.vs["active"] = active
    return graph 

# Helper functions for game details.
'''def get_lcc(g):
    G = g.to_networkx()
    return len(max(nx.connected_components(G), key=len))'''
def get_lcc(G):
    found = set()

    comps = []
    for v in G.vs.indices:
        if v not in found:
            connected = G.bfs(v)[0]
            found.update(connected)
            comps.append(connected)

    #return len(max(comps, key=len))   
    return sum(map((lambda x: len(x)*(len(x)-1)/2), comps))
               
def molloy_reed(g):
  all_degree = np.array(g.degree())
  degs = all_degree
  #nonmax_lcc = list(set(g.vs.indices).difference(set(max(g.connected_components(mode='weak'), key=len))))
  #degs = np.delete(all_degree, np.array(nonmax_lcc, dtype=int))#for non max LCC
  #degs = np.delete(deg,-1)#for supernode
  k = degs.mean()
  k2 = np.mean(degs** 2)
  if k ==0:
    beta = 0
  else:
    beta = k2/k
  return beta



def reduceddegree(g): 
    x = torch.FloatTensor(g.degree()).reshape((-1, 1)) - 1
    return x

def network_dismantle(board, init_lcc):
    """Checks if a line exists, returns "x" or "o" if so, and None otherwise."""
    all_nodes = np.array(board.vs["active"])
    active_nodes = np.where(all_nodes == 1)[0]
    largest_cc = get_lcc(board)
    cond = True if len(active_nodes) <= 2 or board.ecount() <= 2  or (largest_cc/init_lcc) <= 0.01 else False
    #print(cond,len(active_nodes),board.ecount(),(largest_cc/init_lcc))
    return cond, largest_cc

def board_to_string(board):
    """Returns a string representation of the board."""
    value = np.array(list(board.nodes(data="active")))
    return " ".join(str(f) for e, f in value)

def from_igraph(features,graph):
    edges = [edge.tuple for edge in graph.es]
    edge_index = torch.tensor(edges, dtype=torch.int).t().contiguous()
    if edge_index.shape[0]!= 0:
        edge_index = to_undirected(edge_index)
    return Data(x=features.get_centrality_features(graph),edge_index=edge_index)