''' All helper function needed for the Game Environment '''


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
    """
    Generate Random Synthetic Graph given a nodesize and graph type
    """
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
    """
    Adding a super node to the graph
    """
    x = len(graph)
    ebunch = [(i,x) for i in range(x)]
    graph.add_node(x)
    graph.add_edges_from(ebunch)
    return graph
    

def gen_new_graphs(graph_type,seed = None):
    """
    Using the gen_graph to create a new graph of a new type 
    Create a new node attribute "active", to describe the individual node state; 
        - 0 to describe the node is attacked
        - 1 to describe the node is still alive
    """ 
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
    """
    Reset the "active" attribute to 1. 
    """
    active = 1
    graph.vs["active"] = active
    return graph 

def molloy_reed(g):
    """
    Molloy Reed criteron to describe the network robustness.
    The Molloy–Reed criterion is derived from the basic principle that in order for a giant component to exist, on average each node in the network must have at least two links. 
    This is analogous to each person holding two others' hands in order to form a chain. Using this criterion and an involved mathematical proof, 
    one can derive a critical threshold for the fraction of nodes needed to be removed for the breakdown of the giant component of a complex network.
        - Used to describe the second reward function. 
    """
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
    """
    Compute Reduced degree: Degree of the node - 1. 
    """
    x = torch.FloatTensor(g.degree()).reshape((-1, 1)) - 1
    return x

def network_dismantle(board,objectiveFunction, init_gamma):
    """Check if the current state of the graph meets the criteria:
            - Number of active node <= 2
            - Number of edges  <= 2
            - Fraction of the objective function is 1%.   
    """
    all_nodes = np.array(board.vs["active"])
    active_nodes = np.where(all_nodes == 1)[0]
    gamma = objectiveFunction(board)
    if objectiveFunction.__name__ == "numberConnectedComponent":
        init_gamma = board.vcount()
        cond = True if len(active_nodes) <= 2 or board.ecount() <= 2  or ((init_gamma-gamma+1)/init_gamma) <= 0.01 else False
    else:
        cond = True if len(active_nodes) <= 2 or board.ecount() <= 2  or (gamma/init_gamma) <= 0.01 else False

    return cond, gamma

def board_to_string(board):
    """Returns a string representation of the Graph with respect to the "active" attribute."""
    value = np.array(list(board.nodes(data="active")))
    return " ".join(str(f) for e, f in value)

def from_igraph(features,graph):
    """
    Convert the graph feature to PyTorch Geometric Data for GNN module.
    """
    edges = [edge.tuple for edge in graph.es]
    edge_index = torch.tensor(edges, dtype=torch.int).t().contiguous()
    if edge_index.shape[0]!= 0:
        edge_index = to_undirected(edge_index)
    return Data(x=features.get_centrality_features(graph),edge_index=edge_index)