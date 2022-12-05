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

    return len(max(comps, key=len))   

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

def global_feature(g): 
    M = g.ecount()
    N = g.vcount()
    degs = np.array(g.degree())
    k1 = degs.mean()
    k2 = np.mean(degs** 2)
    div = k2 - k1**2
    if k1 != 0:
        heterogeneity = div/k1
        #density = (2*M)/(N*(N-1))
        #resilience = k2/k1
        #degs.sort()
        #gini = np.sum(degs * (degs + 1))/(M*N) - (N+1)/N
        #entrop = entropy(degs/M)/N
        #transitivity = g.transitivity_undirected()
    else:
        heterogeneity = 0
        #density = (2*M)/(N*(N-1))
        #resilience = 0
        #gini = 0
        #entrop = 0
        #transitivity = g.transitivity_undirected()
    global_properties = [heterogeneity]#np.hstack((density,resilience,heterogeneity,gini,entrop,transitivity))
    global_properties = np.nan_to_num(global_properties,nan = 0)
    #global_properties = np.hstack((density,resilience,heterogeneity))
    global_properties = torch.from_numpy(global_properties.astype(np.float32))#.to(device)'''
    #global_properties = torch.empty((6,0), dtype=torch.float)
    return global_properties

def get_Ball(g,v,l,n):
    if l == 1:
        return [v]
    else:
        for i in g.neighbors(v):
            if not(i in n):
                a = get_Ball(g,i,l-1,n)
                if a == None:
                    n = list(set().union([i],n))
                else:
                    n = list(set().union(a,[i],n))
            #print('n',n)
        if v in n:
            return list(set(n)-set([v]))
        else:
            return n
         

def get_ci(g, l):
    ci = []
    degs = np.array(g.degree())
    #G_nx = g.to_networkx()
    for i in g.vs.indices:
        n = get_Ball(g,i,l,[i]) #np.array([path[-1] for path in g.get_shortest_paths(i) if path and len(path) <= l])
        #print("path",n)
        #print("ball",get_Ball(g,i,l,[i]))
        #print("networkx",list(nx.single_source_shortest_path(G_nx,i,l))[0:])
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
def get_centrality_features(g):
    degree_centrality = np.array(g.degree()) / (g.vcount() - 1)
    #precolation_centrality = list(nx.percolation_centrality(g,attribute='active').values())
    #closeness_centrality = list(nx.closeness_centrality(g).values())
    try:
        eigen_centrality = np.array(g.eigenvector_centrality())
    except:
        #ARPACKOptions.tol =  int(10e-2)
        #value = Graph.arpack_defaults.tol = int(10e-2)
        eigen_centrality = np.array(g.eigenvector_centrality())
    #clustering_coeff = np.array(g.transitivity_local_undirected())
    #core_num = np.nan_to_num(np.array(g.coreness('all')),nan = 0)
    #G = g.to_networkx()
    #core_num = np.array(list(nx.core_number(G).values()))
    pagerank = np.array(g.personalized_pagerank())
    ci = get_ci(g, 2)
    #active = np.array(g.nodes.data("active"))[:,1]
    #x = np.column_stack((degree_centrality,clustering_coeff,pagerank, core_num ))
    #x = np.column_stack((degree_centrality,eigen_centrality,pagerank,clustering_coeff, core_num, ci ))
    x = np.column_stack((degree_centrality,eigen_centrality,pagerank,ci))
    #x = degree_centrality.reshape(-1,1)
    x = np.nan_to_num(x,nan = 0)
    return x

def features(g): 
    #actualGraph = g.g(np.arange(len(g)-1)) #for actual graph
    #x = get_centrality_features(actualGraph) #with supernode
    x = get_centrality_features(g)
    #x[:-1,:] =x_actual
    #x_normed = (x - np.mean(x)) / np.std(x) #Standardize features
    #active_nodes =  np.where(np.array(list(g.nodes(data="active")))[:,1] == 0)[0]
    #x_normed[active_nodes,:]=np.zeros(np.shape(x_normed)[1])
    x = torch.from_numpy(x.astype(np.float32))#.to(device)
    return x

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

def from_igraph(graph):
    edges = [edge.tuple for edge in graph.es]
    edge_index = torch.tensor(edges, dtype=torch.int).t().contiguous()
    if edge_index.shape[0]!= 0:
        edge_index = to_undirected(edge_index)
    return Data(x=features(graph),edge_index=edge_index)