from time import time
from scipy.sparse.linalg import eigsh
import networkx as nx
import numpy as np
import multiprocessing as mlp

class GMeasures(object):
    def __init__(self,G):
        '''
        G: networkx undirected graph
        '''
        self.G = G
        self.nodes = G.nodes()
        self.N = G.number_of_nodes()
        self.M = G.number_of_edges()

        self.__get_degrees()
        self.__get_degree_moments()
        self.__get_laplacian_spectrum()

    def __get_degrees(self):
        degs = self.G.degree()
        self.degs = np.array([degs[u] for u in range(len(degs))])

    def __get_degree_moments(self):
        self.k1 = self.degs.mean()
        self.k2 = np.mean(self.degs ** 2)

    def get_average_degree(self):
        return self.k1

    def get_density(self):
        return self.k1

    def get_resilience(self):
        beta = self.k2/self.k1

    def get_symmetry(self):
        return 1.0

    def get_heterogeneity(self):
        div = self.k2 - self.k1**2
        return div/self.k1

    def get_degree_entropy(self):
        from scipy.stats import entropy
        ps = self.degs/self.M
        return entropy(ps)/self.N

    def get_wedge_count(self):
        prod = self.degs * (self.degs - 1)/2
        return sum(prod)

    def get_power_law_exponent(self):
        mink = self.degs.min()
        alpha = 1 + self.N / sum(np.log(self.degs/mink))
        return alpha

    def get_gini_coefficient(self):
        sorted_degs = sorted(self.degs)
        gini = sum([(i+1) * sorted_degs[i] for i in range(self.N)])/(self.M*self.N) - (self.N+1)/self.N
        return gini

    def get_average_node_connectivity(self):
        return nx.average_node_connectivity(self.G)

    def get_average_edge_connectivity(self):
        return nx.algorithms.connectivity.connectivity.edge_connectivity(self.G)

    def get_average_shortest_path_length(self):
        return nx.average_shortest_path_length(self.G)

    def get_average_closeness_centrality(self):
        cs = nx.algorithms.closeness.closeness_centrality(self.G,wf_improved=False).values()
        return np.mean(list(cs))

    def get_average_closeness_centrality_wf(self):
        cs = nx.algorithms.closeness.closeness_centrality(self.G,wf_improved=True).values()
        return np.mean(list(cs))

    def get_average_eccentricity(self):
        varepsilons = nx.algorithms.distance_measures.eccentricity(self.G).values()
        return np.mean(list(varepsilons))

    def get_diameter(self):
        return nx.algorithms.distance_measures.diameter(self.G)

    def get_radius(self):
        return nx.algorithms.distance_measures.radius(self.G)

    def get_persistence(self):
        pass

    def get_average_edge_betweenness_centrality(self):
        bcs = nx.algorithms.centrality.edge_betweenness_centrality(self.G).values()
        return np.mean(list(bcs))

    def get_average_node_betweenness_centrality(self):
        bcs = nx.algorithms.centrality.betweenness_centrality(self.G).values()
        return np.mean(list(bcs))

    def get_central_point_of_dominance(self):
        bcs = np.array(list(nx.algorithms.centrality.betweenness_centrality(self.G).values()))
        cpd = np.sum(bcs.max() - bcs)/(self.N-1)
        return cpd

    def get_degree_assortativity_coefficient(self):
        return nx.algorithms.assortativity.degree_assortativity_coefficient(self.G)

    def get_average_core_number(self):
        core_nums = list(nx.core_number(self.G).values())
        return np.mean(core_nums)

    def get_average_clustering_coefficient(self):
        return nx.average_clustering(self.G)

    def get_modularity_score(self):
        import community
        parts = community.best_partition(self.G)
        clusters = {}
        for i, cidx in enumerate(parts.values()):
            if cidx in clusters:
                clusters[cidx].append(i)
            else:
                clusters[cidx] = [i]

        community = list(clusters.values())
        score = nx.community.quality.modularity(self.G,community)
        return score

    def __get_laplacian_spectrum(self):
        self.laplacian_eigenvalues = nx.linalg.spectrum.laplacian_spectrum(self.G)

    def get_laplacian_min_spectrum(self):
        return self.laplacian_eigenvalues.min()

    def get_laplacian_max_spectrum(self):
        return self.laplacian_eigenvalues.max()

    def get_transitivity(self):
        return nx.algorithms.cluster.transitivity(self.G)

    def get_local_efficiency(self):
        #return nx.algorithms.efficiency_measures.local_efficiency(self.G)
        efficiency_list = (self.__global_efficiency(self.G.subgraph(self.G[v])) for v in self.G)
        return sum(efficiency_list) / self.N

    def get_global_efficiency(self):
        #return nx.algorithms.efficiency_measures.global_efficiency(self.G)
        return self.__global_efficiency(self.G)

    def __global_efficiency(self,G):
        n = len(G)
        denom = n * (n - 1)
        if denom != 0:
            lengths = nx.all_pairs_shortest_path_length(G)
            g_eff = 0
            for source, targets in lengths:
                for target, distance in targets.items():
                    if distance > 0:
                        g_eff += 1 / distance
            g_eff /= denom
        else:
            g_eff = 0
        return g_eff

    def get_feature_values(self,keys=None):
        features = {'density':self.get_density,'resilience':self.get_resilience,
                    'heterogeneity':self.get_heterogeneity,'entropy':self.get_degree_entropy,
                    'wedge':self.get_wedge_count,'powerlaw_exponent':self.get_power_law_exponent,
                    'gini':self.get_gini_coefficient,'node_connectivity':self.get_average_node_connectivity,
                    'edge_connectivity':self.get_average_edge_connectivity,
                    'shortest_path_length':self.get_average_shortest_path_length,
                    'closeness_centrality':self.get_average_closeness_centrality,
                    'closeness_centrality_wf':self.get_average_closeness_centrality_wf,
                    'eccentricity':self.get_average_eccentricity,'diameter':self.get_diameter,
                    'radius':self.get_radius,'degree_centrality':self.get_average_degree_centrality,
                    'edge_betweenness_centrality':self.get_average_edge_betweenness_centrality,
                     'node_betweenness_centrality':self.get_average_node_betweenness_centrality,
                     'central_point_of_dominance':self.get_central_point_of_dominance,
                     'degree_assortativity_coefficient':self.get_degree_assortativity_coefficient,
                     'core_number':self.get_average_core_number,'clustering_coefficient':self.get_average_clustering_coefficient,
                     'modularity':self.get_modularity_score,'laplacian_min_spectrum':self.get_laplacian_min_spectrum,
                     'laplacian_max_spectrum':self.get_laplacian_max_spectrum,
                     'transitivity':self.get_transitivity,
                     'local_efficiency':self.get_local_efficiency,'global_efficiency':self.get_global_efficiency}

        if keys is None:
            values = [func() for func in features.values()]
        else:
            values = [features[key]() for key in keys if key in features]
        return values

class WGMeasures(object):
    def __init__(self,G):
        '''
        G: networkx undirected, weighted graph
        '''
        self.G = G
        self.nodes = G.nodes()
        self.N = G.number_of_nodes()
        self.M = G.number_of_edges()

        self.__get_degrees()
        self.__get_degree_moments()
        self.cpd, self.nbc = np.inf, np.inf
        self.A, self.L = None, None
        self.adj_lams, self.laplacian_lams = None, None
        self.dist_mat = self.__get_shortest_distance()

    def __get_degrees(self):
        degs = self.G.degree(weight='weight')
        self.degs = np.abs([degs[u] for u in range(len(degs))])

    def __get_degree_moments(self):
        self.k1 = self.degs.mean()
        self.k2 = np.mean(self.degs ** 2)

    def __get_shortest_distance(self):
        from scipy.sparse.csgraph import shortest_path
        A = nx.to_scipy_sparse_matrix(self.G, weight='cost',dtype=np.float,format='csr')
        return shortest_path(A, method='J', directed=False)

    def get_average_degree(self):
        return self.k1

    def get_density(self):
        return self.k1

    def get_resilience(self):
        beta = self.k2/self.k1
        return beta

    def get_heterogeneity(self):
        div = self.k2 - self.k1**2
        return div/self.k1

    def get_degree_entropy(self):
        from scipy.stats import entropy
        ps = 2*self.degs/self.degs.sum()
        return entropy(ps)/self.N

    def get_wedge_count(self):
        prod = self.degs * (self.degs - 1)/2
        return sum(prod)

    def get_power_law_exponent(self):
        mink = self.degs.min()
        alpha = 1 + self.N / sum(np.log(self.degs/mink))
        return alpha

    def get_gini_coefficient(self):
        sorted_degs = sorted(self.degs)
        sum_degs = sum(sorted_degs)
        gini = sum([(i+1) * sorted_degs[i] for i in range(self.N)])/(sum_degs*self.N) - (self.N+1)/self.N
        return gini

    def get_average_node_connectivity(self):
        return nx.average_node_connectivity(self.G)

    def get_average_edge_connectivity(self):
        return nx.algorithms.connectivity.connectivity.edge_connectivity(self.G)

    def get_average_shortest_path_length(self):
        return self.dist_mat.sum()/self.N/(self.N-1)

    def get_average_closeness_centrality(self):
        return np.mean(1.0/self.dist_mat.mean(axis=0))

    def get_average_eccentricity(self):
        return self.dist_mat.max(axis=0).mean()

    def get_diameter(self):
        return self.dist_mat.max()

    def get_radius(self):
        return self.dist_mat.min()

    def get_average_degree_centrality(self):
        return self.degs.mean()/(self.N-1)

    def __divide(self, arr, num_chunks):
        '''
        Divide array into specific number of chunks
        '''
        import itertools
        arr_c = iter(arr)
        while arr:
            x = tuple(itertools.islice(arr_c, num_chunks))
            if not x:
                return
            yield x

    def __get_average_betweenness_centrality_parallel(self):
        processes, nodes_per_process = 10, 5
        with mlp.Pool(processes=processes) as pool:
            nodes_per_run = len(pool._pool) * nodes_per_process
            chunks = list(self.__divide(self.G.nodes(), int(self.N / nodes_per_run)))

            num_chunks = len(chunks)
            bt_sc = pool.starmap(nx.betweenness_centrality_subset,
                zip([self.G] * num_chunks, chunks, [list(self.G)] * num_chunks,
                    [True] * num_chunks, ['cost'] * num_chunks,),)

        bt_c = bt_sc[0]
        for bt in bt_sc[1:]:
            for n in bt:
                bt_c[n] += bt[n]
        bt_c = np.array([v for v in bt_c.values()])
        self.cpd = np.sum(bt_c.max() - bt_c)/(self.N-1)
        return np.mean(bt_c)

    def __get_average_edge_betweenness_centrality_parallel(self):
        processes, nodes_per_process = 10, 5
        with mlp.Pool(processes=processes) as pool:
            nodes_per_run = len(pool._pool) * nodes_per_process
            chunks = list(self.__divide(self.G.nodes(), int(self.N / nodes_per_run)))

            num_chunks = len(chunks)
            bt_sc = pool.starmap(nx.edge_betweenness_centrality_subset,
                zip([self.G] * num_chunks, chunks, [list(self.G)] * num_chunks,
                    [True] * num_chunks, ['cost'] * num_chunks,),)

        bt_c = bt_sc[0]
        for bt in bt_sc[1:]:
            for n in bt:
                bt_c[n] += bt[n]
        return np.mean([v for v in bt_c.values()])

    def get_average_edge_betweenness_centrality(self):
        if self.N < 500:
            bcs = list(nx.algorithms.centrality.edge_betweenness_centrality(self.G,weight='cost').values())
            return np.mean(bcs)
        else:
            return self.__get_average_edge_betweenness_centrality_parallel()

    def get_average_node_betweenness_centrality(self):
        if self.nbc != np.inf:
            return self.nbc

        if self.N < 500:
            bcs = np.array(list(nx.algorithms.centrality.betweenness_centrality(self.G,weight='cost').values()))
            self.cpd = np.sum(bcs.max() - bcs)/(self.N-1)
            self.nbc = bcs.mean()
        else:
            self.nbc = self.__get_average_betweenness_centrality_parallel()
        return self.nbc

    def get_central_point_of_dominance(self):
        if self.cpd == np.inf:
            self.get_average_node_betweenness_centrality()
        return self.cpd

    def get_information_centrality(self):
        ics = nx.information_centrality(self.G,weight='weight')
        return np.mean([val for val in ics.values()])

    def get_degree_assortativity_coefficient(self):
        return nx.algorithms.assortativity.degree_assortativity_coefficient(self.G,weight='weight')

    def get_average_core_number(self):
        core_nums = list(nx.core_number(self.G).values())
        return np.mean(core_nums)

    def get_average_clustering_coefficient(self):
        return nx.average_clustering(self.G,weight='weight')

    def get_modularity_score(self):
        from community import best_partition
        from collections import defaultdict

        parts = best_partition(self.G,weight='weight')
        clusters = defaultdict(list)
        for i, cidx in enumerate(parts.values()):
            clusters[cidx].append(i)
        community = list(clusters.values())
        score = nx.community.quality.modularity(self.G,community,weight='weight')
        return score

    def get_transitivity(self):
        return nx.algorithms.cluster.transitivity(self.G)

    def get_local_efficiency(self):
        efficiency_list = [self.__efficiency(self.G.subgraph(self.G[v])) for v in self.G]
        return sum(efficiency_list) / self.N

    def get_global_efficiency(self):
        arr = self.dist_mat.flatten()
        denom = self.N * (self.N - 1)
        return sum(1.0/arr[arr > 0])/denom

    def __efficiency(self,G):
        n = len(G)
        denom, g_eff = n * (n - 1), 0
        if denom == 0:
            return g_eff

        from scipy.sparse.csgraph import shortest_path

        A = nx.to_scipy_sparse_matrix(G, weight='cost',dtype=np.float,format='csr')
        dists = shortest_path(A, method='J', directed=False).flatten()
        return sum(1.0/dists[dists > 0])/denom

    # Adjacency Spectral Measures
    def __get_adjacency_spectrum(self):
        if self.A is None:
            self.A = nx.to_scipy_sparse_matrix(self.G, weight='weight',dtype=np.float,format='csr')
        self.adj_lams = np.sort(eigsh(self.A, k=self.N-1, which='LA', return_eigenvectors=False))

    def get_spectral_radius(self):
        '''
        The spectral radius \lambda_1 of G is defined as the largest eigenvalue of the associated adjacency matrix A.
        The larger the spectral radius, the more robust the graph. It can be viewed from its close relationship
        to the “path” or “loop” capacity in G.
        '''
        if self.adj_lams is None:
            self.__get_adjacency_spectrum()
        return self.adj_lams[-1]

    def get_spectral_gap(self):
        '''
        The spectral gap \lambda_d = \lambda_1 - \lambda_2 is the difference between
        the largest and second largest eigenvalues of the adjacency matrix.
        It has an advantage over spectral radius since it accounts for undesirable bridges in G.
        '''
        if self.adj_lams is None:
            self.__get_adjacency_spectrum()
        return self.adj_lams[-1]-self.adj_lams[-2]

    def __spectral_scaling(self, K):
        '''
        Subgraph centrality (Estrada & Rodríguez-Velásquez, Phys. Rev. E, 2005)
        measures the centrality of a node by taking into account the number of
        subgraphs the node “participates” in.

        It is done by counting, for all k = 1, 2, ... the number of closed walks in
        G starting and ending at node i, with longer walks being penalized (given
        a smaller weight).

        Because A^k_{ii} is the number of closed walk of length k based at node i,
        the centrality can be formulated as a weighted summation

        SC_i = [\sum_k \alpha^k * A^k/k!]_{ii} = [e^{\alpha*A}]_{ii} \ge 1.

        The computing complexity is O(n^2). In real application, the relative ranking of nodes
        based on subgraph centrality matters more. Let \lambda_1\ge \lambda_2\ge ...\ge \lambda_n,
        be the eigenvalues of A and the associated normalized eigenvectors are \gamma_1,...,\gamma_n.
        Therefore, we have e^{\alpha*A} = \sum_k [e^{\alpha*\lambda_k} \gamma_k\gamma_k^T]. We divide
        it by e^{\alpha*\lambda_1} and have the normalized subgraph centrality:
        \hat SC_i = e^{-\alpha*\lambda_1}[e^{\alpha*A}]_{ii}=x_{1i}^2 +
        \sum_{k=2}^n e^{\alpha*(\lambda_k-\lambda_1)} x_{ki}^2.

        Spectral scaling is proposed to measure the good expansion (GE) of a network, i.e.,
        sparse and highly connected. A network G has GE properties if every subset S of nodes
        (up to 50% of the nodes) has a neighborhood that is larger than some “expansion factor”
        multiplied by the number of nodes in S. It is a combination of the spectral gap and
        subgraph centrality, and The smaller the better. A zero spectral scaling indicates
        a perfect good expansion character.

        Nodes in G are characterized according to the number of closed walks of odd (even) length
        containing the node. see Ernesto Estrada. Network robustness to targeted attacks -
        The interplay of expansibility and degree distribution.
        The European Physical Journal B-Condensed Matter and Complex Systems, 52(4):563–574, 2006.
        '''
        if self.A is None:
            self.A = nx.to_scipy_sparse_matrix(self.G, weight='weight',dtype=np.float,format='csr')

        np.random.seed(13)
        v0 = np.random.rand(self.A.shape[0])
        lams, gammas = eigsh(self.A, k=min(K,self.N-1), v0=v0, which='LM', return_eigenvectors=True)
        idx = np.abs(lams).argsort()[::-1]  # sort descending magnitude
        lams, gammas = lams[idx], gammas[:,idx]
        sinh_lams = np.sinh(lams)
        sc_odd = np.abs(np.matmul(gammas ** 2,sinh_lams))
        sc = np.log10(np.abs(gammas[:,0])) + 0.5 * np.log10(abs(sinh_lams[0])) + 0.5 * np.log10(sc_odd)
        sc = np.sqrt(sum(sc ** 2)/self.N)
        return sc

    def get_spectral_scaling(self):
        return self.__spectral_scaling(self.N-1)

    def get_generalized_robustness_index(self):
        '''
        A fast approximation of spectral scaling. It determine if G has many bridges (bad for robustness).
        '''
        return self.__spectral_scaling(30)

    def get_natural_connectivity(self):
        '''
        Natural connectivity has a physical and structural interpretation that is related to
        the connectivity properties of G, identifying alternative pathways in G through
        the weighted number of closed walks. The larger the natural connectivity (average eigenvalue
        of adjacency matrix), the more robust G.
        '''
        if self.adj_lams is None:
            self.__get_adjacency_spectrum()
        return np.log2(sum(np.exp(np.real(self.adj_lams)))/len(self.adj_lams))

    # Laplacian Spectral Measures
    def __get_laplacian_spectrum(self):
        self.L = nx.linalg.laplacianmatrix.laplacian_matrix(self.G,weight='weight')
        self.laplacian_lams = np.sort(eigsh(self.L,k=self.N-1,which='SM', tol=1e-8, return_eigenvectors=False))

    def get_algebraic_connectivity(self):
        '''
        The algebraic connectivity lambda2 of a connected undirected grpah is the second smallest eigenvalue
        of its Laplacian matrix. It has a close connection to the edge connectivity,
        where it serves as a lower bound: 0 < lambda2 < node connectivity < edge connectivity.
        It means that a network with larger algebraic connectivity is harder to disconnect.
        '''
        if self.laplacian_lams is None:
            self.__get_laplacian_spectrum()
        return self.laplacian_lams[1]

    def get_num_spanning_trees(self):
        '''
        The number of spanning trees T is the number of unique spanning trees that can be found in a graph.
        The larger the number of spanning trees, the more robust the graph. It can be viewed from the perspective
        of network connectivity, where a larger set of spanning trees means more alternative pathways in the network
        '''
        if self.laplacian_lams is None:
            self.__get_laplacian_spectrum()
        num_trees = np.prod(self.laplacian_lams[1:]) / self.N
        return num_trees

    def get_effective_resistance(self):
        '''
        A graph is viewed as an electrical circuit where an edge (i,j) corresponds to a resister of
        r_{ij} = 1 Ohm and a node i corresponds to a junction.
        The effective graph resistance R is the sum of resistances for all distinct pairs of vertices.
        The smaller the effective resistance, the more robust the graph.
        '''
        if self.laplacian_lams is None:
            self.__get_laplacian_spectrum()
        resistance = self.N * np.sum(1.0 / self.laplacian_lams[1:])
        return resistance

    def _computing_task(self,param):
        key, metric = param
        t1 = time()
        val = metric()
        t2 = time()
        print(f'{key}={val}, t={t2-t1:.3f}',flush=True)
        return (key, val)

    def get_feature_values(self,keys=None,ncpu=5):
        features = {'density':self.get_density,'resilience':self.get_resilience,
                    'heterogeneity':self.get_heterogeneity,'entropy':self.get_degree_entropy,
                    'wedge':self.get_wedge_count,'powerlaw_exponent':self.get_power_law_exponent,
                    'gini':self.get_gini_coefficient,
                    #'node_connectivity':self.get_average_node_connectivity,
                    #'edge_connectivity':self.get_average_edge_connectivity,
                    'shortest_path_length':self.get_average_shortest_path_length,
                    'closeness_centrality':self.get_average_closeness_centrality,
                    'eccentricity':self.get_average_eccentricity,'diameter':self.get_diameter,
                    'radius':self.get_radius,'degree_centrality':self.get_average_degree_centrality,
                    'edge_betweenness_centrality':self.get_average_edge_betweenness_centrality,
                     'node_betweenness_centrality':self.get_average_node_betweenness_centrality,
                     'central_point_of_dominance':self.get_central_point_of_dominance,
                    'information_centrality':self.get_information_centrality,
                     'degree_assortativity_coefficient':self.get_degree_assortativity_coefficient,
                     'core_number':self.get_average_core_number,
                    'clustering_coefficient':self.get_average_clustering_coefficient,
                     'modularity':self.get_modularity_score,'transitivity':self.get_transitivity,
                    # subgraph forming by the neighbors of nodes comes from two different layers, so no connection
                     #'local_efficiency':self.get_local_efficiency,
                    'global_efficiency':self.get_global_efficiency,
                    # adjacency matrix spectrum
                    'spectral_radius':self.get_spectral_radius,'spectral_gap':self.get_spectral_gap,
                    'natural_connectivity':self.get_natural_connectivity,'spectral_scaling':self.get_spectral_scaling,
                    # laplacian matrix spectrum
                    'algebraic_connectivity':self.get_algebraic_connectivity,
                    'number_spanning_trees':self.get_num_spanning_trees,
                    'effective_resistance':self.get_effective_resistance}

        keys = set(features.keys()) if keys is None else set(keys)
        lightweight = {'density','resilience','heterogeneity','entropy','wedge',
                       'powerlaw_exponent','gini','degree_centrality','degree_assortativity_coefficient',
                      'core_number','clustering_coefficient','modularity','transitivity','information_centrality',
                      'spectral_radius','spectral_gap','natural_connectivity','spectral_scaling',
                      'algebraic_connectivity','number_spanning_trees','effective_resistance',
                      'shortest_path_length','closeness_centrality','eccentricity','diameter','radius','global_efficiency'}

        heavyweight = {'edge_betweenness_centrality','node_betweenness_centrality','central_point_of_dominance'}

        rmv = lightweight - keys
        lightweight -= rmv
        rmv = heavyweight - keys
        heavyweight -= rmv
            
        entries = []
        if ncpu > 1:
            if lightweight:
                params = [(key, features[key]) for key in lightweight]
                ncpu = min(len(params),ncpu)

                with mlp.Pool(processes=ncpu) as pool:
                    res = pool.map(self._computing_task,params)
                    entries.extend(res)
        else:
            for key in lightweight:
                param = (key, features[key])
                res = self._computing_task(param)
                entries.append(res)

        '''
        heavyweight |= lightweight
        '''
        for key in heavyweight:
            param = (key, features[key])
            res = self._computing_task(param)
            entries.append(res)
        return np.array(sorted(entries,key=lambda entry: entry))

