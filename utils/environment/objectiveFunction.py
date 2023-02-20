import igraph

class objectiveFunction:

    # different Objective Functions 

    def get_connectedComponents(self,G):
        found = set()
        comps = []
        for v in G.vs.indices:
            if v not in found:
                connected = G.bfs(v)[0]
                found.update(connected)
                comps.append(connected)
        return comps
            
    def pairwiseConnectivity(self, G):
        comps =  self.get_connectedComponents(G)
        return sum(map((lambda x: len(x)*(len(x)-1)/2), comps))

    def largestConnectedComponent(self, G):
        comps =  self.get_connectedComponents(G)
        return len(max(comps, key=len))   

    def numberConnectedComponent(self, G):
        comps =  self.get_connectedComponents(G)
        return  len(comps)  