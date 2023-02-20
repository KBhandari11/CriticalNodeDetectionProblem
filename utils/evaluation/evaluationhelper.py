from  utils.environment.envhelper import*

def input_graph(graph_path,file):
    GRAPH = Graph.Read_Ncol(graph_path+str(file)+".txt", directed=False)
    nodes = [v.index for v in GRAPH.vs]
    map = {n:int(i) for i, n in enumerate(nodes)}
    GRAPH = reset(GRAPH)  
    Graph.simplify(GRAPH)
    return GRAPH, map
# Given an environment and an trained agent we implement the agent
def EvaluateModel(env,objectiveFunction,nodeCentrality,globalFeature,trained_agent,GRAPH):
    """Evaluates `trained_agents` against a new graph."""
    time_step = env.reset(GRAPH,objectiveFunction,nodeCentrality,globalFeature)
    episode_rewards = []
    action_lists = []
    i = 0
    while not time_step.last():
        agent_output, prob = trained_agent.step(time_step, is_evaluation=True)
        action_lists.append(agent_output)
        time_step = env.step(agent_output)
        i+=1
        episode_rewards.append(env.get_state._rewards)
    reward1 = env.get_state.reward1
    return episode_rewards, reward1, action_lists

# Given an environmnet with all action in  a list 
'''def EvaluateACTION(env, action_list,GRAPH):
    """Evaluates the env for given action_list"""
    env.reset(GRAPH)
    episode_rewards = []
    i = 0
    for action in action_list:
        env.step([action,action])
        i+=1
        episode_rewards.append(env.get_state._rewards[0])
        if env.get_state._is_terminal == True:
            break
    lcc = env.get_state.lcc
    return episode_rewards, lcc, action_list[:len(lcc)]'''
def eval_network_dismantle(board,objectiveFunction, init_gamma):
    """Check if the current state of the graph meets the criteria:
            - Number of active node <= 2
            - Number of edges  <= 2
            - Fraction of the objective function is 1%.   
    """
    all_nodes = np.array(board.vs["active"])
    gamma = objectiveFunction(board)
    cond = True if (gamma/init_gamma) <= 0.01 else False
    return cond, gamma

def EvaluateACTION(action_list,objectiveFunction,GRAPH):
    """Evaluates the env for given action_list"""
    '''G = GRAPH.to_networkx()
    mapping = {}
    for i, j in zip(sorted(G), [sorted(G).index(i) for i in sorted(G)]):
            mapping[i] = j
    G = nx.relabel_nodes(G, mapping)
    print(G.nodes())'''
    gammaList = [objectiveFunction(GRAPH)]
    actionList = []
    for action in action_list:
        ebunch = GRAPH.incident(GRAPH.vs.find(name=str(action)))
        GRAPH.delete_edges(ebunch)
        cond, gamma = eval_network_dismantle(GRAPH,objectiveFunction, gammaList[0])
        gammaList.append(gamma)
        actionList.append(action)
        if cond:
            break
    return None, gammaList[0:GRAPH.vcount()], actionList