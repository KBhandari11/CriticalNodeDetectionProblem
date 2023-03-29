from  utils.environment.envhelper import*

def input_graph(graph_path,file):
    GRAPH = Graph.Read_Ncol(graph_path+str(file)+".txt", directed=False)
    nodes = [v.index for v in GRAPH.vs]
    map = {n:int(i) for i, n in enumerate(nodes)}
    GRAPH = reset(GRAPH)  
    Graph.simplify(GRAPH)
    return GRAPH, map
#Given an action list, with objective function for the NN agent. 
def EvaluateModel_LimitedAction(action_list,objectiveFunction,GRAPH,initialGamma):
    actionList = []
    gammaList = []
    for action in action_list:
        GRAPH.vs[action]["active"] = 0
        ebunch = GRAPH.incident(action)
        GRAPH.delete_edges(ebunch)
        cond, gamma = eval_network_dismantle(GRAPH,objectiveFunction, initialGamma)
        gammaList.append(gamma)
        actionList.append(action)
        if cond:
            break
    return GRAPH, gammaList, actionList, cond

# Given an environment and an trained agent we implement the agent
def EvaluateModel(env,objectiveFunction,nodeCentrality,globalFeature,trained_agent,GRAPH, useSingleStep= None):
    """Evaluates `trained_agents` against a new graph."""
    N = GRAPH.vcount()
    episode_rewards = []
    action_lists = []
    i = 0
    if useSingleStep != None:
        reward1 = [objectiveFunction(GRAPH)]
        cond = False
        while not cond:
            time_step = env.reset(GRAPH,objectiveFunction,nodeCentrality,globalFeature)
            _ , prob = trained_agent.step(time_step, is_evaluation=True)
            legal_actions = env._state._legal_actions() 
            #prob = [prob[p] if p in legal_actions else -100 for p in range(len(prob))]
            i = 0
            actions= []
            while(i < int(useSingleStep*N)) and (i < len(legal_actions)):
                argmax = np.argmax(prob)
                if argmax in legal_actions:
                    actions.append(argmax)
                    i+=1
                prob[argmax]=-100
            GRAPH, output, actionsUsed, cond = EvaluateModel_LimitedAction(actions,objectiveFunction,GRAPH,N)
            action_lists += actionsUsed
            reward1+=output
        return [], reward1, action_lists
    else:
        time_step = env.reset(GRAPH,objectiveFunction,nodeCentrality,globalFeature)
        while not time_step.last():
            agent_output, prob = trained_agent.step(time_step, is_evaluation=True)
            action_lists.append(agent_output)
            time_step = env.step(agent_output)
            i+=1
            episode_rewards.append(env.get_state._rewards)
        reward1 = env.get_state.reward1
        return episode_rewards, reward1, action_lists


def eval_network_dismantle(board,objectiveFunction, init_gamma):
    """Check if the current state of the graph meets the criteria:
            - Number of active node <= 2
            - Number of edges  <= 2
            - Fraction of the objective function is 1%.   
    """
    all_nodes = np.array(board.vs["active"])
    gamma = objectiveFunction(board)
    if objectiveFunction.__name__ == "numberConnectedComponent":
        N = board.vcount()
        cond = True if ((N-gamma)/N) <= 0.001 or (board.ecount() == 0)else False
    else:
        cond = True if (gamma/init_gamma) <= 0.001 else False
    return cond, gamma

def EvaluateACTION(action_list,objectiveFunction,GRAPH, useIndex):
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
        if useIndex:
            ebunch = GRAPH.incident(action)
        else:
            ebunch = GRAPH.incident(GRAPH.vs.find(name=str(action)))
        GRAPH.delete_edges(ebunch)
        cond, gamma = eval_network_dismantle(GRAPH,objectiveFunction, gammaList[0])
        gammaList.append(gamma)
        actionList.append(action)
        if cond:
            break
    return None, gammaList[0:GRAPH.vcount()], actionList