from  utils.environment.envhelper import*

# Given an environment and an trained agent we implement the agent
def EvaluateModel(env, trained_agent,GRAPH):
    """Evaluates `trained_agents` against a new graph."""
    time_step = env.reset(GRAPH)
    episode_rewards = []
    action_lists = []
    i = 0
    while not time_step.last():
        agent_output, prob = trained_agent.step(time_step, is_evaluation=True)
        action_lists.append(agent_output)
        time_step = env.step(agent_output)
        i+=1
        episode_rewards.append(env.get_state._rewards)
    lcc = env.get_state.lcc
    return episode_rewards, lcc, action_lists

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
def eval_network_dismantle(graph, init_lcc):
    largest_cc = get_lcc(graph)
    cond = True if (largest_cc/init_lcc) <= 0.1 else False
    return cond, largest_cc

def EvaluateACTION(action_list,GRAPH):
    """Evaluates the env for given action_list"""
    lcc = [get_lcc(GRAPH)]
    act = []
    for action in action_list:
        ebunch = GRAPH.incident(GRAPH.vs.find(name=str(action)))
        GRAPH.delete_edges(ebunch)
        cond, l = eval_network_dismantle(GRAPH, lcc[0])
        lcc.append(l)
        act.append(action)
        if cond:
            break
    return None, lcc[0:GRAPH.vcount()], act