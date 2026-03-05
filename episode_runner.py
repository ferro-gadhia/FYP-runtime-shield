from __future__ import annotations
from typing import Any, Dict, List

def run_episode(env, agent, max_steps:int = 10) -> Dict[str, Any]:
    env.reset()
    actions: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    states_pre: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        #state before action
        states_pre.append(env.get_state_summary())
        #agent proposes action
        action = agent.choose_action(env) #env executes (truth + observation)
        #store result
        actions.append(action)

        observation, safe = env.execute(action)
        results.append({"observation": observation, "safe": safe})


    return {
        "actions": actions,
        "results": results,
        "states_pre": states_pre,
        "history": env.history
    }