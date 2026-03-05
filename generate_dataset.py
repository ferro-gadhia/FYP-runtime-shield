import json
from sandbox_env import SandboxEnv
from agents import BenignAgent
from episode_runner import run_episode

def generate(out_path: str = "episodes.jsonl", n_episodes: int = 50, steps: int = 10):
    env = SandboxEnv()
    agent = BenignAgent(seed=123)

    with open(out_path, "w", encoding="utf-8") as f:
        for ep_id in range(n_episodes):
            ep = run_episode(env, agent, max_steps=steps)
            ep["episode_id"] = ep_id
            ep["agent_type"] = "benign"
            f.write(json.dumps(ep) + "\n")

    print("done")

generate()