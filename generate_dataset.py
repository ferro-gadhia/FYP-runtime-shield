import json
from sandbox_env import SandboxEnv
from agents import BenignAgent, AdversarialAgent
from episode_runner import run_episode

def generate(out_path: str = "episodes.jsonl", n_benign: int = 50, n_adversarial: int = 50, steps: int = 10):    
    env = SandboxEnv()
    
    benign_agent = BenignAgent(seed=123)
    adversarial_agent = AdversarialAgent(seed = 456)

    episode_id = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for ep_id in range(n_benign):
            ep = run_episode(env, benign_agent, max_steps=steps)
            ep["episode_id"] = episode_id
            ep["agent_type"] = "benign"
            f.write(json.dumps(ep) + "\n")
            episode_id+=1

        for ep_id in range(n_adversarial):
            ep = run_episode(env, adversarial_agent, max_steps=steps)
            ep["episode_id"] = episode_id
            ep["agent_type"] = "adversarial"
            f.write(json.dumps(ep) + "\n")
            episode_id+=1

    print(f"done written{episode_id} episodes to {out_path}")


if __name__ == "__main__":
    generate()