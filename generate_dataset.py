import json
import random

from sandbox_env import SandboxEnv
from agents import BenignAgent, AdversarialAgent
from episode_runner import run_episode


def generate(out_path: str = "episodes.jsonl", n_benign: int = 300, n_adversarial: int = 300, steps: int = 10, shuffle_seed: int = 42):
    episode_specs = ([("benign", i) for i in range(n_benign)] + [("adversarial", i) for i in range(n_adversarial)])

    rng = random.Random(shuffle_seed)
    rng.shuffle(episode_specs)

    episode_id = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for agent_type, local_idx in episode_specs:
            env = SandboxEnv()

            if agent_type == "benign":
                agent = BenignAgent(seed=1000 + local_idx)
            else:
                agent = AdversarialAgent(seed=2000 + local_idx)

            ep = run_episode(env, agent, max_steps=steps)
            ep["episode_id"] = episode_id
            ep["agent_type"] = agent_type

            f.write(json.dumps(ep) + "\n")
            episode_id += 1

    print(f"done written {episode_id} episodes to {out_path}")


if __name__ == "__main__":
    generate()