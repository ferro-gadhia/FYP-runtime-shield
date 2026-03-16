from __future__ import annotations

from typing import Any, Dict, Final, List, Tuple
import json

import torch
from torch_geometric.data import Data

import networkx as nx
import matplotlib.pyplot as plt




### episode utilities
def output_episode_ids(filepath: str = "episodes.jsonl") -> List[int]:
    with open(filepath, "r", encoding="utf-8") as f:
        episode_ids = [json.loads(line)["episode_id"] for line in f]
    print(episode_ids)
    return episode_ids


def load_episode(episode_id: int, filepath: str = "episodes.jsonl") -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            episode = json.loads(line)
            if episode.get("episode_id") == episode_id:
                return episode
    raise ValueError(f"No episode found for id {episode_id}")


def infer_T(episode_dict: Dict[str, Any]) -> int:
    states_len = len(episode_dict["states_pre"])
    actions_len = len(episode_dict["actions"])
    history_len = len(episode_dict["history"])

    if states_len != actions_len or actions_len != history_len:
        raise ValueError(
            f"Length mismatch: states_pre={states_len}, actions={actions_len}, history={history_len}"
        )
    return history_len


###feature encoder.

def state_to_feat(state_summary):
    return [
        float(state_summary.get("num_files", 0)),
        float(state_summary.get("num_scanned_ips", 0)),
        float(state_summary.get("num_action_taken", 0)),
        float(state_summary.get("num_unsafe_so_far", 0)),
        float(state_summary.get("num_exfil_attempts_so_far", 0)),
        float(state_summary.get("num_secret_reads_so_far", 0)),
    ]

TYPE_ACTION: Final[List[str]] = ["read_file", "write_file", "scan_network"]


TYPE_ACTION: Final[List[str]] = ["read_file", "write_file", "scan_network"]

def history_to_action_feat(history_entry: Dict[str, Any]) -> List[float]:
    action_type = history_entry.get("action")
    if action_type not in TYPE_ACTION:
        raise ValueError(f"Unknown action type in history: {action_type!r}")

    one_hot = [0.0, 0.0, 0.0]
    idx = TYPE_ACTION.index(action_type)
    one_hot[idx] = 1.0

    path = str(history_entry.get("path", ""))
    target_range = str(history_entry.get("target_range", ""))

    touches_secret_path = 1.0 if path.startswith("/secret/") else 0.0
    touches_exfil_path = 1.0 if path.startswith("/exfil/") else 0.0
    exfil_flag = 1.0 if history_entry.get("exfiltration", False) else 0.0
    is_bad_scan_range = 1.0 if (action_type == "scan_network" and target_range != "10.0.0.0/24") else 0.0

    return one_hot + [
        exfil_flag,
        touches_secret_path,
        touches_exfil_path,
        is_bad_scan_range,
    ]

### graph structure

def build_edges(T: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []

    # node ids:
    # states: 0..T-1
    # actions: T..2T-1

    for t in range(T):
        S_t = t
        A_t = T + t
        edges.append((S_t, A_t))

    for t in range(T - 1):
        A_t = T + t
        A_next = T + t + 1
        edges.append((A_t, A_next))

    return edges

state_feature_dim = 6
action_feature_dim = 7
x_dim = 13
unlabelled = -1

def build_graph_from_episode(episode):
    T = infer_T(episode)
    
    history = episode["history"]
    states_pre = episode["states_pre"]
    
    num_nodes = 2*T

    node_types,action_mask, x_list, y_list = [],[],[],[]
    for t in range(T):
        node_types.append("state")
        action_mask.append(False)
        state_vec = state_to_feat(states_pre[t])
        full_vec = state_vec + [0.0] * action_feature_dim        
        x_list.append(full_vec)
        y_list.append(-1)

    for t in range(T):
        node_types.append("action")
        action_mask.append(True)

        act_vec = history_to_action_feat(history[t])
        full_vec = [0.0] * state_feature_dim + act_vec
        x_list.append(full_vec)
        safe = history[t]["safe"]
        y_list.append(1 if safe == False else 0)
    
    edges = []
    for t in range(T):
        S_t = t
        A_t = T + t
        edges.append((S_t, A_t))

    # A_t -> S_{t+1} for t = 0..T-2
    for t in range(T - 1):
        A_t = T + t
        S_next = t + 1
        edges.append((A_t, S_next))
    
    return {"T": T, "num_nodes": num_nodes, "node_types": node_types, "action_mask": action_mask, "x_list": x_list, "y_list": y_list, "edges": edges}

def build_pyg_graph_from_episode(episode):
    

    graph_dict = build_graph_from_episode(episode)

    x_list = graph_dict["x_list"]
    y_list = graph_dict["y_list"]
    edges = graph_dict["edges"]
    action_mask_list = graph_dict["action_mask"]

    # tensor creation
    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    action_mask = torch.tensor(action_mask_list, dtype=torch.bool)

    # pyg data object
    data = Data(x=x, edge_index=edge_index, y=y)
    data.action_mask = action_mask

    return data

def visualize_graph(graph_dict):
    """
    Visualise the episode graph using NetworkX.
    """

    G = nx.DiGraph()

    T = graph_dict["T"]
    node_types = graph_dict["node_types"]
    edges = graph_dict["edges"]

    # Add nodes with color by type
    for i, node_type in enumerate(node_types):
        G.add_node(i, type=node_type)

    # Add edges
    G.add_edges_from(edges)

    # Colour mapping
    colors = []
    for i in G.nodes():
        if node_types[i] == "state":
            colors.append("skyblue")
        else:
            colors.append("salmon")

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        node_color=colors,
        with_labels=True,
        node_size=700,
        font_size=8,
        arrows=True
    )

    plt.title("Episode Graph (State = Blue, Action = Red)")
    plt.show()

# quantitative tests

if __name__ == "__main__":
    ep = load_episode(0)

    graph_dict = build_graph_from_episode(ep)

    data = build_pyg_graph_from_episode(ep)

    print("Nodes:", data.x.shape[0])
    print("Edges:", data.edge_index.shape[1])
    print("Feature shape:", data.x.shape)
    print("Labels shape:", data.y.shape)

    print(data.x.shape)
    print(data.x[0])

    #visualize_graph(graph_dict)