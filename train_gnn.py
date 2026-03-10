from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from graph_builder import build_pyg_graph_from_episode
from gnn_model import SafetyGCN

DATA_PATH = "episodes.jsonl"
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3

def load_all_graphs(filepath: str = DATA_PATH):
    graphs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            episode = json.loads(line)
            data = build_pyg_graph_from_episode(episode)
            graphs.append(data)
    return graphs

def split_graphs(graphs, train_ratio: float = 0.8):
    split_idx = int(len(graphs) * train_ratio)
    train_graphs = graphs[:split_idx]
    test_graphs = graphs[split_idx:]
    return train_graphs, test_graphs



def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_action_nodes = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index)

        mask = batch.action_mask
        labels = batch.y[mask].float()
        masked_logits = logits[mask]

        loss = F.binary_cross_entropy_with_logits(masked_logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * int(mask.sum().item())
        total_action_nodes += int(mask.sum().item())

    return total_loss / max(total_action_nodes, 1)


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()

    total = 0
    correct = 0
    tp = fp = tn = fn = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)

        mask = batch.action_mask
        probs = torch.sigmoid(logits[mask])
        preds = (probs >= threshold).long()
        labels = batch.y[mask].long()

        total += labels.numel()
        correct += int((preds == labels).sum().item())

        tp += int(((preds == 1) & (labels == 1)).sum().item())
        fp += int(((preds == 1) & (labels == 0)).sum().item())
        tn += int(((preds == 0) & (labels == 0)).sum().item())
        fn += int(((preds == 0) & (labels == 1)).sum().item())

    accuracy = correct / total if total else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_action_nodes": total,
    }


def main():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run: python generate_dataset.py"
        )

    graphs = load_all_graphs(DATA_PATH)
    if len(graphs) < 2:
        raise ValueError("Need at least 2 graphs to train/test split.")

    train_graphs, test_graphs = split_graphs(graphs)

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SafetyGCN(in_channels=7, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Loaded {len(graphs)} graphs")
    print(f"Train graphs: {len(train_graphs)} | Test graphs: {len(test_graphs)}")
    print(f"Device: {device}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, test_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"FPR={metrics['false_positive_rate']:.4f} | "
            f"FNR={metrics['false_negative_rate']:.4f}"
        )

    final_metrics = evaluate(model, test_loader, device)
    print("\nFinal evaluation:")
    for key, value in final_metrics.items():
        print(f"{key}: {value}")

    torch.save(model.state_dict(), "safety_gcn.pt")
    print("\nSaved model to safety_gcn.pt")


if __name__ == "__main__":
    main()