from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal
import random
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

OutcomeType = Literal["tp", "fp", "tn", "fn"]

DEFAULT_THRESHOLD = 0.50
MIN_THRESHOLD = 0.25
MAX_THRESHOLD = 0.90
THRESHOLD_STEP = 0.01
RECENT_WINDOW_SIZE = 50
TARGET_FP_RATE = 0.15
TARGET_FN_RATE = 0.10
MISTAKE_BUFFER_SIZE = 100
RETRAIN_TRIGGER = 10
RETRAIN_STEPS = 3
RETRAIN_BATCH_SIZE = 8
RETRAIN_LR = 1e-4


@dataclass
class MistakeExample:
    graph: Data
    candidate_node_index: int
    true_label: int              # 1 = unsafe, 0 = safe
    outcome: OutcomeType         # fp or fn
    p_unsafe: float
    threshold: float


@dataclass
class HealingState:
    threshold: float = DEFAULT_THRESHOLD
    recent_outcomes: Deque[OutcomeType] = field(
        default_factory=lambda: deque(maxlen=RECENT_WINDOW_SIZE)
    )
    mistake_buffer: Deque[MistakeExample] = field(
        default_factory=lambda: deque(maxlen=MISTAKE_BUFFER_SIZE)
    )
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0


def init_healing_state(initial_threshold: float = DEFAULT_THRESHOLD) -> HealingState:
    return HealingState(threshold=initial_threshold)


def classify_outcome(predicted_block: bool, ground_truth_safe: bool) -> OutcomeType:
    """
    Shield-centric confusion outcomes:
      tp = blocked an unsafe action
      fp = blocked a safe action
      tn = allowed a safe action
      fn = allowed an unsafe action
    """
    if predicted_block and not ground_truth_safe:
        return "tp"
    if predicted_block and ground_truth_safe:
        return "fp"
    if (not predicted_block) and ground_truth_safe:
        return "tn"
    return "fn"


def recent_rates(healing_state: HealingState) -> Dict[str, float]:
    recent = list(healing_state.recent_outcomes)
    if not recent:
        return {
            "recent_fp_rate": 0.0,
            "recent_fn_rate": 0.0,
            "recent_tp": 0,
            "recent_fp": 0,
            "recent_tn": 0,
            "recent_fn": 0,
        }

    recent_tp = sum(x == "tp" for x in recent)
    recent_fp = sum(x == "fp" for x in recent)
    recent_tn = sum(x == "tn" for x in recent)
    recent_fn = sum(x == "fn" for x in recent)

    fp_denom = recent_fp + recent_tn
    fn_denom = recent_fn + recent_tp

    recent_fp_rate = recent_fp / fp_denom if fp_denom else 0.0
    recent_fn_rate = recent_fn / fn_denom if fn_denom else 0.0

    return {
        "recent_fp_rate": recent_fp_rate,
        "recent_fn_rate": recent_fn_rate,
        "recent_tp": recent_tp,
        "recent_fp": recent_fp,
        "recent_tn": recent_tn,
        "recent_fn": recent_fn,
    }


def record_runtime_outcome(
    healing_state: HealingState,
    predicted_block: bool,
    ground_truth_safe: bool,
    graph: Data,
    candidate_node_index: int,
    p_unsafe: float,
) -> OutcomeType:
    outcome = classify_outcome(predicted_block, ground_truth_safe)
    healing_state.recent_outcomes.append(outcome)

    if outcome == "tp":
        healing_state.tp += 1
    elif outcome == "fp":
        healing_state.fp += 1
    elif outcome == "tn":
        healing_state.tn += 1
    else:
        healing_state.fn += 1

    if outcome in ("fp", "fn"):
        true_label = 0 if ground_truth_safe else 1
        # Store CPU copy so the buffer stays lightweight and device-agnostic.
        stored_graph = graph.cpu()
        healing_state.mistake_buffer.append(
            MistakeExample(
                graph=stored_graph,
                candidate_node_index=candidate_node_index,
                true_label=true_label,
                outcome=outcome,
                p_unsafe=float(p_unsafe),
                threshold=float(healing_state.threshold),
            )
        )

    return outcome


def update_threshold(
    healing_state: HealingState,
    target_fp_rate: float = TARGET_FP_RATE,
    target_fn_rate: float = TARGET_FN_RATE,
    min_threshold: float = MIN_THRESHOLD,
    max_threshold: float = MAX_THRESHOLD,
    step_size: float = THRESHOLD_STEP,
) -> float:
    rates = recent_rates(healing_state)
    threshold = healing_state.threshold

    # Prioritise reducing false negatives first because unsafe actions slipping
    # through are worse than blocking a safe action in this project setting.
    if rates["recent_fn_rate"] > target_fn_rate:
        threshold = max(min_threshold, threshold - step_size)
    elif rates["recent_fp_rate"] > target_fp_rate:
        threshold = min(max_threshold, threshold + step_size)

    healing_state.threshold = threshold
    return threshold


def maybe_fine_tune_on_mistakes(
    model: torch.nn.Module,
    healing_state: HealingState,
    device: torch.device,
    retrain_trigger: int = RETRAIN_TRIGGER,
    retrain_steps: int = RETRAIN_STEPS,
    retrain_batch_size: int = RETRAIN_BATCH_SIZE,
    retrain_lr: float = RETRAIN_LR,
) -> bool:
    """
    Lightweight online fine-tuning on buffered mistakes.
    Returns True if fine-tuning happened, else False.
    """
    if len(healing_state.mistake_buffer) < retrain_trigger:
        return False

    optimizer = torch.optim.Adam(model.parameters(), lr=retrain_lr)
    model.train()

    buffer_list: List[MistakeExample] = list(healing_state.mistake_buffer)

    for _ in range(retrain_steps):
        batch_examples = random.sample(
            buffer_list,
            k=min(retrain_batch_size, len(buffer_list)),
        )

        total_loss = 0.0
        for example in batch_examples:
            graph = example.graph.to(device)
            logits = model(graph.x, graph.edge_index)

            candidate_logit = logits[example.candidate_node_index].view(1)
            true_label = torch.tensor(
                [float(example.true_label)],
                dtype=torch.float32,
                device=device,
            )

            loss = F.binary_cross_entropy_with_logits(candidate_logit, true_label)
            total_loss = total_loss + loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    return True