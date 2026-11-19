"""Small checkpoint helpers shared by the original and multi-scale training paths."""

import random

import numpy as np
import torch


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def parse_feature_paths(entries):
    """Parse repeated dataset=path arguments without imposing a new config format."""
    paths = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError("Feature paths must use DATASET=/absolute/path.h5")
        dataset, path = entry.split("=", 1)
        paths[dataset] = path
    return paths
