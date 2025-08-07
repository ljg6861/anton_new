"""
Handles persisting and loading learning data.
"""
import json
import os
import pickle
from typing import Dict, Any

from server.agent.learning_loop import LearningLoop

LEARNING_DATA_PATH = "data/learning/learning_data.pkl"
METRICS_DATA_PATH = "data/learning/metrics.json"


def save_learning_data(learning_loop: LearningLoop) -> None:
    """Saves the learning loop data to disk."""
    os.makedirs(os.path.dirname(LEARNING_DATA_PATH), exist_ok=True)

    # Save experiences using pickle (binary format is more efficient for large objects)
    with open(LEARNING_DATA_PATH, 'wb') as f:
        pickle.dump(learning_loop.experiences, f)

    # Save metrics as JSON for easier inspection
    with open(METRICS_DATA_PATH, 'w') as f:
        json.dump(learning_loop.performance_metrics, f)


def load_learning_data(learning_loop: LearningLoop) -> None:
    """Loads learning data from disk into the learning loop."""
    try:
        if os.path.exists(LEARNING_DATA_PATH):
            with open(LEARNING_DATA_PATH, 'rb') as f:
                learning_loop.experiences = pickle.load(f)

        if os.path.exists(METRICS_DATA_PATH):
            with open(METRICS_DATA_PATH, 'r') as f:
                learning_loop.performance_metrics = json.load(f)

    except Exception as e:
        print(f"Error loading learning data: {e}")
        # Initialize with empty data
        learning_loop.experiences = []
        learning_loop.performance_metrics = {
            "success_rate": [],
            "task_duration": [],
            "steps_taken": []
        }