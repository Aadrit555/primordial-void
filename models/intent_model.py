"""
models/intent_model.py
-----------------------
Primordial Void — Intent Model

A 3-layer MLP trained via behavioral cloning on correct (intended) trajectories.
This is the proxy for "what the designer meant."

Architecture:
  Input  → 64 neurons (ReLU) → 64 neurons (ReLU) → action_dim (logits)

Key methods:
  train_on_trajectories(trajectories)  → trains the model
  get_action_probs(observation)        → returns softmax distribution over actions
  get_log_probs(observation, action)   → returns log prob of a specific action
  save(path) / load(path)              → persist weights
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class IntentModel(nn.Module):

    def __init__(self, obs_dim: int = 66, action_dim: int = 4, hidden_size: int = 64):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # 3-layer MLP
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            # No final activation — raw logits
        )

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (batch, action_dim)."""
        return self.network(x)

    # ------------------------------------------------------------------ #
    #  Inference helpers
    # ------------------------------------------------------------------ #
    def get_action_probs(self, observation: np.ndarray) -> np.ndarray:
        """
        Given a single observation (numpy array of shape (obs_dim,)),
        return a probability distribution over actions as numpy array of shape (action_dim,).
        """
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            logits = self.forward(obs_tensor)
            probs  = torch.softmax(logits, dim=-1)
        return probs.squeeze(0).numpy()

    def get_log_probs(self, observation: np.ndarray, action: int) -> float:
        """
        Returns the log probability of taking `action` from `observation`
        under the intent model's distribution.
        """
        probs = self.get_action_probs(observation)
        epsilon = 1e-8
        return float(np.log(probs[action] + epsilon))

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train_on_trajectories(
        self,
        trajectories: list,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> list:
        """
        Train via behavioral cloning.

        trajectories: list of trajectories, each a list of (observation, action) tuples.
        Returns list of per-epoch loss values.
        """
        # Flatten all (obs, action) pairs
        all_obs     = []
        all_actions = []
        for traj in trajectories:
            for (obs, action) in traj:
                all_obs.append(obs)
                all_actions.append(action)

        X = torch.FloatTensor(np.array(all_obs))
        y = torch.LongTensor(np.array(all_actions))

        dataset    = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer  = optim.Adam(self.parameters(), lr=learning_rate)
        criterion  = nn.CrossEntropyLoss()

        self.train()
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_obs, batch_actions in dataloader:
                optimizer.zero_grad()
                logits = self.forward(batch_obs)
                loss   = criterion(logits, batch_actions)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            loss_history.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} — Loss: {avg_loss:.4f}")

        return loss_history

    # ------------------------------------------------------------------ #
    #  Accuracy evaluation
    # ------------------------------------------------------------------ #
    def evaluate_accuracy(self, trajectories: list) -> float:
        """Returns action prediction accuracy on a set of trajectories."""
        self.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for traj in trajectories:
                for (obs, action) in traj:
                    probs     = self.get_action_probs(obs)
                    predicted = int(np.argmax(probs))
                    if predicted == action:
                        correct += 1
                    total += 1
        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "obs_dim":    self.obs_dim,
            "action_dim": self.action_dim,
        }, path)
        print(f"Intent model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "IntentModel":
        checkpoint = torch.load(path, map_location="cpu")
        model = cls(
            obs_dim    = checkpoint["obs_dim"],
            action_dim = checkpoint["action_dim"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        print(f"Intent model loaded ← {path}")
        return model